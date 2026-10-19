"""
优化的mDPO数据生成脚本 (修复版)
修复了Codex指出的问题：
1. --no-format-augmentation 现在真正生效
2. 正确处理数据过滤逻辑
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

# 导入现有的工具函数
from preference_utils import (
    build_planning_prompt,
    build_preference_pair_id,
    derive_task_metadata,
    render_plan_response,
    render_oracle_reasoning_trace,
)


# ============================================================================
# 优化的Margin配置（关键改动）
# ============================================================================
OPTIMIZED_MARGIN_BUCKETS = [
    {"name": "none", "min_score": 0.0, "max_score": 0.0, "margin": 0.0},
    {"name": "light", "min_score": 0.0, "max_score": 5.0, "margin": 0.05},
    {"name": "moderate", "min_score": 5.0, "max_score": 10.0, "margin": 0.15},
    {"name": "severe", "min_score": 10.0, "max_score": 15.0, "margin": 0.3},
    {"name": "critical", "min_score": 15.0, "max_score": float("inf"), "margin": 0.5},
]

OPTIMIZED_SOFT_UTILITY_BUCKETS = [
    {"name": "none", "min_gap": 0.0, "max_gap": 0.0, "margin": 0.0},
    {"name": "light", "min_gap": 0.0, "max_gap": 1.0, "margin": 0.05},
    {"name": "moderate", "min_gap": 1.0, "max_gap": 3.0, "margin": 0.15},
    {"name": "severe", "min_gap": 3.0, "max_gap": 6.0, "margin": 0.3},
    {"name": "critical", "min_gap": 6.0, "max_gap": float("inf"), "margin": 0.5},
]

MAJOR_ERROR_TYPES = {
    "capability_mismatch",
    "morphology_constraint",
    "graph_infeasible",
    "coordination_failure",
    "terminal_failure",
}
MINOR_ERROR_TYPES = {"resource_conflict"}
VALID_PAIR_TYPES = {"hard_invalid", "soft_suboptimal"}


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def dump_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _pick_margin_bucket(severity_score: float, buckets: List[Dict]) -> Dict[str, Any]:
    if severity_score <= 0:
        return buckets[0]
    for bucket in buckets[1:]:
        if bucket["min_score"] <= severity_score < bucket["max_score"]:
            return bucket
    return buckets[-1]


def _pick_soft_utility_bucket(utility_gap: float, buckets: List[Dict]) -> Dict[str, Any]:
    if utility_gap <= 0:
        return buckets[0]
    for bucket in buckets[1:]:
        if bucket["min_gap"] <= utility_gap < bucket["max_gap"]:
            return bucket
    return buckets[-1]


def compute_optimized_preference_strength(
    chosen_feedback: Dict[str, Any],
    rejected_feedback: Dict[str, Any],
    *,
    pair_type: str = "hard_invalid",
    chosen_utility: float | None = None,
    rejected_utility: float | None = None,
) -> Dict[str, Any]:
    """使用优化的margin计算偏好强度"""
    chosen_penalty = float(chosen_feedback.get("penalty_score", 0.0) or 0.0)
    rejected_penalty = float(rejected_feedback.get("penalty_score", 0.0) or 0.0)

    breakdown = rejected_feedback.get("violation_breakdown", {}) or {}
    if not isinstance(breakdown, dict):
        breakdown = {}

    error_counter = Counter(breakdown)
    major_count = sum(count for name, count in error_counter.items() if name in MAJOR_ERROR_TYPES)
    minor_count = sum(count for name, count in error_counter.items() if name in MINOR_ERROR_TYPES)
    total_count = sum(error_counter.values())

    chosen_cost = float(chosen_feedback.get("trajectory_cost", 0.0) or 0.0)
    rejected_cost = float(rejected_feedback.get("trajectory_cost", 0.0) or 0.0)

    norm_pair_type = pair_type if isinstance(pair_type, str) and pair_type else "hard_invalid"
    chosen_utility_value = float(chosen_utility if chosen_utility is not None else chosen_cost)
    rejected_utility_value = float(rejected_utility if rejected_utility is not None else rejected_cost)
    utility_gap = max(0.0, rejected_utility_value - chosen_utility_value)

    # 使用优化的margin buckets
    if norm_pair_type == "soft_suboptimal":
        severity_score = utility_gap
        bucket = _pick_soft_utility_bucket(utility_gap, OPTIMIZED_SOFT_UTILITY_BUCKETS)
        severity_source = "utility_gap"
    else:
        severity_score = max(0.0, rejected_penalty - chosen_penalty)
        bucket = _pick_margin_bucket(severity_score, OPTIMIZED_MARGIN_BUCKETS)
        severity_source = "penalty_gap"

    return {
        "pair_type": norm_pair_type,
        "chosen_penalty_score": chosen_penalty,
        "rejected_penalty_score": rejected_penalty,
        "severity_score": severity_score,
        "severity_source": severity_source,
        "severity_level": bucket["name"],
        "severity_margin": bucket["margin"],
        "major_violation_count": int(major_count),
        "minor_violation_count": int(minor_count),
        "total_violation_count": int(total_count),
        "chosen_trajectory_cost": chosen_cost,
        "rejected_trajectory_cost": rejected_cost,
        "trajectory_cost_gap": max(0.0, rejected_cost - chosen_cost),
        "chosen_utility": round(chosen_utility_value, 6),
        "rejected_utility": round(rejected_utility_value, 6),
        "utility_gap": round(utility_gap, 6),
        "error_types": list(rejected_feedback.get("error_type", [])),
        "violation_breakdown": dict(breakdown),
    }


def is_format_augmentation(sample: Dict[str, Any]) -> bool:
    """检查样本是否是format增强生成的"""
    # 检查generation_metadata中的phase4_augmentation字段
    gen_meta = sample.get("generation_metadata", {})
    if isinstance(gen_meta, dict) and gen_meta.get("phase4_augmentation"):
        return True
    
    # 检查augmentation_type字段
    if sample.get("augmentation_type"):
        return True
    
    # 检查target_errors中是否包含format_failure
    target_errors = sample.get("target_errors", [])
    if "format_failure" in target_errors:
        return True
    
    # 检查rejected_feedback中的validation_source
    rejected_feedback = sample.get("rejected", {}).get("feedback", {})
    if rejected_feedback.get("validation_source") == "synthetic_format_augmentation":
        return True
    
    return False


def build_training_record(sample: Dict[str, Any], filter_format_aug: bool = False) -> Dict[str, Any] | None:
    """
    构建训练记录
    
    Args:
        sample: 原始样本
        filter_format_aug: 是否过滤format增强样本
    
    Returns:
        训练记录，如果需要过滤则返回None
    """
    # 如果启用过滤且是format增强样本，返回None
    if filter_format_aug and is_format_augmentation(sample):
        return None
    
    # 构建prompt
    local_graph = sample.get("local_graph")
    if isinstance(local_graph, dict):
        prompt = build_planning_prompt(
            instruction=sample["instruction"],
            scenario_json=local_graph,
            target_item=sample["target_item"],
            item_profile=sample.get("item_profile", {}),
            start_node=sample["start_node"],
            end_node=sample["end_node"],
            reference_paths=sample.get("reference_paths", []),
        )
    else:
        prompt = sample["prompt"]
    
    # 构建chosen response
    chosen = sample["chosen"]
    reasoning_trace = chosen.get("reasoning_trace")
    if isinstance(reasoning_trace, dict) and reasoning_trace:
        thought = render_oracle_reasoning_trace(reasoning_trace)
    else:
        thought = (chosen.get("thought") or "").strip()
    chosen_response = render_plan_response(thought, chosen.get("plan", []))
    
    # 构建rejected response
    rejected = sample["rejected"]
    rejected_response = rejected.get("response_text", "")
    if not rejected_response:
        rejected_thought = (rejected.get("thought") or "").strip()
        rejected_response = render_plan_response(rejected_thought, rejected.get("plan", []))
    
    # 获取feedback
    chosen_feedback = chosen["feedback"]
    rejected_feedback = rejected["feedback"]
    
    # 规范化pair_type
    pair_type = sample.get("pair_type", "hard_invalid")
    if pair_type not in VALID_PAIR_TYPES:
        pair_type = "hard_invalid"
    
    # 计算utility
    chosen_utility = float(sample.get("chosen_utility") or chosen_feedback.get("trajectory_cost", 0.0) or 0.0)
    rejected_utility = float(sample.get("rejected_utility") or rejected_feedback.get("trajectory_cost", chosen_utility) or chosen_utility)
    utility_gap = round(rejected_utility - chosen_utility, 6)
    
    # 计算优化的偏好强度
    preference_strength = compute_optimized_preference_strength(
        chosen_feedback,
        rejected_feedback,
        pair_type=pair_type,
        chosen_utility=chosen_utility,
        rejected_utility=rejected_utility,
    )
    
    # 获取task metadata
    task_metadata = sample.get("task_metadata") or derive_task_metadata(
        sample["start_node"],
        sample["end_node"],
        sample["target_item"],
    )
    
    # 构建pair_id
    pair_id = sample.get("pair_id") or build_preference_pair_id(
        prompt,
        chosen["plan"],
        rejected["plan"],
    )
    
    # 提取selected_robot
    selected_robot = (
        (chosen.get("reasoning_trace") or {}).get("selection", {}).get("selected_robot")
        or (chosen["plan"][0].get("robot") if chosen.get("plan") else "unknown")
    )
    
    return {
        "pair_id": pair_id,
        "task_id": sample.get("task_id"),
        "pair_type": pair_type,
        "dataset_schema_version": sample.get("dataset_schema_version", "pvr_dpo_v2"),
        "prompt_version": sample.get("prompt_version", "graph_prompt_v3"),
        "selected_robot": selected_robot,
        "chosen_valid": bool(sample.get("chosen_valid", chosen_feedback.get("is_valid", False))),
        "rejected_valid": bool(sample.get("rejected_valid", rejected_feedback.get("is_valid", False))),
        "chosen_utility": round(chosen_utility, 6),
        "rejected_utility": round(rejected_utility, 6),
        "utility_gap": utility_gap,
        "prompt": prompt,
        "chosen": chosen_response,
        "rejected": rejected_response,
        "margin": preference_strength["severity_margin"],
        "preference_strength": preference_strength,
        "instruction": sample["instruction"],
        "target_errors": sample.get("target_errors", []),
        "task": {
            "start_node": sample["start_node"],
            "end_node": sample["end_node"],
            "target_item": sample["target_item"],
            "item_profile": sample.get("item_profile", {}),
            "task_metadata": task_metadata,
            "local_graph": sample.get("local_graph"),
            "reference_paths": sample.get("reference_paths", []),
        },
        "chosen_plan": chosen["plan"],
        "rejected_plan": rejected["plan"],
        "chosen_feedback": chosen_feedback,
        "chosen_reasoning_trace": chosen.get("reasoning_trace"),
        "rejected_feedback": rejected_feedback,
        "generation_metadata": sample.get("generation_metadata", {}),
        "augmentation_type": sample.get("augmentation_type"),
        "exclude_from_sft": sample.get("exclude_from_sft", False),
    }


def split_data(
    records: List[Dict[str, Any]],
    *,
    split_by: str = "task",
    stratify_by: str = "robot",
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """分割数据集"""
    import random
    random.seed(seed)
    
    # 按split_by分组
    groups = defaultdict(list)
    for record in records:
        if split_by == "task":
            key = record["task"]["task_metadata"]["task_signature"]
        elif split_by == "item":
            key = record["task"]["task_metadata"]["target_item"]
        elif split_by == "building_pair":
            key = record["task"]["task_metadata"]["building_pair"]
        elif split_by == "random":
            key = record["pair_id"]
        else:
            key = record["pair_id"]
        groups[key].append(record)
    
    # 打乱组
    group_keys = list(groups.keys())
    random.shuffle(group_keys)
    
    # 分割
    total = len(group_keys)
    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)
    
    train_keys = group_keys[:train_end]
    val_keys = group_keys[train_end:val_end]
    test_keys = group_keys[val_end:]
    
    train_records = [r for k in train_keys for r in groups[k]]
    val_records = [r for k in val_keys for r in groups[k]]
    test_records = [r for k in test_keys for r in groups[k]]
    
    return train_records, val_records, test_records


def generate_statistics(records: List[Dict[str, Any]], split_name: str) -> Dict[str, Any]:
    """生成统计信息"""
    if not records:
        return {}
    
    pair_types = Counter(r["pair_type"] for r in records)
    margins = Counter(str(r["margin"]) for r in records)
    severity_levels = Counter(r["preference_strength"]["severity_level"] for r in records)
    robots = Counter(r["selected_robot"] for r in records)
    
    # 错误类型统计
    error_types = Counter()
    for r in records:
        for error in r.get("target_errors", []):
            error_types[error] += 1
    
    # Task profile统计
    task_profiles = Counter()
    for r in records:
        task_meta = r["task"]["task_metadata"]
        # 简单的profile分类
        start_building = task_meta.get("start_building", "")
        end_building = task_meta.get("end_building", "")
        if start_building == end_building:
            profile = "same_building"
        else:
            profile = "cross_building"
        task_profiles[profile] += 1
    
    return {
        "count": len(records),
        "pair_type_distribution": dict(pair_types),
        "margin_distribution": dict(margins),
        "severity_distribution": dict(severity_levels),
        "robot_distribution": dict(robots),
        "error_type_distribution": dict(error_types.most_common(20)),
        "task_profile_distribution": dict(task_profiles),
    }


def main():
    parser = argparse.ArgumentParser(description="生成优化的mDPO训练数据")
    parser.add_argument(
        "--input",
        type=str,
        default="dpo_dataset_filtered.jsonl",
        help="输入文件路径",
    )
    parser.add_argument(
        "--output-prefix",
        type=str,
        default="dpo_mdpo_optimized",
        help="输出文件前缀",
    )
    parser.add_argument(
        "--split-by",
        type=str,
        default="task",
        choices=["task", "item", "building_pair", "random"],
        help="数据分割方式",
    )
    parser.add_argument(
        "--stratify-by",
        type=str,
        default="robot",
        help="分层采样依据",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.8,
        help="训练集比例",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.1,
        help="验证集比例",
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.1,
        help="测试集比例",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="随机种子",
    )
    parser.add_argument(
        "--no-format-augmentation",
        action="store_true",
        help="禁用format_failure增强（推荐）",
    )
    args = parser.parse_args()

    print("=" * 80)
    print("优化的mDPO数据生成 (修复版)")
    print("=" * 80)
    print(f"输入文件: {args.input}")
    print(f"输出前缀: {args.output_prefix}")
    print(f"分割方式: split_by={args.split_by}, stratify_by={args.stratify_by}")
    print(f"比例: train={args.train_ratio}, val={args.val_ratio}, test={args.test_ratio}")
    print(f"Format增强: {'禁用' if args.no_format_augmentation else '启用'}")
    print("=" * 80)

    # 加载数据
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"[ERROR] 输入文件不存在: {input_path}")
        sys.exit(1)

    samples = load_jsonl(input_path)
    print(f"加载了 {len(samples)} 个样本")

    # 转换为训练记录（应用过滤）
    records = []
    filtered_count = 0
    for sample in samples:
        try:
            record = build_training_record(sample, filter_format_aug=args.no_format_augmentation)
            if record is None:
                filtered_count += 1
                continue
            records.append(record)
        except Exception as e:
            print(f"[WARNING] 跳过样本 {sample.get('pair_id', 'unknown')}: {e}")
            continue

    print(f"成功转换 {len(records)} 个记录")
    if args.no_format_augmentation:
        print(f"过滤了 {filtered_count} 个format增强样本")

    # 分割数据
    train_records, val_records, test_records = split_data(
        records,
        split_by=args.split_by,
        stratify_by=args.stratify_by,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )

    print(f"数据分割: train={len(train_records)}, val={len(val_records)}, test={len(test_records)}")

    # 保存数据
    output_dir = Path(".")
    train_path = output_dir / f"{args.output_prefix}_train.jsonl"
    val_path = output_dir / f"{args.output_prefix}_val.jsonl"
    test_path = output_dir / f"{args.output_prefix}_test.jsonl"
    all_path = output_dir / f"{args.output_prefix}_all.jsonl"

    dump_jsonl(train_path, train_records)
    dump_jsonl(val_path, val_records)
    dump_jsonl(test_path, test_records)
    dump_jsonl(all_path, records)

    print(f"已保存: {train_path}")
    print(f"已保存: {val_path}")
    print(f"已保存: {test_path}")
    print(f"已保存: {all_path}")

    # 生成统计报告
    report = {
        "input_file": str(input_path),
        "output_prefix": args.output_prefix,
        "split_by": args.split_by,
        "stratify_by": args.stratify_by,
        "seed": args.seed,
        "train_ratio": args.train_ratio,
        "val_ratio": args.val_ratio,
        "test_ratio": args.test_ratio,
        "format_augmentation_disabled": args.no_format_augmentation,
        "format_augmentation_filtered_count": filtered_count if args.no_format_augmentation else 0,
        "total_input": len(samples),
        "prepared": len(records),
        "dropped": len(samples) - len(records) - filtered_count,
        "overall": generate_statistics(records, "overall"),
        "train": generate_statistics(train_records, "train"),
        "val": generate_statistics(val_records, "val"),
        "test": generate_statistics(test_records, "test"),
    }

    report_path = output_dir / f"{args.output_prefix}_report.json"
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"已保存报告: {report_path}")
    print("=" * 80)
    print("数据生成完成！")
    print(f"关键统计:")
    print(f"  - 总输入: {len(samples)}")
    print(f"  - 过滤format增强: {filtered_count}")
    print(f"  - 最终记录: {len(records)}")
    print(f"  - 训练集: {len(train_records)}")
    print(f"  - 验证集: {len(val_records)}")
    print(f"  - 测试集: {len(test_records)}")


if __name__ == "__main__":
    main()
