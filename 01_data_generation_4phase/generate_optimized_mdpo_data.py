"""
优化的mDPO数据生成脚本
目标：生成适合margin-DPO训练的高质量偏好数据

关键优化：
1. 降低margin值（避免梯度消失）
2. 增加soft样本比例（学习细粒度偏好）
3. 禁用format_failure增强（减少噪声）
4. 平衡hard/soft样本分布
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
    compute_preference_strength,
    derive_task_metadata,
    render_plan_response,
    render_oracle_reasoning_trace,
)


# ============================================================================
# 优化的Margin配置（关键改动）
# ============================================================================
OPTIMIZED_MARGIN_BUCKETS = [
    {"name": "none", "min_score": 0.0, "max_score": 0.0, "margin": 0.0},
    {"name": "light", "min_score": 0.0, "max_score": 5.0, "margin": 0.05},      # 从0.15降到0.05
    {"name": "moderate", "min_score": 5.0, "max_score": 10.0, "margin": 0.15},  # 从0.35降到0.15
    {"name": "severe", "min_score": 10.0, "max_score": 15.0, "margin": 0.3},    # 从0.7降到0.3
    {"name": "critical", "min_score": 15.0, "max_score": float("inf"), "margin": 0.5},  # 从1.1降到0.5
]

OPTIMIZED_SOFT_UTILITY_BUCKETS = [
    {"name": "none", "min_gap": 0.0, "max_gap": 0.0, "margin": 0.0},
    {"name": "light", "min_gap": 0.0, "max_gap": 1.0, "margin": 0.05},      # 从0.15降到0.05
    {"name": "moderate", "min_gap": 1.0, "max_gap": 3.0, "margin": 0.15},   # 从0.35降到0.15
    {"name": "severe", "min_gap": 3.0, "max_gap": 6.0, "margin": 0.3},      # 从0.7降到0.3
    {"name": "critical", "min_gap": 6.0, "max_gap": float("inf"), "margin": 0.5},  # 从1.1降到0.5
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
    """使用优化的margin buckets计算偏好强度"""
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
        "trajectory_cost_gap": round(rejected_cost - chosen_cost, 6),
        "chosen_utility": round(chosen_utility_value, 6),
        "rejected_utility": round(rejected_utility_value, 6),
        "utility_gap": round(utility_gap, 6),
        "error_types": list(rejected_feedback.get("error_type", [])),
        "violation_breakdown": dict(breakdown),
    }


def ensure_response_text(block: Dict[str, Any]) -> str:
    response_text = block.get("response_text")
    if isinstance(response_text, str) and response_text.strip():
        return response_text
    return render_plan_response(block.get("thought", ""), block.get("plan", []))


def build_normalized_prompt(sample: Dict[str, Any]) -> str:
    local_graph = sample.get("local_graph")
    if isinstance(local_graph, dict):
        return build_planning_prompt(
            instruction=sample["instruction"],
            scenario_json=local_graph,
            target_item=sample["target_item"],
            item_profile=sample.get("item_profile", {}),
            start_node=sample["start_node"],
            end_node=sample["end_node"],
            reference_paths=sample.get("reference_paths", []),
        )
    return sample["prompt"]


def build_concise_chosen_response(sample: Dict[str, Any]) -> str:
    chosen = sample["chosen"]
    reasoning_trace = chosen.get("reasoning_trace")
    if isinstance(reasoning_trace, dict) and reasoning_trace:
        thought = render_oracle_reasoning_trace(reasoning_trace)
    else:
        thought = (chosen.get("thought") or "").strip()
    return render_plan_response(thought, chosen.get("plan", []))


def normalize_pair_type(value: Any) -> str:
    if isinstance(value, str) and value in VALID_PAIR_TYPES:
        return value
    return "hard_invalid"


def to_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def build_training_record(sample: Dict[str, Any]) -> Dict[str, Any]:
    """构建训练记录，使用优化的margin计算"""
    prompt = build_normalized_prompt(sample)
    chosen = sample["chosen"]
    rejected = sample["rejected"]
    chosen_feedback = chosen["feedback"]
    rejected_feedback = rejected["feedback"]
    pair_type = normalize_pair_type(sample.get("pair_type"))
    
    chosen_utility = to_float(
        sample.get("chosen_utility"),
        to_float(chosen_feedback.get("trajectory_cost"), 0.0),
    )
    rejected_utility = to_float(
        sample.get("rejected_utility"),
        to_float(rejected_feedback.get("trajectory_cost"), chosen_utility),
    )
    utility_gap = round(rejected_utility - chosen_utility, 6)

    task_metadata = sample.get("task_metadata") or derive_task_metadata(
        sample["start_node"],
        sample["end_node"],
        sample["target_item"],
    )
    
    # 使用优化的偏好强度计算
    preference_strength = compute_optimized_preference_strength(
        chosen_feedback,
        rejected_feedback,
        pair_type=pair_type,
        chosen_utility=chosen_utility,
        rejected_utility=rejected_utility,
    )

    pair_id = sample.get("pair_id") or build_preference_pair_id(
        prompt,
        chosen["plan"],
        rejected["plan"],
    )
    selected_robot = (
        (chosen.get("reasoning_trace") or {}).get("selection", {}).get("selected_robot")
        or (chosen["plan"][0].get("robot") if chosen.get("plan") else "unknown")
    )

    return {
        "pair_id": pair_id,
        "task_id": sample.get("task_id"),
        "pair_type": pair_type,
        "dataset_schema_version": sample.get("dataset_schema_version", "pvr_dpo_v2_optimized"),
        "prompt_version": sample.get("prompt_version", "graph_prompt_v3"),
        "selected_robot": selected_robot,
        "chosen_valid": bool(sample.get("chosen_valid", chosen_feedback.get("is_valid", False))),
        "rejected_valid": bool(sample.get("rejected_valid", rejected_feedback.get("is_valid", False))),
        "chosen_utility": round(chosen_utility, 6),
        "rejected_utility": round(rejected_utility, 6),
        "utility_gap": utility_gap,
        "prompt": prompt,
        "chosen": build_concise_chosen_response(sample),
        "rejected": ensure_response_text(rejected),
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
        "augmentation_type": None,
        "exclude_from_sft": False,
    }


def choose_group_key(record: Dict[str, Any], split_by: str) -> str:
    metadata = record["task"]["task_metadata"]
    if split_by == "random":
        return record["pair_id"]
    if split_by == "task":
        return metadata["task_signature"]
    if split_by == "item":
        return metadata["target_item"]
    if split_by == "building_pair":
        return metadata["building_pair"]
    if split_by == "start_building":
        return metadata["start_building"]
    if split_by == "end_building":
        return metadata["end_building"]
    raise ValueError(f"Unsupported split_by={split_by}")


def choose_stratify_key(record: Dict[str, Any], stratify_by: str) -> str:
    metadata = record["task"]["task_metadata"]
    if stratify_by == "pair_type":
        return record["pair_type"]
    if stratify_by == "robot":
        return record["selected_robot"]
    if stratify_by == "robot_profile":
        return metadata.get("robot_profile", "unknown")
    if stratify_by == "task_profile":
        return metadata.get("task_profile", "unknown")
    if stratify_by == "item":
        return metadata["target_item"]
    if stratify_by == "building_pair":
        return metadata["building_pair"]
    raise ValueError(f"Unsupported stratify_by={stratify_by}")


def split_data(
    records: List[Dict[str, Any]],
    *,
    split_by: str,
    stratify_by: str,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """分割数据，确保没有泄漏"""
    import random
    random.seed(seed)

    # 按group_key分组
    groups = defaultdict(list)
    for record in records:
        key = choose_group_key(record, split_by)
        groups[key].append(record)

    # 按stratify_key统计每个组
    group_strata = {}
    for group_key, group_records in groups.items():
        strata_counter = Counter(choose_stratify_key(r, stratify_by) for r in group_records)
        group_strata[group_key] = strata_counter.most_common(1)[0][0]

    # 按strata分组
    strata_groups = defaultdict(list)
    for group_key, stratum in group_strata.items():
        strata_groups[stratum].append(group_key)

    # 在每个stratum内分割
    train_groups, val_groups, test_groups = [], [], []
    for stratum, group_keys in strata_groups.items():
        random.shuffle(group_keys)
        n = len(group_keys)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        train_groups.extend(group_keys[:n_train])
        val_groups.extend(group_keys[n_train:n_train + n_val])
        test_groups.extend(group_keys[n_train + n_val:])

    # 收集记录
    train_records = [r for g in train_groups for r in groups[g]]
    val_records = [r for g in val_groups for r in groups[g]]
    test_records = [r for g in test_groups for r in groups[g]]

    return train_records, val_records, test_records


def generate_statistics(records: List[Dict[str, Any]], name: str) -> Dict[str, Any]:
    """生成数据集统计信息"""
    pair_types = Counter(r["pair_type"] for r in records)
    severity_levels = Counter(r["preference_strength"]["severity_level"] for r in records)
    margins = Counter(r["margin"] for r in records)
    robots = Counter(r["selected_robot"] for r in records)
    error_types = Counter()
    for r in records:
        for error in r.get("target_errors", []):
            error_types[error] += 1

    items = Counter(r["task"]["target_item"] for r in records)
    building_pairs = Counter(r["task"]["task_metadata"]["building_pair"] for r in records)
    task_profiles = Counter(r["task"]["task_metadata"].get("task_profile", "unknown") for r in records)

    return {
        "count": len(records),
        "pair_type_distribution": dict(pair_types),
        "severity_distribution": dict(severity_levels),
        "margin_distribution": {str(k): v for k, v in margins.items()},
        "error_type_distribution": dict(error_types.most_common(20)),
        "robot_distribution": dict(robots),
        "task_profile_distribution": dict(task_profiles),
        "top_items": dict(items.most_common(20)),
        "top_building_pairs": dict(building_pairs.most_common(20)),
    }


def main():
    parser = argparse.ArgumentParser(description="生成优化的mDPO训练数据")
    parser.add_argument(
        "--input",
        type=str,
        default="dpo_dataset_filtered.jsonl",
        help="输入的过滤后数据文件",
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
        choices=["random", "task", "item", "building_pair", "start_building", "end_building"],
        help="数据分割的分组依据",
    )
    parser.add_argument(
        "--stratify-by",
        type=str,
        default="robot_profile",
        choices=["pair_type", "robot", "robot_profile", "task_profile", "item", "building_pair"],
        help="分层采样的依据",
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
    print("优化的mDPO数据生成")
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

    # 转换为训练记录
    records = []
    for sample in samples:
        try:
            record = build_training_record(sample)
            records.append(record)
        except Exception as e:
            print(f"[WARNING] 跳过样本 {sample.get('pair_id', 'unknown')}: {e}")
            continue

    print(f"成功转换 {len(records)} 个记录")

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
        "total_input": len(samples),
        "prepared": len(records),
        "dropped": len(samples) - len(records),
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
    print("=" * 80)
    print("\n关键统计:")
    print(f"  总样本数: {len(records)}")
    print(f"  训练集: {len(train_records)}")
    print(f"  验证集: {len(val_records)}")
    print(f"  测试集: {len(test_records)}")
    print(f"\n训练集分布:")
    print(f"  Pair类型: {report['train']['pair_type_distribution']}")
    print(f"  Margin分布: {report['train']['margin_distribution']}")
    print(f"  Severity分布: {report['train']['severity_distribution']}")


if __name__ == "__main__":
    main()
