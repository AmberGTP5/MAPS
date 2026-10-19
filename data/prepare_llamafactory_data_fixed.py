"""
准备LLaMA-Factory格式数据 (修复版)
修复了Codex指出的问题：
1. SFT数据只包含chosen，不是ranking格式
2. DPO数据包含prompt/chosen/rejected/margin
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def dump_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def convert_to_sft_format(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    转换为SFT格式（只包含prompt和chosen）
    """
    sft_data = []
    for record in records:
        prompt = record.get("prompt")
        chosen = record.get("chosen")
        if isinstance(chosen, dict):
            chosen = chosen.get("response_text")
        if not (isinstance(prompt, str) and isinstance(chosen, str)):
            continue

        # SFT只需要prompt和response（chosen）
        sft_data.append({
            "instruction": prompt,
            "output": chosen,
        })
    return sft_data


def convert_to_dpo_format(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    转换为DPO格式（包含prompt/chosen/rejected/margin）
    """
    def infer_margin(record: Dict[str, Any]) -> float:
        margin = record.get("margin")
        if isinstance(margin, (int, float)):
            return float(margin)

        pref_margin = record.get("preference_margin")
        if isinstance(pref_margin, (int, float)):
            return float(pref_margin)

        pref = record.get("preference_strength")
        if isinstance(pref, dict):
            sev_margin = pref.get("severity_margin")
            if isinstance(sev_margin, (int, float)):
                return float(sev_margin)

            chosen_pen = pref.get("chosen_penalty_score")
            rejected_pen = pref.get("rejected_penalty_score")
            if isinstance(chosen_pen, (int, float)) and isinstance(rejected_pen, (int, float)):
                gap = float(rejected_pen) - float(chosen_pen)
                return max(0.05, min(2.0, 0.05 * gap))

        pair_type = record.get("pair_type", "hard_invalid")
        return 1.0 if pair_type == "hard_invalid" else 0.3

    dpo_data = []
    for record in records:
        prompt = record.get("prompt")
        chosen = record.get("chosen")
        rejected = record.get("rejected")

        if isinstance(chosen, dict):
            chosen = chosen.get("response_text")
        if isinstance(rejected, dict):
            rejected = rejected.get("response_text")

        if not (isinstance(prompt, str) and isinstance(chosen, str) and isinstance(rejected, str)):
            continue

        dpo_data.append({
            "prompt": prompt,
            "chosen": chosen,
            "rejected": rejected,
            "margin": infer_margin(record),
        })
    return dpo_data


def create_dataset_info(output_dir: Path, dataset_name: str, stage: str) -> Dict[str, Any]:
    """
    创建dataset_info.json
    
    Args:
        output_dir: 输出目录
        dataset_name: 数据集名称
        stage: 训练阶段 ("sft" 或 "dpo")
    """
    if stage == "sft":
        return {
            f"{dataset_name}_train": {
                "file_name": "train.jsonl",
                "formatting": "alpaca",
                "columns": {
                    "prompt": "instruction",
                    "response": "output",
                },
            },
            f"{dataset_name}_val": {
                "file_name": "val.jsonl",
                "formatting": "alpaca",
                "columns": {
                    "prompt": "instruction",
                    "response": "output",
                },
            },
        }
    else:  # dpo
        return {
            f"{dataset_name}_train": {
                "file_name": "train.jsonl",
                "formatting": "alpaca",
                "ranking": True,
                "columns": {
                    "prompt": "prompt",
                    "chosen": "chosen",
                    "rejected": "rejected",
                },
            },
            f"{dataset_name}_val": {
                "file_name": "val.jsonl",
                "formatting": "alpaca",
                "ranking": True,
                "columns": {
                    "prompt": "prompt",
                    "chosen": "chosen",
                    "rejected": "rejected",
                },
            },
        }


def main():
    parser = argparse.ArgumentParser(description="准备LLaMA-Factory格式数据")
    parser.add_argument(
        "--input-prefix",
        type=str,
        default="dpo_mdpo_optimized",
        help="输入文件前缀",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/pvr_mdpo_optimized",
        help="输出目录",
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        default="pvr_mdpo_optimized",
        help="数据集名称",
    )
    parser.add_argument(
        "--stage",
        type=str,
        default="dpo",
        choices=["sft", "dpo", "both"],
        help="训练阶段",
    )
    args = parser.parse_args()

    print("=" * 80)
    print("准备LLaMA-Factory格式数据 (修复版)")
    print("=" * 80)
    print(f"输入前缀: {args.input_prefix}")
    print(f"输出目录: {args.output_dir}")
    print(f"数据集名称: {args.dataset_name}")
    print(f"训练阶段: {args.stage}")
    print("=" * 80)

    # 加载数据
    train_path = Path(f"{args.input_prefix}_train.jsonl")
    val_path = Path(f"{args.input_prefix}_val.jsonl")

    if not train_path.exists():
        print(f"[ERROR] 训练集不存在: {train_path}")
        return

    if not val_path.exists():
        print(f"[ERROR] 验证集不存在: {val_path}")
        return

    train_records = load_jsonl(train_path)
    val_records = load_jsonl(val_path)

    print(f"加载训练集: {len(train_records)} 条")
    print(f"加载验证集: {len(val_records)} 条")

    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 根据stage转换数据
    if args.stage in ["sft", "both"]:
        print("\n生成SFT格式数据...")
        sft_dir = output_dir / "sft"
        sft_dir.mkdir(parents=True, exist_ok=True)

        sft_train = convert_to_sft_format(train_records)
        sft_val = convert_to_sft_format(val_records)

        dump_jsonl(sft_dir / "train.jsonl", sft_train)
        dump_jsonl(sft_dir / "val.jsonl", sft_val)

        # 创建dataset_info.json
        dataset_info = create_dataset_info(sft_dir, f"{args.dataset_name}_sft", "sft")
        with (sft_dir / "dataset_info.json").open("w", encoding="utf-8") as f:
            json.dump(dataset_info, f, ensure_ascii=False, indent=2)

        print(f"  已保存: {sft_dir / 'train.jsonl'} ({len(sft_train)} 条)")
        print(f"  已保存: {sft_dir / 'val.jsonl'} ({len(sft_val)} 条)")
        print(f"  已保存: {sft_dir / 'dataset_info.json'}")

    if args.stage in ["dpo", "both"]:
        print("\n生成DPO格式数据...")
        dpo_dir = output_dir / "dpo"
        dpo_dir.mkdir(parents=True, exist_ok=True)

        dpo_train = convert_to_dpo_format(train_records)
        dpo_val = convert_to_dpo_format(val_records)

        dump_jsonl(dpo_dir / "train.jsonl", dpo_train)
        dump_jsonl(dpo_dir / "val.jsonl", dpo_val)

        # 创建dataset_info.json
        dataset_info = create_dataset_info(dpo_dir, f"{args.dataset_name}_dpo", "dpo")
        with (dpo_dir / "dataset_info.json").open("w", encoding="utf-8") as f:
            json.dump(dataset_info, f, ensure_ascii=False, indent=2)

        print(f"  已保存: {dpo_dir / 'train.jsonl'} ({len(dpo_train)} 条)")
        print(f"  已保存: {dpo_dir / 'val.jsonl'} ({len(dpo_val)} 条)")
        print(f"  已保存: {dpo_dir / 'dataset_info.json'}")

    print("\n=" * 80)
    print("数据准备完成！")
    print("\n使用说明:")
    if args.stage in ["sft", "both"]:
        print(f"  SFT训练: dataset_dir={sft_dir}, dataset={args.dataset_name}_sft_train")
    if args.stage in ["dpo", "both"]:
        print(f"  DPO训练: dataset_dir={dpo_dir}, dataset={args.dataset_name}_dpo_train")


if __name__ == "__main__":
    main()
