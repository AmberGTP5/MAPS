"""
准备LLaMA-Factory格式的数据文件

将优化后的mDPO数据转换为LLaMA-Factory可以直接使用的格式
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List


def convert_to_llamafactory_format(input_path: Path, output_path: Path) -> int:
    """
    转换为LLaMA-Factory的DPO格式
    
    输入格式: {pair_id, prompt, chosen, rejected, margin, ...}
    输出格式: {prompt, chosen, rejected, margin}
    """
    kept = 0
    with input_path.open("r", encoding="utf-8") as fin, output_path.open("w", encoding="utf-8") as fout:
        for line in fin:
            if not line.strip():
                continue
            
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue

            # 提取必要字段
            prompt = row.get("prompt")
            chosen = row.get("chosen")
            rejected = row.get("rejected")
            margin = row.get("margin", 0.0)

            if not (isinstance(prompt, str) and isinstance(chosen, str) and isinstance(rejected, str)):
                continue

            # LLaMA-Factory格式
            payload = {
                "prompt": prompt,
                "chosen": chosen,
                "rejected": rejected,
                "margin": float(margin),  # 确保margin是浮点数
            }
            
            fout.write(json.dumps(payload, ensure_ascii=False) + "\n")
            kept += 1
    
    return kept


def generate_dataset_info(
    train_file: str,
    val_file: str,
    test_file: str,
    output_path: Path,
) -> None:
    """生成dataset_info.json配置文件"""
    dataset_info = {
        "pvr_mdpo_train_optimized": {
            "file_name": train_file,
            "formatting": "alpaca",
            "ranking": True,
            "columns": {
                "prompt": "prompt",
                "chosen": "chosen",
                "rejected": "rejected",
            },
        },
        "pvr_mdpo_val_optimized": {
            "file_name": val_file,
            "formatting": "alpaca",
            "ranking": True,
            "columns": {
                "prompt": "prompt",
                "chosen": "chosen",
                "rejected": "rejected",
            },
        },
        "pvr_mdpo_test_optimized": {
            "file_name": test_file,
            "formatting": "alpaca",
            "ranking": True,
            "columns": {
                "prompt": "prompt",
                "chosen": "chosen",
                "rejected": "rejected",
            },
        },
    }
    
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(dataset_info, f, ensure_ascii=False, indent=2)


def main():
    parser = argparse.ArgumentParser(description="准备LLaMA-Factory格式的数据")
    parser.add_argument(
        "--input-prefix",
        type=str,
        default="dpo_mdpo_optimized",
        help="输入文件前缀",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="llamafactory_data",
        help="输出目录",
    )
    args = parser.parse_args()

    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    # 转换文件
    train_in = Path(f"{args.input_prefix}_train.jsonl")
    val_in = Path(f"{args.input_prefix}_val.jsonl")
    test_in = Path(f"{args.input_prefix}_test.jsonl")

    train_out = output_dir / "train.jsonl"
    val_out = output_dir / "val.jsonl"
    test_out = output_dir / "test.jsonl"

    print("=" * 80)
    print("准备LLaMA-Factory数据")
    print("=" * 80)

    train_count = convert_to_llamafactory_format(train_in, train_out)
    val_count = convert_to_llamafactory_format(val_in, val_out)
    test_count = convert_to_llamafactory_format(test_in, test_out)

    print(f"训练集: {train_count} 样本 -> {train_out}")
    print(f"验证集: {val_count} 样本 -> {val_out}")
    print(f"测试集: {test_count} 样本 -> {test_out}")

    # 生成dataset_info.json
    dataset_info_path = output_dir / "dataset_info.json"
    generate_dataset_info(
        train_file="train.jsonl",
        val_file="val.jsonl",
        test_file="test.jsonl",
        output_path=dataset_info_path,
    )
    print(f"配置文件: {dataset_info_path}")

    print("=" * 80)
    print("数据准备完成！")
    print("=" * 80)
    print("\n下一步:")
    print(f"1. 将 {output_dir} 目录上传到服务器")
    print("2. 运行训练脚本")


if __name__ == "__main__":
    main()
