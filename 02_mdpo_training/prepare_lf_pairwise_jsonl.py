from __future__ import annotations

import argparse
import json
from pathlib import Path


def convert_file(input_path: Path, output_path: Path) -> tuple[int, int]:
    kept = 0
    dropped = 0
    with input_path.open("r", encoding="utf-8") as fin, output_path.open("w", encoding="utf-8") as fout:
        for line in fin:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                dropped += 1
                continue

            prompt = row.get("prompt")
            chosen = row.get("chosen")
            rejected = row.get("rejected")
            if not (isinstance(prompt, str) and isinstance(chosen, str) and isinstance(rejected, str)):
                dropped += 1
                continue

            payload = {
                "prompt": prompt,
                "chosen": chosen,
                "rejected": rejected,
            }
            fout.write(json.dumps(payload, ensure_ascii=False) + "\n")
            kept += 1
    return kept, dropped


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare minimal pairwise jsonl for LLaMA-Factory RM/DPO stage.")
    parser.add_argument(
        "--train-in",
        default="dpo_margin_dataset_v3_train.jsonl",
    )
    parser.add_argument(
        "--val-in",
        default="dpo_margin_dataset_v3_val.jsonl",
    )
    parser.add_argument(
        "--train-out",
        default="dpo_margin_dataset_v3_train_lf.jsonl",
    )
    parser.add_argument(
        "--val-out",
        default="dpo_margin_dataset_v3_val_lf.jsonl",
    )
    args = parser.parse_args()

    train_kept, train_dropped = convert_file(Path(args.train_in), Path(args.train_out))
    val_kept, val_dropped = convert_file(Path(args.val_in), Path(args.val_out))

    print(
        json.dumps(
            {
                "train": {"kept": train_kept, "dropped": train_dropped, "output": args.train_out},
                "val": {"kept": val_kept, "dropped": val_dropped, "output": args.val_out},
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
