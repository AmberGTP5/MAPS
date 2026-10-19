from __future__ import annotations

import argparse
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def dump_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _largest_remainder_alloc(total: int, weights: Dict[str, int]) -> Dict[str, int]:
    weight_sum = sum(weights.values())
    if total <= 0 or weight_sum <= 0:
        return {k: 0 for k in weights}

    raw = {k: total * v / weight_sum for k, v in weights.items()}
    base = {k: int(math.floor(x)) for k, x in raw.items()}
    remainder = total - sum(base.values())

    frac = sorted(((raw[k] - base[k], k) for k in weights), reverse=True)
    for i in range(remainder):
        _, key = frac[i]
        base[key] += 1
    return base


def _split_counts(total: int, train_ratio: float, val_ratio: float) -> Tuple[int, int, int]:
    train = int(math.floor(total * train_ratio))
    val = int(math.floor(total * val_ratio))
    test = total - train - val
    return train, val, test


def _sample_signatures_by_bucket(
    *,
    bucket_signatures: Dict[str, List[str]],
    target_total: int,
    seed: int,
    train_ratio: float,
    val_ratio: float,
) -> Dict[str, Any]:
    bucket_sizes = {b: len(sigs) for b, sigs in bucket_signatures.items()}
    target_by_bucket = _largest_remainder_alloc(target_total, bucket_sizes)

    rng = random.Random(seed)
    sampled: Dict[str, List[str]] = {}
    split_bucket: Dict[str, Dict[str, List[str]]] = {}
    train_sigs: List[str] = []
    val_sigs: List[str] = []
    test_sigs: List[str] = []

    for bucket in sorted(bucket_signatures):
        candidates = list(bucket_signatures[bucket])
        rng.shuffle(candidates)
        need = target_by_bucket[bucket]
        if need > len(candidates):
            raise ValueError(f"Bucket {bucket} needs {need}, but only {len(candidates)} signatures available.")
        picked = candidates[:need]
        sampled[bucket] = picked

        t, v, te = _split_counts(need, train_ratio, val_ratio)
        rng.shuffle(picked)
        train_part = picked[:t]
        val_part = picked[t : t + v]
        test_part = picked[t + v : t + v + te]

        split_bucket[bucket] = {
            "train": train_part,
            "val": val_part,
            "test": test_part,
        }
        train_sigs.extend(train_part)
        val_sigs.extend(val_part)
        test_sigs.extend(test_part)

    return {
        "target_by_bucket": target_by_bucket,
        "sampled_by_bucket": sampled,
        "split_by_bucket": split_bucket,
        "train_signatures": sorted(train_sigs),
        "val_signatures": sorted(val_sigs),
        "test_signatures": sorted(test_sigs),
    }


def _rows_by_signature(rows: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    out: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        sig = (row.get("task_metadata") or {}).get("task_signature")
        if isinstance(sig, str) and sig:
            out[sig].append(row)
    return out


def _flatten(sig_list: List[str], grouped: Dict[str, List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for sig in sig_list:
        rows.extend(grouped.get(sig, []))
    return rows


def _pair_type_dist(rows: List[Dict[str, Any]]) -> Dict[str, int]:
    return dict(Counter(str(r.get("pair_type", "unknown")) for r in rows))


def _to_sft_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    out: List[Dict[str, str]] = []
    for r in rows:
        prompt = r.get("prompt")
        chosen = r.get("chosen")
        if isinstance(chosen, dict):
            chosen = chosen.get("response_text")
        if not (isinstance(prompt, str) and isinstance(chosen, str)):
            continue
        out.append({"instruction": prompt, "output": chosen})
    return out


def _to_dpo_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for r in rows:
        prompt = r.get("prompt")
        chosen = r.get("chosen")
        rejected = r.get("rejected")
        if isinstance(chosen, dict):
            chosen = chosen.get("response_text")
        if isinstance(rejected, dict):
            rejected = rejected.get("response_text")
        if not (isinstance(prompt, str) and isinstance(chosen, str) and isinstance(rejected, str)):
            continue
        margin = r.get("margin", 0.0)
        out.append(
            {
                "prompt": prompt,
                "chosen": chosen,
                "rejected": rejected,
                "margin": float(margin) if isinstance(margin, (int, float)) else 0.0,
            }
        )
    return out


def _write_dataset_info_sft(path: Path, dataset_name: str) -> None:
    payload = {
        f"{dataset_name}_sft_train": {
            "file_name": "train.jsonl",
            "formatting": "alpaca",
            "columns": {"prompt": "instruction", "response": "output"},
        },
        f"{dataset_name}_sft_val": {
            "file_name": "val.jsonl",
            "formatting": "alpaca",
            "columns": {"prompt": "instruction", "response": "output"},
        },
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_dataset_info_dpo(path: Path, dataset_name: str) -> None:
    payload = {
        f"{dataset_name}_dpo_train": {
            "file_name": "train.jsonl",
            "formatting": "alpaca",
            "ranking": True,
            "columns": {"prompt": "prompt", "chosen": "chosen", "rejected": "rejected"},
        },
        f"{dataset_name}_dpo_val": {
            "file_name": "val.jsonl",
            "formatting": "alpaca",
            "ranking": True,
            "columns": {"prompt": "prompt", "chosen": "chosen", "rejected": "rejected"},
        },
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _yaml_sft(scale: int) -> str:
    return f"""### model
model_name_or_path: ./models/Qwen2.5-7B-Instruct
trust_remote_code: true
flash_attn: fa2

### method
stage: sft
do_train: true
finetuning_type: lora
lora_target: all
lora_rank: 64
lora_alpha: 128
lora_dropout: 0.05

### dataset
dataset_dir: ./data/pvr_p1_scale_{scale}/sft
dataset: p1_scale_{scale}_sft_train
eval_dataset: p1_scale_{scale}_sft_val
template: qwen
cutoff_len: 4096
overwrite_cache: true
preprocessing_num_workers: 16

### output
output_dir: ./saves/qwen25_7b/lora/p1_scale_{scale}_sft
logging_steps: 5
save_steps: 100
plot_loss: true
overwrite_output_dir: true

### train
per_device_train_batch_size: 1
gradient_accumulation_steps: 8
learning_rate: 5.0e-5
num_train_epochs: 2.0
lr_scheduler_type: cosine
warmup_ratio: 0.1
max_grad_norm: 1.0
bf16: true
ddp_timeout: 180000000

### eval
per_device_eval_batch_size: 1
eval_strategy: steps
eval_steps: 100
save_total_limit: 2

### optimizer
optim: adamw_torch
weight_decay: 0.01
"""


def _yaml_dpo_vanilla(scale: int) -> str:
    return f"""### model
model_name_or_path: ./models/Qwen2.5-7B-Instruct
trust_remote_code: true
flash_attn: fa2

### method
stage: dpo
do_train: true
finetuning_type: lora
pref_loss: sigmoid
pref_beta: 0.1

### initialize from SFT
adapter_name_or_path: ./saves/qwen25_7b/lora/p1_scale_{scale}_sft
ref_model: ./models/Qwen2.5-7B-Instruct
ref_model_adapters: ./saves/qwen25_7b/lora/p1_scale_{scale}_sft

### dataset (hard-only preference rows)
dataset_dir: ./data/pvr_p1_scale_{scale}/dpo
dataset: p1_scale_{scale}_dpo_train
eval_dataset: p1_scale_{scale}_dpo_val
template: qwen
cutoff_len: 4096
overwrite_cache: true
preprocessing_num_workers: 16

### output
output_dir: ./saves/qwen25_7b/lora/p1_scale_{scale}_dpo_vanilla
logging_steps: 5
save_steps: 100
plot_loss: true
overwrite_output_dir: true

### train
per_device_train_batch_size: 1
gradient_accumulation_steps: 8
learning_rate: 5.0e-6
num_train_epochs: 2.0
lr_scheduler_type: cosine
warmup_ratio: 0.1
max_grad_norm: 1.0
bf16: true
ddp_timeout: 180000000

### eval
per_device_eval_batch_size: 1
eval_strategy: steps
eval_steps: 100
save_total_limit: 2

### optimizer
optim: adamw_torch
weight_decay: 0.01
"""


def _yaml_mdpo(scale: int) -> str:
    return f"""### model
model_name_or_path: ./models/Qwen2.5-7B-Instruct
trust_remote_code: true
flash_attn: fa2

### method
stage: dpo
do_train: true
finetuning_type: lora
pref_loss: sigmoid
pref_beta: 0.1

### initialize from SFT
adapter_name_or_path: ./saves/qwen25_7b/lora/p1_scale_{scale}_sft
ref_model: ./models/Qwen2.5-7B-Instruct
ref_model_adapters: ./saves/qwen25_7b/lora/p1_scale_{scale}_sft

### sample-wise margin DPO
pref_margin_mode: sample
pref_margin_field: margin
pref_margin_scale: 0.5
pref_margin_weight: 0.05

### dataset (hard-only preference rows)
dataset_dir: ./data/pvr_p1_scale_{scale}/dpo
dataset: p1_scale_{scale}_dpo_train
eval_dataset: p1_scale_{scale}_dpo_val
template: qwen
cutoff_len: 4096
overwrite_cache: true
preprocessing_num_workers: 16

### output
output_dir: ./saves/qwen25_7b/lora/p1_scale_{scale}_mdpo
logging_steps: 5
save_steps: 100
plot_loss: true
overwrite_output_dir: true

### train
per_device_train_batch_size: 1
gradient_accumulation_steps: 8
learning_rate: 5.0e-6
num_train_epochs: 2.0
lr_scheduler_type: cosine
warmup_ratio: 0.1
max_grad_norm: 1.0
bf16: true
ddp_timeout: 180000000

### eval
per_device_eval_batch_size: 1
eval_strategy: steps
eval_steps: 100
save_total_limit: 2

### optimizer
optim: adamw_torch
weight_decay: 0.01
"""


def _write_yaml(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")


def _build_scale(
    *,
    scale: int,
    seed: int,
    rows_all: List[Dict[str, Any]],
    bucket_signatures: Dict[str, List[str]],
    train_ratio: float,
    val_ratio: float,
    data_dir: Path,
) -> Dict[str, Any]:
    sampled = _sample_signatures_by_bucket(
        bucket_signatures=bucket_signatures,
        target_total=scale,
        seed=seed + scale,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
    )

    grouped = _rows_by_signature(rows_all)
    train_rows = _flatten(sampled["train_signatures"], grouped)
    val_rows = _flatten(sampled["val_signatures"], grouped)
    test_rows = _flatten(sampled["test_signatures"], grouped)
    all_rows = train_rows + val_rows + test_rows

    prefix = data_dir / f"p1_scale_{scale}"
    dump_jsonl(Path(str(prefix) + "_train.jsonl"), train_rows)
    dump_jsonl(Path(str(prefix) + "_val.jsonl"), val_rows)
    dump_jsonl(Path(str(prefix) + "_test.jsonl"), test_rows)
    dump_jsonl(Path(str(prefix) + "_all.jsonl"), all_rows)

    task_payload = {
        "train_signatures": sampled["train_signatures"],
        "val_signatures": sampled["val_signatures"],
        "test_signatures": sampled["test_signatures"],
        "bucket_signatures": sampled["sampled_by_bucket"],
        "target_by_bucket": sampled["target_by_bucket"],
    }
    Path(str(prefix) + "_tasks.json").write_text(
        json.dumps(task_payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    hard_train = [r for r in train_rows if str(r.get("pair_type")) == "hard_invalid"]
    hard_val = [r for r in val_rows if str(r.get("pair_type")) == "hard_invalid"]
    hard_test = [r for r in test_rows if str(r.get("pair_type")) == "hard_invalid"]
    hard_all = hard_train + hard_val + hard_test
    hard_prefix = data_dir / f"p1_scale_{scale}_hardonly"
    dump_jsonl(Path(str(hard_prefix) + "_train.jsonl"), hard_train)
    dump_jsonl(Path(str(hard_prefix) + "_val.jsonl"), hard_val)
    dump_jsonl(Path(str(hard_prefix) + "_test.jsonl"), hard_test)
    dump_jsonl(Path(str(hard_prefix) + "_all.jsonl"), hard_all)

    # Build LLaMA-Factory-ready folders:
    # - sft from full train/val
    # - dpo from hard-only train/val
    lf_dir = data_dir / f"pvr_p1_scale_{scale}"
    sft_dir = lf_dir / "sft"
    dpo_dir = lf_dir / "dpo"
    sft_dir.mkdir(parents=True, exist_ok=True)
    dpo_dir.mkdir(parents=True, exist_ok=True)

    sft_train = _to_sft_rows(train_rows)
    sft_val = _to_sft_rows(val_rows)
    dpo_train = _to_dpo_rows(hard_train)
    dpo_val = _to_dpo_rows(hard_val)

    dump_jsonl(sft_dir / "train.jsonl", sft_train)
    dump_jsonl(sft_dir / "val.jsonl", sft_val)
    dump_jsonl(dpo_dir / "train.jsonl", dpo_train)
    dump_jsonl(dpo_dir / "val.jsonl", dpo_val)
    _write_dataset_info_sft(sft_dir / "dataset_info.json", f"p1_scale_{scale}")
    _write_dataset_info_dpo(dpo_dir / "dataset_info.json", f"p1_scale_{scale}")

    _write_yaml(data_dir / f"train_sft_p1_scale_{scale}.yaml", _yaml_sft(scale))
    _write_yaml(data_dir / f"train_dpo_vanilla_p1_scale_{scale}.yaml", _yaml_dpo_vanilla(scale))
    _write_yaml(data_dir / f"train_mdpo_p1_scale_{scale}.yaml", _yaml_mdpo(scale))

    return {
        "scale": scale,
        "target_by_bucket": sampled["target_by_bucket"],
        "signatures": {
            "train": len(sampled["train_signatures"]),
            "val": len(sampled["val_signatures"]),
            "test": len(sampled["test_signatures"]),
            "all": len(sampled["train_signatures"])
            + len(sampled["val_signatures"])
            + len(sampled["test_signatures"]),
        },
        "rows": {
            "train": len(train_rows),
            "val": len(val_rows),
            "test": len(test_rows),
            "all": len(all_rows),
        },
        "pair_type_distribution": {
            "train": _pair_type_dist(train_rows),
            "val": _pair_type_dist(val_rows),
            "test": _pair_type_dist(test_rows),
        },
        "hard_only_rows": {
            "train": len(hard_train),
            "val": len(hard_val),
            "test": len(hard_test),
            "all": len(hard_all),
        },
        "artifacts": {
            "prefix": str(prefix),
            "hard_prefix": str(hard_prefix),
            "lf_dir": str(lf_dir),
            "yaml": {
                "sft": f"train_sft_p1_scale_{scale}.yaml",
                "dpo_vanilla": f"train_dpo_vanilla_p1_scale_{scale}.yaml",
                "mdpo": f"train_mdpo_p1_scale_{scale}.yaml",
            },
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build P1 scale sweep assets (375/1000) from locked P0 dataset.")
    parser.add_argument("--source-all", default="p0_bucketed_1000_all.jsonl")
    parser.add_argument("--source-tasks", default="p0_bucketed_1000_tasks.json")
    parser.add_argument("--scales", default="375,1000")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    args = parser.parse_args()

    data_dir = Path(__file__).resolve().parent
    source_all = data_dir / args.source_all
    source_tasks = data_dir / args.source_tasks
    paper_dir = data_dir / "paper_artifacts"
    paper_dir.mkdir(parents=True, exist_ok=True)

    rows_all = load_jsonl(source_all)
    task_split = json.loads(source_tasks.read_text(encoding="utf-8"))
    bucket_signatures = task_split.get("bucket_signatures") or {}
    if not bucket_signatures:
        raise ValueError("source tasks json does not contain bucket_signatures.")

    scales = [int(x.strip()) for x in args.scales.split(",") if x.strip()]
    report = {
        "source_all": str(source_all),
        "source_tasks": str(source_tasks),
        "seed": args.seed,
        "train_ratio": args.train_ratio,
        "val_ratio": args.val_ratio,
        "scales": [],
    }

    for scale in scales:
        report["scales"].append(
            _build_scale(
                scale=scale,
                seed=args.seed,
                rows_all=rows_all,
                bucket_signatures=bucket_signatures,
                train_ratio=args.train_ratio,
                val_ratio=args.val_ratio,
                data_dir=data_dir,
            )
        )

    out_json = paper_dir / "p1_scale_setup_report.json"
    out_md = paper_dir / "p1_scale_setup_report.md"
    out_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    md_lines = [
        "# P1-1 Scale Sweep Asset Setup\n",
        f"- source_all: `{report['source_all']}`",
        f"- source_tasks: `{report['source_tasks']}`",
        f"- seed: `{report['seed']}`",
        "",
    ]
    for item in report["scales"]:
        md_lines.extend(
            [
                f"## Scale {item['scale']}",
                f"- signatures(train/val/test/all): {item['signatures']['train']}/"
                f"{item['signatures']['val']}/{item['signatures']['test']}/{item['signatures']['all']}",
                f"- rows(train/val/test/all): {item['rows']['train']}/"
                f"{item['rows']['val']}/{item['rows']['test']}/{item['rows']['all']}",
                f"- hard-only rows(train/val/test/all): {item['hard_only_rows']['train']}/"
                f"{item['hard_only_rows']['val']}/{item['hard_only_rows']['test']}/{item['hard_only_rows']['all']}",
                f"- target_by_bucket: {item['target_by_bucket']}",
                f"- lf_dir: `{item['artifacts']['lf_dir']}`",
                f"- yaml: `{item['artifacts']['yaml']['sft']}`, `{item['artifacts']['yaml']['dpo_vanilla']}`, "
                f"`{item['artifacts']['yaml']['mdpo']}`",
                "",
            ]
        )
    out_md.write_text("\n".join(md_lines), encoding="utf-8")

    print(f"[saved] {out_json}")
    print(f"[saved] {out_md}")
    for item in report["scales"]:
        print(
            f"[scale {item['scale']}] signatures(all)={item['signatures']['all']}, "
            f"rows(train/val/test)={item['rows']['train']}/{item['rows']['val']}/{item['rows']['test']}, "
            f"hard_train={item['hard_only_rows']['train']}"
        )


if __name__ == "__main__":
    main()
