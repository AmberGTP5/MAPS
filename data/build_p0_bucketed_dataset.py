"""
Build a P0-locked 1000-task dataset with fixed bucket quotas and task-level splits.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

from preference_utils import extract_building_id


FLOOR_RE = re.compile(r"_(\d+)F(?:_|$)")
BUCKETS = ("B1", "B2", "B3", "B4")


@dataclass
class TaskRecord:
    signature: str
    bucket: str
    row_ids: List[int]
    pair_types: Set[str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build fixed-bucket P0 dataset from filtered jsonl pools.")
    parser.add_argument(
        "--input-files",
        nargs="+",
        default=["dpo_dataset_filtered.jsonl"],
        help="Filtered jsonl files to merge (supports multiple runs).",
    )
    parser.add_argument("--output-prefix", default="p0_bucketed_1000", help="Output prefix.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--target-b1", type=int, default=300)
    parser.add_argument("--target-b2", type=int, default=250)
    parser.add_argument("--target-b3", type=int, default=200)
    parser.add_argument("--target-b4", type=int, default=250)
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument(
        "--require-both-pair-types",
        action="store_true",
        help="Require each selected task signature to contain both hard_invalid and soft_suboptimal.",
    )
    return parser.parse_args()


def floor_of(node_id: str) -> Optional[int]:
    if not isinstance(node_id, str):
        return None
    m = FLOOR_RE.search(node_id)
    if not m:
        return None
    return int(m.group(1))


def bucket_of(start_node: str, end_node: str) -> str:
    sb = extract_building_id(start_node)
    eb = extract_building_id(end_node)
    sf = floor_of(start_node)
    ef = floor_of(end_node)
    if sf is None or ef is None:
        return "unknown"
    if sb == eb and sf == ef:
        return "B1"
    if sb == eb and sf != ef:
        return "B2"
    if sb != eb and sf == ef:
        return "B3"
    return "B4"


def stable_row_key(row: Dict[str, Any]) -> str:
    pair_id = row.get("pair_id")
    if isinstance(pair_id, str) and pair_id:
        return f"pair:{pair_id}"
    payload = json.dumps(row, ensure_ascii=False, sort_keys=True)
    digest = hashlib.sha1(payload.encode("utf-8")).hexdigest()
    return f"sha1:{digest}"


def load_rows(paths: Iterable[Path]) -> Tuple[List[Dict[str, Any]], Counter]:
    rows: List[Dict[str, Any]] = []
    seen: Set[str] = set()
    src_counter: Counter = Counter()

    for path in paths:
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                key = stable_row_key(row)
                if key in seen:
                    continue
                seen.add(key)
                row["_source_file"] = path.name
                rows.append(row)
                src_counter[path.name] += 1
    return rows, src_counter


def task_signature_of(row: Dict[str, Any]) -> str:
    meta = row.get("task_metadata") or {}
    sig = meta.get("task_signature")
    if isinstance(sig, str) and sig:
        return sig
    task_id = row.get("task_id")
    if isinstance(task_id, str) and task_id:
        return task_id
    return stable_row_key(row)


def split_counts(total: int, train_ratio: float, val_ratio: float) -> Tuple[int, int, int]:
    train = int(round(total * train_ratio))
    val = int(round(total * val_ratio))
    if train + val > total:
        val = max(0, total - train)
    test = total - train - val
    return train, val, test


def main() -> None:
    args = parse_args()
    random.seed(args.seed)

    input_paths = [Path(path).resolve() for path in args.input_files]
    missing = [str(path) for path in input_paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing input files: {missing}")

    rows, src_counter = load_rows(input_paths)
    if not rows:
        raise RuntimeError("No rows loaded from input files.")

    task_map: Dict[str, TaskRecord] = {}
    for idx, row in enumerate(rows):
        sig = task_signature_of(row)
        bucket = bucket_of(row.get("start_node"), row.get("end_node"))
        pair_type = str(row.get("pair_type", "unknown"))

        if sig not in task_map:
            task_map[sig] = TaskRecord(signature=sig, bucket=bucket, row_ids=[idx], pair_types={pair_type})
        else:
            rec = task_map[sig]
            # Keep first bucket assignment; mixed-bucket for same signature should not happen.
            rec.row_ids.append(idx)
            rec.pair_types.add(pair_type)

    target_counts = {
        "B1": int(args.target_b1),
        "B2": int(args.target_b2),
        "B3": int(args.target_b3),
        "B4": int(args.target_b4),
    }

    candidates: Dict[str, List[TaskRecord]] = {bucket: [] for bucket in BUCKETS}
    for rec in task_map.values():
        if rec.bucket not in BUCKETS:
            continue
        if args.require_both_pair_types and not {"hard_invalid", "soft_suboptimal"}.issubset(rec.pair_types):
            continue
        candidates[rec.bucket].append(rec)

    deficits = {}
    for bucket in BUCKETS:
        if len(candidates[bucket]) < target_counts[bucket]:
            deficits[bucket] = {
                "required": target_counts[bucket],
                "available": len(candidates[bucket]),
                "shortfall": target_counts[bucket] - len(candidates[bucket]),
            }
    if deficits:
        raise RuntimeError(
            "Insufficient task signatures for strict bucket targets. "
            f"Deficits: {json.dumps(deficits, ensure_ascii=False)}"
        )

    selected_signatures: Dict[str, Set[str]] = {bucket: set() for bucket in BUCKETS}
    for bucket in BUCKETS:
        bucket_candidates = list(candidates[bucket])
        random.shuffle(bucket_candidates)
        chosen = bucket_candidates[: target_counts[bucket]]
        selected_signatures[bucket] = {rec.signature for rec in chosen}

    train_sigs: Set[str] = set()
    val_sigs: Set[str] = set()
    test_sigs: Set[str] = set()
    split_plan = {}
    for bucket in BUCKETS:
        signatures = list(selected_signatures[bucket])
        random.shuffle(signatures)
        train_n, val_n, test_n = split_counts(len(signatures), args.train_ratio, args.val_ratio)
        train_part = signatures[:train_n]
        val_part = signatures[train_n : train_n + val_n]
        test_part = signatures[train_n + val_n : train_n + val_n + test_n]
        train_sigs.update(train_part)
        val_sigs.update(val_part)
        test_sigs.update(test_part)
        split_plan[bucket] = {
            "total": len(signatures),
            "train": len(train_part),
            "val": len(val_part),
            "test": len(test_part),
        }

    selected_all = train_sigs | val_sigs | test_sigs
    split_rows = {"train": [], "val": [], "test": [], "all": []}
    for row in rows:
        sig = task_signature_of(row)
        if sig not in selected_all:
            continue
        out_row = dict(row)
        out_row.pop("_source_file", None)
        split_rows["all"].append(out_row)
        if sig in train_sigs:
            split_rows["train"].append(out_row)
        elif sig in val_sigs:
            split_rows["val"].append(out_row)
        elif sig in test_sigs:
            split_rows["test"].append(out_row)

    prefix = Path(args.output_prefix)
    out_paths = {
        "all": prefix.with_name(f"{prefix.name}_all.jsonl"),
        "train": prefix.with_name(f"{prefix.name}_train.jsonl"),
        "val": prefix.with_name(f"{prefix.name}_val.jsonl"),
        "test": prefix.with_name(f"{prefix.name}_test.jsonl"),
        "report": prefix.with_name(f"{prefix.name}_report.json"),
        "task_map": prefix.with_name(f"{prefix.name}_tasks.json"),
    }

    for split in ("all", "train", "val", "test"):
        with out_paths[split].open("w", encoding="utf-8") as f:
            for row in split_rows[split]:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    task_dump = {
        "train_signatures": sorted(train_sigs),
        "val_signatures": sorted(val_sigs),
        "test_signatures": sorted(test_sigs),
        "bucket_signatures": {bucket: sorted(list(sigs)) for bucket, sigs in selected_signatures.items()},
    }
    out_paths["task_map"].write_text(json.dumps(task_dump, ensure_ascii=False, indent=2), encoding="utf-8")

    def summarize(rows_subset: List[Dict[str, Any]]) -> Dict[str, Any]:
        pair_dist = Counter(str(row.get("pair_type", "unknown")) for row in rows_subset)
        bucket_dist = Counter(bucket_of(row.get("start_node"), row.get("end_node")) for row in rows_subset)
        signatures = {task_signature_of(row) for row in rows_subset}
        return {
            "rows": len(rows_subset),
            "unique_tasks": len(signatures),
            "pair_type_distribution": dict(pair_dist),
            "bucket_distribution": dict(bucket_dist),
        }

    report = {
        "input_files": [str(path) for path in input_paths],
        "source_row_counts_after_dedup": dict(src_counter),
        "rows_loaded_total": len(rows),
        "tasks_loaded_total": len(task_map),
        "require_both_pair_types": args.require_both_pair_types,
        "targets": target_counts,
        "split_plan_by_bucket": split_plan,
        "split_summaries": {
            "all": summarize(split_rows["all"]),
            "train": summarize(split_rows["train"]),
            "val": summarize(split_rows["val"]),
            "test": summarize(split_rows["test"]),
        },
        "output_files": {k: str(v) for k, v in out_paths.items()},
    }
    out_paths["report"].write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print("=" * 72)
    print("P0 bucketed dataset build complete")
    print("=" * 72)
    print(f"rows_loaded_total: {report['rows_loaded_total']}")
    print(f"tasks_loaded_total: {report['tasks_loaded_total']}")
    print(f"all split summary: {report['split_summaries']['all']}")
    print(f"train split summary: {report['split_summaries']['train']}")
    print(f"val split summary: {report['split_summaries']['val']}")
    print(f"test split summary: {report['split_summaries']['test']}")
    print(f"report: {out_paths['report']}")


if __name__ == "__main__":
    main()
