import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
import re
from typing import Dict, Iterable, List, Optional, Tuple


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build an unseen evaluation set by excluding task signatures present in the training split."
    )
    parser.add_argument("--train-file", required=True, help="Training jsonl used for model fitting.")
    parser.add_argument(
        "--candidate-files",
        nargs="+",
        required=True,
        help="One or more candidate jsonl files to sample unseen tasks from.",
    )
    parser.add_argument("--output-file", required=True, help="Output jsonl path for unseen evaluation records.")
    parser.add_argument("--report-file", default="", help="Optional report json path.")
    parser.add_argument("--target-size", type=int, default=580, help="Number of unseen rows to output.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for deterministic sampling.")
    parser.add_argument(
        "--building-scope",
        choices=("any", "same", "cross"),
        default="any",
        help="Optional filter by start/end building relation: any, same-building, cross-building.",
    )
    parser.add_argument(
        "--dedupe-by-signature",
        action="store_true",
        help="Keep at most one row per task signature in the final output.",
    )
    return parser.parse_args()


def read_jsonl(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def extract_signature(record: dict) -> Optional[str]:
    top_meta = record.get("task_metadata") or {}
    nested_task = record.get("task") or {}
    nested_meta = nested_task.get("task_metadata") or {}

    signature = top_meta.get("task_signature") or nested_meta.get("task_signature")
    if signature:
        return signature

    start = record.get("start_node") or nested_task.get("start_node")
    end = record.get("end_node") or nested_task.get("end_node")
    item = record.get("target_item") or nested_task.get("target_item")
    if start and end and item:
        return f"{start}|{end}|{item}"
    return None


def pick_pair_type(record: dict) -> str:
    pair_type = record.get("pair_type")
    if pair_type:
        return str(pair_type)
    pref = record.get("preference_strength") or {}
    if pref.get("pair_type"):
        return str(pref["pair_type"])
    return "unknown"


_BLD_RE = re.compile(r"^([A-Z]\d+)_")


def _infer_building_from_node(node_id: Optional[str]) -> Optional[str]:
    if not node_id:
        return None
    m = _BLD_RE.match(node_id)
    if m:
        return m.group(1)
    return None


def building_relation(record: dict) -> Optional[str]:
    task = record.get("task") or {}
    top_meta = record.get("task_metadata") or {}
    nested_meta = task.get("task_metadata") or {}

    start_node = record.get("start_node") or task.get("start_node")
    end_node = record.get("end_node") or task.get("end_node")

    start_building = (
        top_meta.get("start_building")
        or nested_meta.get("start_building")
        or _infer_building_from_node(start_node)
    )
    end_building = (
        top_meta.get("end_building")
        or nested_meta.get("end_building")
        or _infer_building_from_node(end_node)
    )

    if not start_building or not end_building:
        return None
    return "same" if start_building == end_building else "cross"


def main() -> None:
    args = parse_args()

    train_path = Path(args.train_file)
    candidate_paths = [Path(p) for p in args.candidate_files]
    output_path = Path(args.output_file)
    report_path = Path(args.report_file) if args.report_file else output_path.with_suffix(".report.json")

    if not train_path.exists():
        raise FileNotFoundError(f"train file not found: {train_path}")
    for path in candidate_paths:
        if not path.exists():
            raise FileNotFoundError(f"candidate file not found: {path}")

    train_signatures = set()
    for rec in read_jsonl(train_path):
        sig = extract_signature(rec)
        if sig:
            train_signatures.add(sig)

    source_stats: Dict[str, Counter] = defaultdict(Counter)
    unseen_records: List[Tuple[str, dict, str]] = []

    for path in candidate_paths:
        src = str(path)
        for rec in read_jsonl(path):
            source_stats[src]["rows_total"] += 1
            sig = extract_signature(rec)
            if not sig:
                source_stats[src]["missing_signature"] += 1
                continue
            if sig in train_signatures:
                source_stats[src]["seen_in_train"] += 1
                continue
            relation = building_relation(rec)
            if args.building_scope != "any":
                if relation is None:
                    source_stats[src]["missing_building_relation"] += 1
                    continue
                if relation != args.building_scope:
                    source_stats[src]["filtered_by_building_scope"] += 1
                    continue
            if "prompt" not in rec or not rec.get("prompt"):
                source_stats[src]["missing_prompt"] += 1
                continue
            unseen_records.append((sig, rec, src))
            source_stats[src]["eligible_unseen"] += 1

    if args.dedupe_by_signature:
        unique_pool: Dict[str, Tuple[dict, str]] = {}
        for sig, rec, src in unseen_records:
            if sig not in unique_pool:
                unique_pool[sig] = (rec, src)
        pool_items = [(sig, pair[0], pair[1]) for sig, pair in unique_pool.items()]
    else:
        pool_items = unseen_records

    if len(pool_items) < args.target_size:
        raise RuntimeError(
            f"Not enough unseen candidates. Need {args.target_size}, found {len(pool_items)}."
        )

    rng = random.Random(args.seed)
    if args.dedupe_by_signature:
        keys = sorted([sig for sig, _, _ in pool_items])
        rng.shuffle(keys)
        selected_keys = keys[: args.target_size]
        selected_lookup = {sig: (rec, src) for sig, rec, src in pool_items}
        selected = [(sig, *selected_lookup[sig]) for sig in selected_keys]
    else:
        indices = list(range(len(pool_items)))
        rng.shuffle(indices)
        selected = [pool_items[i] for i in indices[: args.target_size]]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for _, rec, _ in selected:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    pair_type_counter = Counter()
    source_counter = Counter()
    signature_counter = Counter()
    for sig, rec, src in selected:
        pair_type_counter[pick_pair_type(rec)] += 1
        source_counter[src] += 1
        signature_counter[sig] += 1

    report = {
        "train_file": str(train_path),
        "candidate_files": [str(p) for p in candidate_paths],
        "target_size": args.target_size,
        "seed": args.seed,
        "building_scope": args.building_scope,
        "dedupe_by_signature": args.dedupe_by_signature,
        "train_unique_signatures": len(train_signatures),
        "candidate_pool_size": len(pool_items),
        "selected_size": len(selected),
        "selected_unique_signatures": len(signature_counter),
        "selected_pair_type_distribution": dict(pair_type_counter),
        "selected_source_distribution": dict(source_counter),
        "source_stats": {src: dict(cnt) for src, cnt in source_stats.items()},
    }

    with report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print("============================================================")
    print("Unseen eval set construction complete")
    print("============================================================")
    print(f"Train signatures: {len(train_signatures)}")
    print(f"Candidate pool: {len(pool_items)}")
    print(f"Selected: {len(selected)}")
    print(f"Selected unique signatures: {len(signature_counter)}")
    print(f"Output: {output_path}")
    print(f"Report: {report_path}")
    print("Pair types:", dict(pair_type_counter))
    print("============================================================")


if __name__ == "__main__":
    main()
