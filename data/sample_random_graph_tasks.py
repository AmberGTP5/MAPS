from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any

from graph_engine import build_default_graph_engine
from oracle_planner import OraclePlanner
from task_generator_english import EnglishTaskGenerator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sample random graph-grounded tasks from the dataset.")
    parser.add_argument("--num-samples", type=int, default=20, help="Number of unique tasks to sample.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("sampled_graph_tasks.jsonl"),
        help="Where to write sampled tasks.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--with-oracle",
        action="store_true",
        help="Attach a valid oracle plan for each sampled task.",
    )
    parser.add_argument(
        "--max-attempts",
        type=int,
        default=2000,
        help="Maximum total sampling attempts before stopping.",
    )
    return parser.parse_args()


def compact_task_record(task: dict[str, Any]) -> dict[str, Any]:
    return {
        "task_id": task["task_id"],
        "instruction": task["instruction"],
        "start": task["start"],
        "end": task["end"],
        "item": task["item"],
        "sampling_profile": task.get("sampling_profile"),
        "preferred_robot_hint": task.get("preferred_robot_hint"),
        "force_preferred_if_feasible": task.get("force_preferred_if_feasible", False),
        "task_metadata": task.get("task_metadata", {}),
        "shortest_path_length": task.get("shortest_path_length"),
        "found_k_paths": task.get("found_k_paths"),
        "reference_paths": task.get("reference_paths", []),
        "prompt": task.get("prompt", ""),
    }


def main() -> None:
    args = parse_args()
    random.seed(args.seed)

    engine = build_default_graph_engine()
    generator = EnglishTaskGenerator(engine)
    oracle = OraclePlanner(engine) if args.with_oracle else None

    seen_signatures: set[str] = set()
    profile_counter: Counter[str] = Counter()
    robot_counter: Counter[str] = Counter()
    records: list[dict[str, Any]] = []

    attempts = 0
    while len(records) < args.num_samples and attempts < args.max_attempts:
        attempts += 1
        task = generator.generate_task(excluded_task_signatures=seen_signatures)
        if not task:
            continue

        task_signature = task.get("task_metadata", {}).get("task_signature")
        if task_signature:
            seen_signatures.add(task_signature)

        record = compact_task_record(task)
        profile_counter[record.get("sampling_profile") or "unknown"] += 1

        if oracle is not None:
            oracle_result = oracle.plan_task(
                start_node=task["start"],
                end_node=task["end"],
                target_item=task["item"],
                preferred_robot=task.get("preferred_robot_hint"),
                force_preferred_if_feasible=task.get("force_preferred_if_feasible", False),
            )
            if oracle_result is None:
                continue

            robot_counter[oracle_result.robot] += 1
            record["oracle"] = {
                "robot": oracle_result.robot,
                "thought": oracle_result.thought,
                "plan": oracle_result.plan,
                "trajectory_cost": oracle_result.feedback.get("trajectory_cost"),
                "step_count": len(oracle_result.plan),
                "path": oracle_result.path,
                "validator_feedback": oracle_result.feedback,
            }

        records.append(record)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    summary = {
        "output": str(args.output.resolve()),
        "num_samples_requested": args.num_samples,
        "num_samples_written": len(records),
        "sampling_attempts": attempts,
        "with_oracle": args.with_oracle,
        "profile_distribution": dict(profile_counter),
        "oracle_robot_distribution": dict(robot_counter),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
