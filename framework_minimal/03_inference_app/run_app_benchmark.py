from __future__ import annotations

import argparse
import asyncio
import contextlib
import csv
import io
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.agent.coordinator import RobotCoordinator
from app.search.state import SearchAction
from app.search.verifier import PlanSearchVerifier, SearchConstraints


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a dry-run benchmark for greedy/retry/search planning modes.")
    parser.add_argument(
        "--tasks-file",
        type=Path,
        default=PROJECT_ROOT / "data" / "app_benchmark_tasks.json",
        help="JSON file containing benchmark prompts and optional expected source/handover constraints.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "app_benchmark_results",
        help="Directory where per-run and aggregate benchmark artifacts will be written.",
    )
    parser.add_argument(
        "--modes",
        nargs="+",
        default=["greedy", "retry", "search"],
        choices=["greedy", "retry", "search"],
        help="Planning modes to benchmark.",
    )
    parser.add_argument("--max-retries", type=int, default=2, help="Retry budget for retry mode.")
    parser.add_argument("--search-iterations", type=int, default=96, help="MCTS iterations for search mode.")
    parser.add_argument("--search-depth", type=int, default=18, help="MCTS rollout depth for search mode.")
    parser.add_argument("--dry-run", action="store_true", default=True, help="Use symbolic dry-run execution.")
    parser.add_argument(
        "--capture-stdout",
        action="store_true",
        help="Capture coordinator stdout into per-run logs instead of printing the planner stream to console.",
    )
    return parser.parse_args()


def load_tasks(tasks_file: Path) -> list[dict[str, Any]]:
    with tasks_file.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, list):
        raise ValueError(f"Benchmark tasks file must contain a JSON list: {tasks_file}")
    return payload


def infer_constraints(task: dict[str, Any]) -> SearchConstraints:
    sources: tuple[str, ...]
    handover_zones: tuple[str, ...]

    expected_source = task.get("expected_source")
    expected_handover = task.get("expected_handover")

    if isinstance(expected_source, str) and expected_source.strip():
        sources = (expected_source.strip(),)
    else:
        sources = ("Starbucks_Outdoor", "Vending_Machine_2F")

    if isinstance(expected_handover, str) and expected_handover.strip():
        handover_zones = (expected_handover.strip(),)
    else:
        handover_zones = ("Info_Hub_4F", "Elevator_Lobby_4F")

    return SearchConstraints(allowed_sources=sources, allowed_handover_zones=handover_zones)


def extract_plan_metadata(plan_steps: list[dict[str, Any]], constraints: SearchConstraints) -> dict[str, Any]:
    verifier = PlanSearchVerifier(constraints=constraints)
    state = verifier.initial_state()

    pickup_source = ""
    handover_zone = ""
    delivered_explicitly = False
    handover_count = 0

    for step in plan_steps:
        action = SearchAction(
            robot=str(step.get("robot", "")),
            action=str(step.get("action", "")),
            target=str(step.get("target", "")),
            details=str(step.get("details", "")),
        )
        if action.action == "grasp" and not pickup_source:
            pickup_source = state.chassis_location
        if action.action == "transfer_to_dog":
            handover_zone = state.chassis_location
            handover_count += 1
        if action.action == "deliver":
            delivered_explicitly = True

        transition = verifier.apply_action(state, action)
        if not transition.is_valid:
            break
        state = transition.next_state

    return {
        "pickup_source": pickup_source or None,
        "handover_zone": handover_zone or None,
        "handover_count": handover_count,
        "explicit_deliver_step": delivered_explicitly,
    }


def summarize_run(
    task: dict[str, Any],
    mode: str,
    trace_history: dict[str, Any],
    runtime_sec: float,
) -> dict[str, Any]:
    attempts = trace_history.get("attempts", [])
    attempt = attempts[-1] if attempts else {}
    executed_plan = attempt.get("executed_plan", []) or []
    planner_plan = attempt.get("planner_plan", []) or []
    trace_constraints = trace_history.get("meta", {}).get("task_constraints", {})
    constraints = SearchConstraints(
        allowed_sources=tuple(trace_constraints.get("allowed_sources", ("Starbucks_Outdoor", "Vending_Machine_2F"))),
        allowed_handover_zones=tuple(
            trace_constraints.get("allowed_handover_zones", ("Info_Hub_4F", "Elevator_Lobby_4F"))
        ),
    )

    plan_metadata = extract_plan_metadata(executed_plan, constraints) if executed_plan else {
        "pickup_source": None,
        "handover_zone": None,
        "handover_count": 0,
        "explicit_deliver_step": False,
    }

    success = trace_history.get("meta", {}).get("status") in {"success", "success_dry_run"}
    selected_valid = attempt.get("selected_plan_valid")
    if selected_valid is None:
        selected_valid = attempt.get("planner_plan_valid")

    result = {
        "task_id": task["id"],
        "category": task.get("category", ""),
        "prompt": task["prompt"],
        "mode": mode,
        "success": success,
        "trace_status": trace_history.get("meta", {}).get("status", ""),
        "attempts": len(attempts),
        "runtime_sec": round(runtime_sec, 3),
        "planner_valid": attempt.get("planner_plan_valid"),
        "selected_valid": selected_valid,
        "planner_plan_length": len(planner_plan),
        "executed_plan_length": len(executed_plan),
        "selected_plan_source": attempt.get("selected_plan_source"),
        "planner_error": attempt.get("planner_plan_error", ""),
        "selected_error": attempt.get("selected_plan_error") or attempt.get("error", ""),
        "pickup_source": plan_metadata["pickup_source"],
        "handover_zone": plan_metadata["handover_zone"],
        "handover_count": plan_metadata["handover_count"],
        "explicit_deliver_step": plan_metadata["explicit_deliver_step"],
        "expected_source": task.get("expected_source"),
        "expected_handover": task.get("expected_handover"),
        "source_match": None,
        "handover_match": None,
        "failure_reason": "",
    }

    if task.get("expected_source"):
        result["source_match"] = result["pickup_source"] == task["expected_source"]
    if task.get("expected_handover"):
        result["handover_match"] = result["handover_zone"] == task["expected_handover"]
    if not success:
        result["failure_reason"] = attempt.get("selected_plan_error") or attempt.get("error") or attempt.get("planner_plan_error", "")

    return result


def aggregate_results(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    summary_rows: list[dict[str, Any]] = []
    modes = sorted({row["mode"] for row in results})

    for mode in modes:
        rows = [row for row in results if row["mode"] == mode]
        source_rows = [row for row in rows if row["source_match"] is not None]
        handover_rows = [row for row in rows if row["handover_match"] is not None]

        summary_rows.append(
            {
                "mode": mode,
                "num_tasks": len(rows),
                "success_rate": round(sum(1 for row in rows if row["success"]) / max(len(rows), 1), 3),
                "planner_valid_rate": round(
                    sum(1 for row in rows if row["planner_valid"]) / max(len(rows), 1),
                    3,
                ),
                "selected_valid_rate": round(
                    sum(1 for row in rows if row["selected_valid"]) / max(len(rows), 1),
                    3,
                ),
                "source_match_rate": round(
                    sum(1 for row in source_rows if row["source_match"]) / max(len(source_rows), 1),
                    3,
                ) if source_rows else None,
                "handover_match_rate": round(
                    sum(1 for row in handover_rows if row["handover_match"]) / max(len(handover_rows), 1),
                    3,
                ) if handover_rows else None,
                "avg_plan_length": round(statistics.mean(row["executed_plan_length"] for row in rows), 3),
                "avg_runtime_sec": round(statistics.mean(row["runtime_sec"] for row in rows), 3),
                "avg_handover_count": round(statistics.mean(row["handover_count"] for row in rows), 3),
                "search_selected_rate": round(
                    sum(1 for row in rows if row["selected_plan_source"] == "search") / max(len(rows), 1),
                    3,
                ) if mode == "search" else None,
            }
        )

    return summary_rows


def write_markdown_summary(output_path: Path, summary_rows: list[dict[str, Any]], results: list[dict[str, Any]]) -> None:
    lines = [
        "# App Benchmark Summary",
        "",
        "## Aggregate",
        "",
        "| Mode | Tasks | Success | Planner Valid | Selected Valid | Source Match | Handover Match | Avg Plan Len | Avg Runtime (s) | Search Selected |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]

    for row in summary_rows:
        lines.append(
            "| {mode} | {num_tasks} | {success_rate:.3f} | {planner_valid_rate:.3f} | {selected_valid_rate:.3f} | {source_match} | {handover_match} | {avg_plan_length:.3f} | {avg_runtime_sec:.3f} | {search_selected} |".format(
                mode=row["mode"],
                num_tasks=row["num_tasks"],
                success_rate=row["success_rate"],
                planner_valid_rate=row["planner_valid_rate"],
                selected_valid_rate=row["selected_valid_rate"],
                source_match="-" if row["source_match_rate"] is None else f"{row['source_match_rate']:.3f}",
                handover_match="-" if row["handover_match_rate"] is None else f"{row['handover_match_rate']:.3f}",
                avg_plan_length=row["avg_plan_length"],
                avg_runtime_sec=row["avg_runtime_sec"],
                search_selected="-" if row["search_selected_rate"] is None else f"{row['search_selected_rate']:.3f}",
            )
        )

    lines.extend(
        [
            "",
            "## Per Task",
            "",
            "| Task | Mode | Success | Source | Handover | Source Match | Handover Match | Plan Len | Runtime (s) | Error |",
            "| --- | --- | --- | --- | --- | --- | --- | ---: | ---: | --- |",
        ]
    )

    for row in results:
        lines.append(
            "| {task_id} | {mode} | {success} | {pickup_source} | {handover_zone} | {source_match} | {handover_match} | {executed_plan_length} | {runtime_sec:.3f} | {error} |".format(
                task_id=row["task_id"],
                mode=row["mode"],
                success="yes" if row["success"] else "no",
                pickup_source=row["pickup_source"] or "-",
                handover_zone=row["handover_zone"] or "-",
                source_match="-" if row["source_match"] is None else ("yes" if row["source_match"] else "no"),
                handover_match="-" if row["handover_match"] is None else ("yes" if row["handover_match"] else "no"),
                executed_plan_length=row["executed_plan_length"],
                runtime_sec=row["runtime_sec"],
                error=(row["failure_reason"] or "-").replace("|", "/"),
            )
        )

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


async def run_single_task(
    task: dict[str, Any],
    mode: str,
    args: argparse.Namespace,
    run_logs_dir: Path,
) -> dict[str, Any]:
    coordinator = RobotCoordinator(
        planning_mode=mode,
        max_retries=args.max_retries,
        search_iterations=args.search_iterations,
        search_rollout_depth=args.search_depth,
        dry_run=args.dry_run,
    )

    start = time.perf_counter()
    if args.capture_stdout:
        capture = io.StringIO()
        with contextlib.redirect_stdout(capture):
            await coordinator.run(task["prompt"])
        (run_logs_dir / f"{task['id']}_{mode}.stdout.txt").write_text(capture.getvalue(), encoding="utf-8")
    else:
        await coordinator.run(task["prompt"])
    runtime_sec = time.perf_counter() - start

    return summarize_run(task, mode, coordinator.trace_history, runtime_sec)


async def main() -> None:
    args = parse_args()
    tasks = load_tasks(args.tasks_file)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    run_logs_dir = args.output_dir / "stdout_logs"
    run_logs_dir.mkdir(parents=True, exist_ok=True)

    results: list[dict[str, Any]] = []

    for task in tasks:
        for mode in args.modes:
            print(f"[benchmark] running task={task['id']} mode={mode}")
            row = await run_single_task(task, mode, args, run_logs_dir)
            results.append(row)

    summary_rows = aggregate_results(results)

    json_path = args.output_dir / "benchmark_results.json"
    csv_path = args.output_dir / "benchmark_results.csv"
    summary_json_path = args.output_dir / "benchmark_summary.json"
    summary_md_path = args.output_dir / "benchmark_summary.md"

    json_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    summary_json_path.write_text(json.dumps(summary_rows, indent=2, ensure_ascii=False), encoding="utf-8")

    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)

    write_markdown_summary(summary_md_path, summary_rows, results)

    print(f"[benchmark] wrote detailed results to {json_path}")
    print(f"[benchmark] wrote aggregate summary to {summary_md_path}")


if __name__ == "__main__":
    asyncio.run(main())
