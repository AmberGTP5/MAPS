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
import traceback
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.agent.graph_coordinator import GraphTaskCoordinator, load_graph_tasks


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark graph-mode greedy/retry/search on full-graph tasks.")
    parser.add_argument("--tasks-file", type=Path, required=True, help="JSON or JSONL task file.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "graph_benchmark_results",
        help="Where benchmark artifacts will be written.",
    )
    parser.add_argument(
        "--modes",
        nargs="+",
        default=["greedy", "retry", "search"],
        choices=["greedy", "retry", "search"],
    )
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument(
        "--guidance-mode",
        type=str,
        default="route_hint",
        choices=["none", "feasibility", "route_hint"],
        help="How much symbolic planner guidance to inject into the prompt.",
    )
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--max-samples", type=int, default=0, help="0 means all rows from start-index.")
    parser.add_argument("--capture-stdout", action="store_true")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint rows already written to the output dir.")
    return parser.parse_args()


def slice_tasks(rows: list[dict[str, Any]], start_index: int, max_samples: int) -> list[dict[str, Any]]:
    selected = rows[start_index:]
    if max_samples > 0:
        selected = selected[:max_samples]
    return selected


def summarize_attempt(
    task: dict[str, Any],
    mode: str,
    guidance_mode: str,
    trace_history: dict[str, Any],
    runtime_sec: float,
) -> dict[str, Any]:
    attempts = trace_history.get("attempts", [])
    attempt = attempts[-1] if attempts else {}
    planner_eval = attempt.get("planner_eval", {}) or {}
    selected_eval = attempt.get("selected_eval", {}) or {}
    search_summary = attempt.get("search_summary", {}) or {}

    return {
        "task_id": task.get("task_id") or task.get("id"),
        "task_profile": task.get("sampling_profile")
        or task.get("category")
        or task.get("task_metadata", {}).get("task_profile", "unknown"),
        "mode": mode,
        "guidance_mode": guidance_mode,
        "success": trace_history.get("meta", {}).get("status") == "success_dry_run",
        "trace_status": trace_history.get("meta", {}).get("status", ""),
        "attempts": len(attempts),
        "runtime_sec": round(runtime_sec, 3),
        "planner_parse_ok": bool(planner_eval.get("parse_ok")),
        "planner_schema_pass": bool(planner_eval.get("schema_pass")),
        "planner_valid": bool(planner_eval.get("is_valid")),
        "selected_valid": bool(selected_eval.get("is_valid")),
        "planner_robot": planner_eval.get("selected_robot"),
        "selected_robot": selected_eval.get("selected_robot"),
        "selected_plan_source": attempt.get("selected_plan_source"),
        "planner_trajectory_cost": planner_eval.get("trajectory_cost"),
        "selected_trajectory_cost": selected_eval.get("trajectory_cost"),
        "planner_step_count": planner_eval.get("step_count", 0),
        "selected_step_count": selected_eval.get("step_count", 0),
        "planner_handover_count": planner_eval.get("handover_count", 0),
        "selected_handover_count": selected_eval.get("handover_count", 0),
        "failure_reason": "" if trace_history.get("meta", {}).get("status") == "success_dry_run" else (
            selected_eval.get("error_details")
            or planner_eval.get("error_details")
            or selected_eval.get("parse_error")
            or planner_eval.get("parse_error")
            or "unknown_failure"
        ),
        "search_preferred_robot": search_summary.get("preferred_robot"),
        "search_selected_robot": search_summary.get("selected_robot"),
    }


def aggregate_results(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    summary_rows: list[dict[str, Any]] = []
    modes = sorted({row["mode"] for row in results})

    for mode in modes:
        rows = [row for row in results if row["mode"] == mode]
        cost_values = [float(row["selected_trajectory_cost"]) for row in rows if row["selected_trajectory_cost"] is not None]
        summary_rows.append(
            {
                "mode": mode,
                "num_tasks": len(rows),
                "success_rate": round(sum(1 for row in rows if row["success"]) / max(len(rows), 1), 3),
                "planner_parse_rate": round(sum(1 for row in rows if row["planner_parse_ok"]) / max(len(rows), 1), 3),
                "planner_schema_rate": round(sum(1 for row in rows if row["planner_schema_pass"]) / max(len(rows), 1), 3),
                "planner_valid_rate": round(sum(1 for row in rows if row["planner_valid"]) / max(len(rows), 1), 3),
                "selected_valid_rate": round(sum(1 for row in rows if row["selected_valid"]) / max(len(rows), 1), 3),
                "avg_selected_cost": round(statistics.mean(cost_values), 3) if cost_values else None,
                "avg_selected_steps": round(statistics.mean(row["selected_step_count"] for row in rows), 3),
                "avg_runtime_sec": round(statistics.mean(row["runtime_sec"] for row in rows), 3),
                "search_selected_rate": round(
                    sum(1 for row in rows if str(row["selected_plan_source"]).startswith("search")) / max(len(rows), 1),
                    3,
                ) if mode == "search" else None,
            }
        )
    return summary_rows


def write_markdown_summary(
    output_path: Path,
    summary_rows: list[dict[str, Any]],
    results: list[dict[str, Any]],
) -> None:
    lines = [
        "# Graph Benchmark Summary",
        "",
        "## Aggregate",
        "",
        "| Mode | Tasks | Success | Planner Parse | Planner Schema | Planner Valid | Selected Valid | Avg Selected Cost | Avg Selected Steps | Avg Runtime (s) | Search Selected |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]

    for row in summary_rows:
        lines.append(
            "| {mode} | {num_tasks} | {success_rate:.3f} | {planner_parse_rate:.3f} | {planner_schema_rate:.3f} | {planner_valid_rate:.3f} | {selected_valid_rate:.3f} | {avg_selected_cost} | {avg_selected_steps:.3f} | {avg_runtime_sec:.3f} | {search_selected} |".format(
                mode=row["mode"],
                num_tasks=row["num_tasks"],
                success_rate=row["success_rate"],
                planner_parse_rate=row["planner_parse_rate"],
                planner_schema_rate=row["planner_schema_rate"],
                planner_valid_rate=row["planner_valid_rate"],
                selected_valid_rate=row["selected_valid_rate"],
                avg_selected_cost="-" if row["avg_selected_cost"] is None else f"{row['avg_selected_cost']:.3f}",
                avg_selected_steps=row["avg_selected_steps"],
                avg_runtime_sec=row["avg_runtime_sec"],
                search_selected="-" if row["search_selected_rate"] is None else f"{row['search_selected_rate']:.3f}",
            )
        )

    lines.extend(
        [
            "",
            "## Per Task",
            "",
            "| Task | Profile | Mode | Success | Planner Valid | Selected Valid | Planner Robot | Selected Robot | Selected Source | Cost | Steps | Error |",
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- | ---: | ---: | --- |",
        ]
    )

    for row in results:
        lines.append(
            "| {task_id} | {task_profile} | {mode} | {success} | {planner_valid} | {selected_valid} | {planner_robot} | {selected_robot} | {selected_plan_source} | {selected_trajectory_cost} | {selected_step_count} | {failure_reason} |".format(
                task_id=row["task_id"],
                task_profile=row["task_profile"],
                mode=row["mode"],
                success="yes" if row["success"] else "no",
                planner_valid="yes" if row["planner_valid"] else "no",
                selected_valid="yes" if row["selected_valid"] else "no",
                planner_robot=row["planner_robot"] or "-",
                selected_robot=row["selected_robot"] or "-",
                selected_plan_source=row["selected_plan_source"] or "-",
                selected_trajectory_cost="-" if row["selected_trajectory_cost"] is None else row["selected_trajectory_cost"],
                selected_step_count=row["selected_step_count"],
                failure_reason=(row["failure_reason"] or "-").replace("|", "/"),
            )
        )

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


async def run_single_task(
    task: dict[str, Any],
    mode: str,
    args: argparse.Namespace,
    run_logs_dir: Path,
) -> dict[str, Any]:
    coordinator = GraphTaskCoordinator(
        planning_mode=mode,
        max_retries=args.max_retries,
        dry_run=True,
        guidance_mode=args.guidance_mode,
    )

    start = time.perf_counter()
    if args.capture_stdout:
        capture = io.StringIO()
        with contextlib.redirect_stdout(capture):
            await coordinator.run_task(task)
        (run_logs_dir / f"{task.get('task_id', task.get('id', 'task'))}_{mode}.stdout.txt").write_text(
            capture.getvalue(),
            encoding="utf-8",
        )
    else:
        await coordinator.run_task(task)
    runtime_sec = time.perf_counter() - start

    return summarize_attempt(task, mode, args.guidance_mode, coordinator.trace_history, runtime_sec)


def build_exception_row(
    task: dict[str, Any],
    mode: str,
    guidance_mode: str,
    runtime_sec: float,
    exc: Exception,
) -> dict[str, Any]:
    task_id = task.get("task_id") or task.get("id")
    task_profile = task.get("sampling_profile") or task.get("category") or task.get("task_metadata", {}).get("task_profile", "unknown")
    return {
        "task_id": task_id,
        "task_profile": task_profile,
        "mode": mode,
        "guidance_mode": guidance_mode,
        "success": False,
        "trace_status": "runtime_exception",
        "attempts": 0,
        "runtime_sec": round(runtime_sec, 3),
        "planner_parse_ok": False,
        "planner_schema_pass": False,
        "planner_valid": False,
        "selected_valid": False,
        "planner_robot": None,
        "selected_robot": None,
        "selected_plan_source": "runtime_exception",
        "planner_trajectory_cost": None,
        "selected_trajectory_cost": None,
        "planner_step_count": 0,
        "selected_step_count": 0,
        "planner_handover_count": 0,
        "selected_handover_count": 0,
        "failure_reason": f"{type(exc).__name__}: {exc}",
        "search_preferred_robot": None,
        "search_selected_robot": None,
    }


def result_key(row: dict[str, Any]) -> tuple[str, str, str]:
    return (
        str(row.get("task_id")),
        str(row.get("mode")),
        str(row.get("guidance_mode")),
    )


def persist_results(output_dir: Path, results: list[dict[str, Any]]) -> None:
    checkpoint_path = output_dir / "benchmark_results.jsonl"
    json_path = output_dir / "benchmark_results.json"
    csv_path = output_dir / "benchmark_results.csv"
    summary_json_path = output_dir / "benchmark_summary.json"
    summary_md_path = output_dir / "benchmark_summary.md"

    checkpoint_lines = [json.dumps(row, ensure_ascii=False) for row in results]
    checkpoint_path.write_text("\n".join(checkpoint_lines) + ("\n" if checkpoint_lines else ""), encoding="utf-8")
    json_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")

    if not results:
        summary_json_path.write_text("[]\n", encoding="utf-8")
        summary_md_path.write_text("# Graph Benchmark Summary\n\nNo results.\n", encoding="utf-8")
        return

    summary_rows = aggregate_results(results)
    summary_json_path.write_text(json.dumps(summary_rows, indent=2, ensure_ascii=False), encoding="utf-8")

    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)

    write_markdown_summary(summary_md_path, summary_rows, results)


def load_checkpoint_rows(checkpoint_path: Path) -> list[dict[str, Any]]:
    if not checkpoint_path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with checkpoint_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


async def main() -> None:
    args = parse_args()
    tasks = slice_tasks(load_graph_tasks(args.tasks_file), args.start_index, args.max_samples)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    run_logs_dir = args.output_dir / "stdout_logs"
    run_logs_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = args.output_dir / "benchmark_results.jsonl"
    results: list[dict[str, Any]] = load_checkpoint_rows(checkpoint_path) if args.resume else []
    seen_keys = {result_key(row) for row in results}

    for task in tasks:
        task_key = task.get("task_id") or task.get("id")
        for mode in args.modes:
            row_key = (str(task_key), str(mode), str(args.guidance_mode))
            if row_key in seen_keys:
                print(f"[graph-benchmark] skipping existing task={task_key} mode={mode} guidance={args.guidance_mode}")
                continue
            print(f"[graph-benchmark] running task={task_key} mode={mode}")
            start = time.perf_counter()
            try:
                row = await run_single_task(task, mode, args, run_logs_dir)
            except Exception as exc:
                runtime_sec = time.perf_counter() - start
                traceback.print_exc()
                row = build_exception_row(task, mode, args.guidance_mode, runtime_sec, exc)
            results.append(row)
            seen_keys.add(row_key)
            persist_results(args.output_dir, results)

    print(f"[graph-benchmark] wrote detailed results to {args.output_dir / 'benchmark_results.json'}")
    print(f"[graph-benchmark] wrote aggregate summary to {args.output_dir / 'benchmark_summary.md'}")


if __name__ == "__main__":
    asyncio.run(main())
