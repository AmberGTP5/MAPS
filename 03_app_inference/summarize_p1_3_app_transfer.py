from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize ProcTHOR/ALFWorld transfer runs produced by run_app_benchmark.py"
    )
    parser.add_argument(
        "--runs",
        nargs="+",
        type=Path,
        required=True,
        help="Each run directory should contain benchmark_summary.json and benchmark_results.json",
    )
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=Path("paper_artifacts/p1_3_transfer_summary"),
    )
    parser.add_argument(
        "--target-success",
        type=float,
        default=0.35,
        help="Pass threshold for search success_rate.",
    )
    return parser.parse_args()


def _binom_cdf(k: int, n: int) -> float:
    if k < 0:
        return 0.0
    if k >= n:
        return 1.0
    denom = 2.0**n
    num = 0
    for i in range(k + 1):
        num += math.comb(n, i)
    return num / denom


def mcnemar_exact_pvalue(improved: int, worse: int) -> float:
    n = improved + worse
    if n == 0:
        return 1.0
    lo = min(improved, worse)
    hi = max(improved, worse)
    left_tail = _binom_cdf(lo, n)
    right_tail = 1.0 - _binom_cdf(hi - 1, n)
    p = 2.0 * min(left_tail, right_tail)
    return min(1.0, p)


def load_run(run_dir: Path, target_success: float) -> dict[str, Any]:
    summary_path = run_dir / "benchmark_summary.json"
    results_path = run_dir / "benchmark_results.json"
    if not summary_path.exists() or not results_path.exists():
        raise FileNotFoundError(f"Missing benchmark files in {run_dir}")

    summary_rows = json.loads(summary_path.read_text(encoding="utf-8"))
    results_rows = json.loads(results_path.read_text(encoding="utf-8"))

    by_mode = {row["mode"]: row for row in summary_rows}
    required = {"greedy", "retry", "search"}
    if not required.issubset(by_mode):
        missing = sorted(required - set(by_mode))
        raise ValueError(f"{run_dir} missing modes in summary: {missing}")

    per_task: dict[str, dict[str, bool]] = {}
    for row in results_rows:
        task_id = str(row.get("task_id", ""))
        mode = str(row.get("mode", ""))
        success = bool(row.get("success", False))
        if not task_id or mode not in required:
            continue
        per_task.setdefault(task_id, {})[mode] = success

    improved = 0
    worse = 0
    same = 0
    for task_id, state in per_task.items():
        if not required.issubset(state):
            continue
        greedy_ok = state["greedy"]
        search_ok = state["search"]
        if search_ok and not greedy_ok:
            improved += 1
        elif greedy_ok and not search_ok:
            worse += 1
        else:
            same += 1

    return {
        "run_dir": str(run_dir),
        "num_tasks": int(by_mode["search"]["num_tasks"]),
        "greedy_success": float(by_mode["greedy"]["success_rate"]),
        "retry_success": float(by_mode["retry"]["success_rate"]),
        "search_success": float(by_mode["search"]["success_rate"]),
        "target_success": target_success,
        "search_pass_target": bool(float(by_mode["search"]["success_rate"]) >= target_success),
        "paired_search_vs_greedy": {
            "n": improved + worse + same,
            "improved": improved,
            "worse": worse,
            "same": same,
            "p_value": mcnemar_exact_pvalue(improved, worse),
        },
    }


def build_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# P1-3 Third-Party Transfer Summary",
        "",
        f"- Target line: search success >= {report['target_success']:.2f}",
        "",
        "| Run | Tasks | Greedy | Retry | Search | Target Pass | Search vs Greedy McNemar p |",
        "| --- | ---: | ---: | ---: | ---: | --- | ---: |",
    ]
    for row in report["runs"]:
        lines.append(
            "| {run} | {tasks} | {g:.3f} | {r:.3f} | {s:.3f} | {ok} | {p:.6f} |".format(
                run=Path(row["run_dir"]).name,
                tasks=row["num_tasks"],
                g=row["greedy_success"],
                r=row["retry_success"],
                s=row["search_success"],
                ok="PASS" if row["search_pass_target"] else "FAIL",
                p=row["paired_search_vs_greedy"]["p_value"],
            )
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    runs = [load_run(run_dir, args.target_success) for run_dir in args.runs]
    overall_pass = all(row["search_pass_target"] for row in runs)

    report = {
        "target_success": args.target_success,
        "overall_pass": overall_pass,
        "runs": runs,
    }

    out_json = args.output_prefix.with_suffix(".json")
    out_md = args.output_prefix.with_suffix(".md")
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    out_md.write_text(build_markdown(report), encoding="utf-8")

    print(f"[OK] Wrote {out_json}")
    print(f"[OK] Wrote {out_md}")
    print(f"[CHECK] overall_pass={overall_pass}")


if __name__ == "__main__":
    main()

