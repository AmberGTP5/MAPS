from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DATA_DIR = Path(__file__).resolve().parent
PAPER_DIR = DATA_DIR / "paper_artifacts"

from app.search.mcts import PlanSearchEngine
from app.search.verifier import PlanSearchVerifier, SearchConstraints
from graph_engine import build_default_graph_engine
from oracle_planner import OraclePlanner
from symbolic_validator import SymbolicValidator


Z95 = 1.959963984540054


@dataclass
class PairwiseResult:
    left: str
    right: str
    n: int
    improved: int
    worse: int
    same: int
    p_value: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build critical head-to-head baseline artifacts.")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DATA_DIR,
        help="Data directory containing locked benchmark artifacts.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=PAPER_DIR,
        help="Where critical report files will be written.",
    )
    parser.add_argument(
        "--search-iterations",
        type=int,
        default=96,
        help="Search iterations for symbolic no-prior app baseline.",
    )
    parser.add_argument(
        "--search-depth",
        type=int,
        default=18,
        help="Search rollout depth for symbolic no-prior app baseline.",
    )
    return parser.parse_args()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    return rows


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


def paired_compare(left_map: dict[str, bool], right_map: dict[str, bool], left: str, right: str) -> PairwiseResult:
    keys = sorted(set(left_map) & set(right_map))
    improved = 0
    worse = 0
    same = 0
    for key in keys:
        l_ok = bool(left_map[key])
        r_ok = bool(right_map[key])
        if r_ok and not l_ok:
            improved += 1
        elif l_ok and not r_ok:
            worse += 1
        else:
            same += 1
    return PairwiseResult(
        left=left,
        right=right,
        n=len(keys),
        improved=improved,
        worse=worse,
        same=same,
        p_value=mcnemar_exact_pvalue(improved, worse),
    )


def wilson_ci(successes: int, total: int) -> tuple[float, float]:
    if total <= 0:
        return 0.0, 0.0
    p = successes / total
    zz = Z95 * Z95
    denom = 1.0 + zz / total
    center = (p + zz / (2.0 * total)) / denom
    half = Z95 * math.sqrt((p * (1.0 - p) + zz / (4.0 * total)) / total) / denom
    return max(0.0, center - half), min(1.0, center + half)


def infer_constraints(task: dict[str, Any]) -> SearchConstraints:
    expected_source = task.get("expected_source")
    expected_handover = task.get("expected_handover")

    if isinstance(expected_source, str) and expected_source.strip():
        sources = (expected_source.strip(),)
    else:
        sources = ("Starbucks_Outdoor", "Vending_Machine_2F")

    if isinstance(expected_handover, str) and expected_handover.strip():
        handovers = (expected_handover.strip(),)
    else:
        handovers = ("Info_Hub_4F", "Elevator_Lobby_4F")

    return SearchConstraints(allowed_sources=sources, allowed_handover_zones=handovers)


def evaluate_symbolic_app_no_prior(
    tasks_file: Path,
    *,
    iterations: int,
    depth: int,
) -> dict[str, Any]:
    tasks = load_json(tasks_file)
    if not isinstance(tasks, list):
        raise ValueError(f"App task file must be a JSON list: {tasks_file}")

    success_map: dict[str, bool] = {}
    runtimes: list[float] = []
    step_counts: list[int] = []

    for idx, task in enumerate(tasks):
        task_id = str(task.get("id", f"task_{idx}"))
        verifier = PlanSearchVerifier(constraints=infer_constraints(task))
        engine = PlanSearchEngine(
            verifier=verifier,
            max_iterations=iterations,
            rollout_depth=depth,
            exploration_weight=1.25,
            seed=42 + idx,
        )
        t0 = time.perf_counter()
        outcome = engine.search(prior_plan=[])
        runtime = time.perf_counter() - t0
        is_valid, _, _ = verifier.plan_is_valid(outcome.plan)
        success_map[task_id] = bool(is_valid)
        runtimes.append(runtime)
        step_counts.append(len(outcome.plan))

    n = len(success_map)
    s = sum(1 for ok in success_map.values() if ok)
    ci_low, ci_high = wilson_ci(s, n)
    return {
        "tasks_file": str(tasks_file),
        "num_tasks": n,
        "success_count": s,
        "success_rate": s / max(n, 1),
        "success_ci_low": ci_low,
        "success_ci_high": ci_high,
        "avg_runtime_sec": statistics.mean(runtimes) if runtimes else 0.0,
        "avg_steps": statistics.mean(step_counts) if step_counts else 0.0,
        "task_success": success_map,
    }


def _load_graph_mode_rows(data_dir: Path, guidance: str, mode: str) -> list[dict[str, Any]]:
    path = data_dir / f"graph_benchmark_clean_google_gemini_2_5_flash_20260326_182800_{guidance}" / "benchmark_results.jsonl"
    rows = load_jsonl(path)
    return [row for row in rows if str(row.get("mode")) == mode]


def evaluate_symbolic_graph_oracle(tasks_file: Path, data_dir: Path) -> dict[str, Any]:
    engine = build_default_graph_engine(data_dir)
    validator = SymbolicValidator(engine)
    oracle = OraclePlanner(engine=engine, validator=validator, enable_diverse_selection=True)

    tasks = load_jsonl(tasks_file)
    success_map: dict[str, bool] = {}
    runtimes: list[float] = []
    costs: list[float] = []
    steps: list[int] = []
    robot_counts: dict[str, int] = {}

    for task in tasks:
        task_id = str(task.get("task_id") or task.get("id"))
        start = str(task.get("start") or task.get("start_node"))
        end = str(task.get("end") or task.get("end_node"))
        item = str(task.get("item") or task.get("target_item"))
        t0 = time.perf_counter()
        result = oracle.plan_task(start_node=start, end_node=end, target_item=item)
        runtimes.append(time.perf_counter() - t0)
        if result is None:
            success_map[task_id] = False
            continue

        is_valid = bool(result.feedback.get("is_valid"))
        success_map[task_id] = is_valid
        if is_valid:
            costs.append(float(result.feedback.get("trajectory_cost") or 0.0))
            steps.append(len(result.plan))
            robot_counts[result.robot] = robot_counts.get(result.robot, 0) + 1

    n = len(success_map)
    s = sum(1 for ok in success_map.values() if ok)
    ci_low, ci_high = wilson_ci(s, n)
    return {
        "tasks_file": str(tasks_file),
        "num_tasks": n,
        "success_count": s,
        "success_rate": s / max(n, 1),
        "success_ci_low": ci_low,
        "success_ci_high": ci_high,
        "avg_runtime_sec": statistics.mean(runtimes) if runtimes else 0.0,
        "avg_cost": statistics.mean(costs) if costs else 0.0,
        "avg_steps": statistics.mean(steps) if steps else 0.0,
        "robot_counts": robot_counts,
        "task_success": success_map,
    }


def build_head_to_head_report(data_dir: Path, graph_oracle: dict[str, Any], app_symbolic8: dict[str, Any], app_symbolic40: dict[str, Any]) -> dict[str, Any]:
    graph_ablation = load_json(data_dir / "graph_benchmark_clean_google_gemini_2_5_flash_20260326_182800_ablation_report.json")
    app_gemini = load_json(data_dir / "app_benchmark_clean_google_gemini_2_5_flash_20260327_195823" / "benchmark_summary.json")
    app_qwen = load_json(data_dir / "app_benchmark_clean_qwen_qwen_2_5_7b_instruct_20260327_201401" / "benchmark_summary.json")
    app_random40 = load_json(data_dir / "app_benchmark_results_random40" / "benchmark_summary.json")

    graph_none_greedy = graph_ablation["aggregate"]["none"]["greedy"]
    graph_none_retry = graph_ablation["aggregate"]["none"]["retry"]
    graph_none_search = graph_ablation["aggregate"]["none"]["search"]
    graph_route_search = graph_ablation["aggregate"]["route_hint"]["search"]

    def _find_mode(rows: list[dict[str, Any]], mode: str) -> dict[str, Any]:
        for row in rows:
            if str(row.get("mode")) == mode:
                return row
        raise KeyError(f"Mode '{mode}' not found in summary rows.")

    gemini_g = _find_mode(app_gemini, "greedy")
    gemini_r = _find_mode(app_gemini, "retry")
    gemini_s = _find_mode(app_gemini, "search")

    qwen_g = _find_mode(app_qwen, "greedy")
    qwen_r = _find_mode(app_qwen, "retry")
    qwen_s = _find_mode(app_qwen, "search")

    rand_g = _find_mode(app_random40, "greedy")
    rand_r = _find_mode(app_random40, "retry")
    rand_s = _find_mode(app_random40, "search")

    paradigms: list[dict[str, Any]] = [
        {
            "method": "ReAct-style one-shot",
            "graph_success": graph_none_greedy["success_rate"],
            "graph_search_selected": graph_none_greedy["search_selected_rate"],
            "app8_gemini_success": gemini_g["success_rate"],
            "app8_qwen_success": qwen_g["success_rate"],
            "app40_gemini_success": rand_g["success_rate"],
        },
        {
            "method": "Reflexion-style retry",
            "graph_success": graph_none_retry["success_rate"],
            "graph_search_selected": graph_none_retry["search_selected_rate"],
            "app8_gemini_success": gemini_r["success_rate"],
            "app8_qwen_success": qwen_r["success_rate"],
            "app40_gemini_success": rand_r["success_rate"],
        },
        {
            "method": "ToT-style search",
            "graph_success": graph_none_search["success_rate"],
            "graph_search_selected": graph_none_search["search_selected_rate"],
            "app8_gemini_success": gemini_s["success_rate"],
            "app8_qwen_success": qwen_s["success_rate"],
            "app40_gemini_success": rand_s["success_rate"],
        },
        {
            "method": "Classical symbolic planner",
            "graph_success": graph_oracle["success_rate"],
            "graph_search_selected": None,
            "app8_gemini_success": app_symbolic8["success_rate"],
            "app8_qwen_success": app_symbolic8["success_rate"],
            "app40_gemini_success": app_symbolic40["success_rate"],
        },
        {
            "method": "PVR full stack",
            "graph_success": graph_route_search["success_rate"],
            "graph_search_selected": graph_route_search["search_selected_rate"],
            "app8_gemini_success": gemini_s["success_rate"],
            "app8_qwen_success": qwen_s["success_rate"],
            "app40_gemini_success": rand_s["success_rate"],
        },
    ]

    graph_none_greedy_rows = _load_graph_mode_rows(data_dir, guidance="none", mode="greedy")
    graph_none_retry_rows = _load_graph_mode_rows(data_dir, guidance="none", mode="retry")
    graph_none_search_rows = _load_graph_mode_rows(data_dir, guidance="none", mode="search")
    graph_route_search_rows = _load_graph_mode_rows(data_dir, guidance="route_hint", mode="search")

    map_none_greedy = {str(row["task_id"]): bool(row.get("success")) for row in graph_none_greedy_rows}
    map_none_retry = {str(row["task_id"]): bool(row.get("success")) for row in graph_none_retry_rows}
    map_none_search = {str(row["task_id"]): bool(row.get("success")) for row in graph_none_search_rows}
    map_route_search = {str(row["task_id"]): bool(row.get("success")) for row in graph_route_search_rows}
    map_graph_oracle = dict(graph_oracle["task_success"])

    app8_rows = load_json(data_dir / "app_benchmark_clean_google_gemini_2_5_flash_20260327_195823" / "benchmark_results.json")
    app8_rows_qwen = load_json(data_dir / "app_benchmark_clean_qwen_qwen_2_5_7b_instruct_20260327_201401" / "benchmark_results.json")

    app8_greedy_map = {str(row["task_id"]): bool(row.get("success")) for row in app8_rows if str(row.get("mode")) == "greedy"}
    app8_retry_map = {str(row["task_id"]): bool(row.get("success")) for row in app8_rows if str(row.get("mode")) == "retry"}
    app8_search_map = {str(row["task_id"]): bool(row.get("success")) for row in app8_rows if str(row.get("mode")) == "search"}
    app8_qwen_greedy_map = {str(row["task_id"]): bool(row.get("success")) for row in app8_rows_qwen if str(row.get("mode")) == "greedy"}
    app8_symbolic_map = dict(app_symbolic8["task_success"])

    pairwise = [
        paired_compare(map_none_greedy, map_route_search, "graph_react", "graph_pvr"),
        paired_compare(map_none_retry, map_route_search, "graph_reflexion", "graph_pvr"),
        paired_compare(map_none_search, map_route_search, "graph_tot", "graph_pvr"),
        paired_compare(map_none_greedy, map_graph_oracle, "graph_react", "graph_symbolic"),
        paired_compare(app8_greedy_map, app8_search_map, "app8_gemini_react", "app8_gemini_search"),
        paired_compare(app8_retry_map, app8_search_map, "app8_gemini_retry", "app8_gemini_search"),
        paired_compare(app8_qwen_greedy_map, app8_symbolic_map, "app8_qwen_react", "app8_symbolic"),
    ]

    return {
        "paradigms": paradigms,
        "pairwise": [
            {
                "left": row.left,
                "right": row.right,
                "n": row.n,
                "improved": row.improved,
                "worse": row.worse,
                "same": row.same,
                "p_value": row.p_value,
            }
            for row in pairwise
        ],
        "graph_repair_intervention": {
            "tot_style_search_selected": graph_none_search["search_selected_rate"],
            "pvr_search_selected": graph_route_search["search_selected_rate"],
        },
        "symbolic_baselines": {
            "graph_oracle": graph_oracle,
            "app8_no_prior_search": app_symbolic8,
            "app40_no_prior_search": app_symbolic40,
        },
    }


def fmt_rate(x: float | None) -> str:
    if x is None:
        return "-"
    return f"{x:.3f}"


def build_markdown(report: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("# Critical Head-to-Head Baseline Report")
    lines.append("")
    lines.append("## Protocol-Matched Baseline Families")
    lines.append("")
    lines.append("| Method | Graph-30 Success | Graph Search-Selected | App-8 Gemini | App-8 Qwen | App-40 Gemini |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: |")
    for row in report["paradigms"]:
        lines.append(
            "| {method} | {graph_success} | {graph_search_selected} | {app8_gemini_success} | {app8_qwen_success} | {app40_gemini_success} |".format(
                method=row["method"],
                graph_success=fmt_rate(row["graph_success"]),
                graph_search_selected=fmt_rate(row["graph_search_selected"]),
                app8_gemini_success=fmt_rate(row["app8_gemini_success"]),
                app8_qwen_success=fmt_rate(row["app8_qwen_success"]),
                app40_gemini_success=fmt_rate(row["app40_gemini_success"]),
            )
        )

    lines.append("")
    lines.append("## Pairwise Exact McNemar Tests")
    lines.append("")
    lines.append("| Right vs Left | n | Improved | Worse | p-value |")
    lines.append("| --- | ---: | ---: | ---: | ---: |")
    for row in report["pairwise"]:
        lines.append(
            "| {right} vs {left} | {n} | {improved} | {worse} | {p_value:.6f} |".format(
                right=row["right"],
                left=row["left"],
                n=row["n"],
                improved=row["improved"],
                worse=row["worse"],
                p_value=row["p_value"],
            )
        )

    lines.append("")
    lines.append("## Symbolic Baseline Diagnostics")
    lines.append("")
    graph_oracle = report["symbolic_baselines"]["graph_oracle"]
    app8 = report["symbolic_baselines"]["app8_no_prior_search"]
    app40 = report["symbolic_baselines"]["app40_no_prior_search"]
    lines.append(
        f"- Graph Oracle: success={graph_oracle['success_rate']:.3f} ({graph_oracle['success_count']}/{graph_oracle['num_tasks']}), "
        f"avg_cost={graph_oracle['avg_cost']:.3f}, avg_steps={graph_oracle['avg_steps']:.3f}, avg_runtime={graph_oracle['avg_runtime_sec']:.4f}s."
    )
    lines.append(
        f"- App no-prior search (8): success={app8['success_rate']:.3f} ({app8['success_count']}/{app8['num_tasks']}), "
        f"avg_steps={app8['avg_steps']:.3f}, avg_runtime={app8['avg_runtime_sec']:.4f}s."
    )
    lines.append(
        f"- App no-prior search (40): success={app40['success_rate']:.3f} ({app40['success_count']}/{app40['num_tasks']}), "
        f"avg_steps={app40['avg_steps']:.3f}, avg_runtime={app40['avg_runtime_sec']:.4f}s."
    )
    return "\n".join(lines) + "\n"


def build_latex_table(report: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("\\begin{table}[t]")
    lines.append("\\centering")
    lines.append("\\caption{Protocol-matched head-to-head baseline families under locked tasks.}")
    lines.append("\\label{tab:critical-headtohead}")
    lines.append("\\small")
    lines.append("\\resizebox{\\columnwidth}{!}{%")
    lines.append("\\begin{tabular}{lccccc}")
    lines.append("\\toprule")
    lines.append("Method & Graph-30 & Graph Search-Selected & App-8 (Gemini) & App-8 (Qwen) & App-40 (Gemini) \\\\")
    lines.append("\\midrule")
    for row in report["paradigms"]:
        lines.append(
            "{method} & {graph_success} & {graph_search_selected} & {app8_gemini_success} & {app8_qwen_success} & {app40_gemini_success} \\\\".format(
                method=row["method"].replace("_", "\\_"),
                graph_success=fmt_rate(row["graph_success"]),
                graph_search_selected=fmt_rate(row["graph_search_selected"]),
                app8_gemini_success=fmt_rate(row["app8_gemini_success"]),
                app8_qwen_success=fmt_rate(row["app8_qwen_success"]),
                app40_gemini_success=fmt_rate(row["app40_gemini_success"]),
            )
        )
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("}")
    lines.append("\\end{table}")
    lines.append("")
    lines.append("\\begin{table}[t]")
    lines.append("\\centering")
    lines.append("\\caption{Critical head-to-head paired significance (exact McNemar).}")
    lines.append("\\label{tab:critical-headtohead-significance}")
    lines.append("\\small")
    lines.append("\\begin{tabular}{lcccc}")
    lines.append("\\toprule")
    lines.append("Right vs Left & $n$ & Improved & Worse & $p$-value \\\\")
    lines.append("\\midrule")
    for row in report["pairwise"]:
        lines.append(
            "{right} vs {left} & {n} & {improved} & {worse} & {p_value:.6f} \\\\".format(
                right=row["right"].replace("_", "\\_"),
                left=row["left"].replace("_", "\\_"),
                n=row["n"],
                improved=row["improved"],
                worse=row["worse"],
                p_value=row["p_value"],
            )
        )
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    data_dir = args.data_dir.resolve()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    graph_oracle = evaluate_symbolic_graph_oracle(data_dir / "graph_tasks_30.jsonl", data_dir)
    app_symbolic8 = evaluate_symbolic_app_no_prior(
        data_dir / "app_benchmark_tasks.json",
        iterations=args.search_iterations,
        depth=args.search_depth,
    )
    app_symbolic40 = evaluate_symbolic_app_no_prior(
        data_dir / "app_benchmark_tasks_random_40.json",
        iterations=args.search_iterations,
        depth=args.search_depth,
    )

    report = build_head_to_head_report(data_dir, graph_oracle, app_symbolic8, app_symbolic40)

    report_json = out_dir / "critical_baseline_report.json"
    report_md = out_dir / "critical_baseline_report.md"
    report_tex = out_dir / "critical_baseline_tables.tex"

    report_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    report_md.write_text(build_markdown(report), encoding="utf-8")
    report_tex.write_text(build_latex_table(report), encoding="utf-8")

    print(f"[OK] Wrote {report_json}")
    print(f"[OK] Wrote {report_md}")
    print(f"[OK] Wrote {report_tex}")


if __name__ == "__main__":
    main()
