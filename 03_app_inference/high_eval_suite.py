from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any

from critical_eval_baselines import (
    evaluate_symbolic_app_no_prior,
    load_json,
    load_jsonl,
    mcnemar_exact_pvalue,
)
from graph_engine import build_default_graph_engine
from oracle_planner import OraclePlanner
from symbolic_validator import SymbolicValidator


DATA_DIR = Path(__file__).resolve().parent
PAPER_DIR = DATA_DIR / "paper_artifacts"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build High-level reviewer-fix artifacts.")
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--out-dir", type=Path, default=PAPER_DIR)
    return parser.parse_args()


def paired_compare(left_map: dict[str, bool], right_map: dict[str, bool], left: str, right: str) -> dict[str, Any]:
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
    return {
        "left": left,
        "right": right,
        "n": len(keys),
        "improved": improved,
        "worse": worse,
        "same": same,
        "p_value": mcnemar_exact_pvalue(improved, worse),
    }


def _load_graph_mode_rows(data_dir: Path, guidance: str, mode: str) -> list[dict[str, Any]]:
    path = data_dir / f"graph_benchmark_clean_google_gemini_2_5_flash_20260326_182800_{guidance}" / "benchmark_results.jsonl"
    rows = load_jsonl(path)
    return [row for row in rows if str(row.get("mode")) == mode]


def _robot_entropy(robot_counts: dict[str, int]) -> float:
    total = sum(robot_counts.values())
    if total <= 0:
        return 0.0
    ent = 0.0
    for c in robot_counts.values():
        if c <= 0:
            continue
        p = c / total
        ent -= p * math.log(p)
    return ent


def build_component_decoupling(data_dir: Path) -> dict[str, Any]:
    report = load_json(data_dir / "graph_benchmark_clean_google_gemini_2_5_flash_20260326_182800_ablation_report.json")
    agg = report["aggregate"]

    settings = [
        {
            "name": "none/greedy",
            "guidance": False,
            "verifier_feedback": False,
            "search": False,
            "success_rate": agg["none"]["greedy"]["success_rate"],
            "planner_valid_rate": agg["none"]["greedy"]["planner_valid_rate"],
            "selected_valid_rate": agg["none"]["greedy"]["selected_valid_rate"],
            "search_selected_rate": agg["none"]["greedy"]["search_selected_rate"],
        },
        {
            "name": "none/retry",
            "guidance": False,
            "verifier_feedback": True,
            "search": False,
            "success_rate": agg["none"]["retry"]["success_rate"],
            "planner_valid_rate": agg["none"]["retry"]["planner_valid_rate"],
            "selected_valid_rate": agg["none"]["retry"]["selected_valid_rate"],
            "search_selected_rate": agg["none"]["retry"]["search_selected_rate"],
        },
        {
            "name": "none/search",
            "guidance": False,
            "verifier_feedback": True,
            "search": True,
            "success_rate": agg["none"]["search"]["success_rate"],
            "planner_valid_rate": agg["none"]["search"]["planner_valid_rate"],
            "selected_valid_rate": agg["none"]["search"]["selected_valid_rate"],
            "search_selected_rate": agg["none"]["search"]["search_selected_rate"],
        },
        {
            "name": "route_hint/greedy",
            "guidance": True,
            "verifier_feedback": False,
            "search": False,
            "success_rate": agg["route_hint"]["greedy"]["success_rate"],
            "planner_valid_rate": agg["route_hint"]["greedy"]["planner_valid_rate"],
            "selected_valid_rate": agg["route_hint"]["greedy"]["selected_valid_rate"],
            "search_selected_rate": agg["route_hint"]["greedy"]["search_selected_rate"],
        },
        {
            "name": "route_hint/retry",
            "guidance": True,
            "verifier_feedback": True,
            "search": False,
            "success_rate": agg["route_hint"]["retry"]["success_rate"],
            "planner_valid_rate": agg["route_hint"]["retry"]["planner_valid_rate"],
            "selected_valid_rate": agg["route_hint"]["retry"]["selected_valid_rate"],
            "search_selected_rate": agg["route_hint"]["retry"]["search_selected_rate"],
        },
        {
            "name": "route_hint/search",
            "guidance": True,
            "verifier_feedback": True,
            "search": True,
            "success_rate": agg["route_hint"]["search"]["success_rate"],
            "planner_valid_rate": agg["route_hint"]["search"]["planner_valid_rate"],
            "selected_valid_rate": agg["route_hint"]["search"]["selected_valid_rate"],
            "search_selected_rate": agg["route_hint"]["search"]["search_selected_rate"],
        },
    ]

    none_greedy_rows = _load_graph_mode_rows(data_dir, guidance="none", mode="greedy")
    none_retry_rows = _load_graph_mode_rows(data_dir, guidance="none", mode="retry")
    none_search_rows = _load_graph_mode_rows(data_dir, guidance="none", mode="search")
    route_greedy_rows = _load_graph_mode_rows(data_dir, guidance="route_hint", mode="greedy")
    route_search_rows = _load_graph_mode_rows(data_dir, guidance="route_hint", mode="search")

    maps = {
        "none/greedy": {str(r["task_id"]): bool(r.get("success")) for r in none_greedy_rows},
        "none/retry": {str(r["task_id"]): bool(r.get("success")) for r in none_retry_rows},
        "none/search": {str(r["task_id"]): bool(r.get("success")) for r in none_search_rows},
        "route_hint/greedy": {str(r["task_id"]): bool(r.get("success")) for r in route_greedy_rows},
        "route_hint/search": {str(r["task_id"]): bool(r.get("success")) for r in route_search_rows},
    }

    pairwise = [
        paired_compare(maps["none/greedy"], maps["route_hint/greedy"], "none/greedy", "route_hint/greedy"),
        paired_compare(maps["none/greedy"], maps["none/retry"], "none/greedy", "none/retry"),
        paired_compare(maps["none/greedy"], maps["none/search"], "none/greedy", "none/search"),
        paired_compare(maps["none/search"], maps["route_hint/search"], "none/search", "route_hint/search"),
        paired_compare(maps["route_hint/greedy"], maps["route_hint/search"], "route_hint/greedy", "route_hint/search"),
    ]

    return {"settings": settings, "pairwise": pairwise}


def build_mcts_sensitivity(data_dir: Path) -> dict[str, Any]:
    task_file_8 = data_dir / "app_benchmark_tasks.json"
    task_file_40 = data_dir / "app_benchmark_tasks_random_40.json"

    fixed_depth = 18
    iter_grid = [8, 12, 16, 24, 32, 48, 96, 160]
    iter_rows: list[dict[str, Any]] = []
    for it in iter_grid:
        r8 = evaluate_symbolic_app_no_prior(task_file_8, iterations=it, depth=fixed_depth)
        r40 = evaluate_symbolic_app_no_prior(task_file_40, iterations=it, depth=fixed_depth)
        iter_rows.append(
            {
                "iterations": it,
                "depth": fixed_depth,
                "app8_success": r8["success_rate"],
                "app8_runtime": r8["avg_runtime_sec"],
                "app40_success": r40["success_rate"],
                "app40_runtime": r40["avg_runtime_sec"],
                "app40_steps": r40["avg_steps"],
                "task_success_app40": r40["task_success"],
            }
        )

    fixed_iter = 96
    depth_grid = [4, 6, 8, 10, 12, 18, 24]
    depth_rows: list[dict[str, Any]] = []
    for depth in depth_grid:
        r8 = evaluate_symbolic_app_no_prior(task_file_8, iterations=fixed_iter, depth=depth)
        r40 = evaluate_symbolic_app_no_prior(task_file_40, iterations=fixed_iter, depth=depth)
        depth_rows.append(
            {
                "iterations": fixed_iter,
                "depth": depth,
                "app8_success": r8["success_rate"],
                "app8_runtime": r8["avg_runtime_sec"],
                "app40_success": r40["success_rate"],
                "app40_runtime": r40["avg_runtime_sec"],
                "app40_steps": r40["avg_steps"],
                "task_success_app40": r40["task_success"],
            }
        )

    baseline_map = next(row["task_success_app40"] for row in iter_rows if row["iterations"] == 96)
    pairwise_vs_baseline: list[dict[str, Any]] = []
    for row in iter_rows:
        if row["iterations"] == 96:
            continue
        pairwise_vs_baseline.append(
            paired_compare(
                row["task_success_app40"],
                baseline_map,
                left=f"iter={row['iterations']},depth=18",
                right="iter=96,depth=18",
            )
        )
    for row in depth_rows:
        if row["depth"] == 18:
            continue
        pairwise_vs_baseline.append(
            paired_compare(
                row["task_success_app40"],
                baseline_map,
                left=f"iter=96,depth={row['depth']}",
                right="iter=96,depth=18",
            )
        )

    for row in iter_rows:
        row.pop("task_success_app40", None)
    for row in depth_rows:
        row.pop("task_success_app40", None)

    return {
        "fixed_depth18_iteration_sweep": iter_rows,
        "fixed_iter96_depth_sweep": depth_rows,
        "pairwise_vs_baseline_96x18": pairwise_vs_baseline,
    }


def build_oracle_sensitivity(data_dir: Path) -> dict[str, Any]:
    engine = build_default_graph_engine(data_dir)
    validator = SymbolicValidator(engine)
    tasks = load_jsonl(data_dir / "graph_tasks_30.jsonl")

    def eval_setting(cost_ratio: float, cost_gap: float) -> dict[str, Any]:
        planner = OraclePlanner(
            engine=engine,
            validator=validator,
            enable_diverse_selection=True,
            max_cost_ratio=cost_ratio,
            max_cost_gap=cost_gap,
        )
        success = 0
        costs: list[float] = []
        steps: list[int] = []
        robots: Counter[str] = Counter()
        for task in tasks:
            task_id = str(task.get("task_id") or task.get("id"))
            _ = task_id
            start = str(task.get("start") or task.get("start_node"))
            end = str(task.get("end") or task.get("end_node"))
            item = str(task.get("item") or task.get("target_item"))
            result = planner.plan_task(start_node=start, end_node=end, target_item=item)
            if result is None or not bool(result.feedback.get("is_valid")):
                continue
            success += 1
            costs.append(float(result.feedback.get("trajectory_cost") or 0.0))
            steps.append(len(result.plan))
            robots[result.robot] += 1

        return {
            "max_cost_ratio": cost_ratio,
            "max_cost_gap": cost_gap,
            "success_rate": success / max(len(tasks), 1),
            "avg_cost": sum(costs) / max(len(costs), 1),
            "avg_steps": sum(steps) / max(len(steps), 1),
            "robot_counts": dict(robots),
            "robot_entropy": _robot_entropy(dict(robots)),
        }

    ratio_sweep = [eval_setting(cost_ratio=x, cost_gap=35.0) for x in [1.1, 1.2, 1.35, 1.5, 1.8]]
    gap_sweep = [eval_setting(cost_ratio=1.35, cost_gap=x) for x in [20.0, 35.0, 50.0, 80.0]]
    return {"ratio_sweep_gap35": ratio_sweep, "gap_sweep_ratio135": gap_sweep}


def _parse_len_norm_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    text = path.read_text(encoding="utf-8")
    rows: list[dict[str, Any]] = []
    for line in text.splitlines():
        line = line.strip()
        if not line.startswith("|"):
            continue
        if "Method" in line or "---" in line:
            continue
        parts = [x.strip() for x in line.strip("|").split("|")]
        if len(parts) < 4:
            continue
        try:
            rows.append(
                {
                    "method": parts[0],
                    "balanced_executable_rate": float(parts[1]),
                    "exact_delivery_rate": float(parts[2]),
                    "avg_penalty": float(parts[3]),
                }
            )
        except ValueError:
            continue
    return rows


def _severity_bucket(severity_score: float, cuts: tuple[float, float, float]) -> float:
    c1, c2, c3 = cuts
    if severity_score <= 0:
        return 0.0
    if severity_score < c1:
        return 0.15
    if severity_score < c2:
        return 0.35
    if severity_score < c3:
        return 0.7
    return 1.1


def _bucket_entropy(counter: dict[str, int]) -> float:
    total = sum(counter.values())
    if total == 0:
        return 0.0
    ent = 0.0
    for c in counter.values():
        if c <= 0:
            continue
        p = c / total
        ent -= p * math.log(p)
    return ent


def build_mdpo_sensitivity(data_dir: Path) -> dict[str, Any]:
    pvr_eval = data_dir / "llama-factory" / "pvr_eval"
    v2_reports = {
        "SFT v2": pvr_eval / "sft_v2_balanced_report.json",
        "Vanilla DPO v2": pvr_eval / "dpo_vanilla_v2_balanced_report.json",
        "Margin DPO v2 (alpha=1.0, lambda=0.0, beta=0.1)": pvr_eval / "dpo_margin_v2_balanced_report.json",
    }
    v2_rows: list[dict[str, Any]] = []
    for name, path in v2_reports.items():
        row = load_json(path)
        overall = row["overall"]
        v2_rows.append(
            {
                "method": name,
                "balanced_executable_rate": float(overall["executable_plan_rate"]),
                "exact_delivery_rate": float(overall["exact_delivery_rate"]),
                "avg_penalty": float(overall["avg_penalty_score"]),
                "count": int(overall["count"]),
            }
        )

    len_norm_rows = _parse_len_norm_rows(data_dir / "RESULTS_SUMMARY_LEN_NORM.md")

    train_rows = load_jsonl(data_dir / "dpo_margin_dataset_train.jsonl")
    severity_scores = [float(row.get("preference_strength", {}).get("severity_score", 0.0) or 0.0) for row in train_rows]
    threshold_sets = {
        "aggressive_4_8_12": (4.0, 8.0, 12.0),
        "default_5_10_15": (5.0, 10.0, 15.0),
        "conservative_6_12_18": (6.0, 12.0, 18.0),
    }

    threshold_rows: list[dict[str, Any]] = []
    for name, cuts in threshold_sets.items():
        margins = [_severity_bucket(score, cuts) for score in severity_scores]
        bucket_counter = Counter(str(m) for m in margins if m > 0.0)
        threshold_rows.append(
            {
                "setting": name,
                "cuts": list(cuts),
                "mean_margin": sum(margins) / max(len(margins), 1),
                "bucket_counts": dict(bucket_counter),
                "bucket_entropy": _bucket_entropy(dict(bucket_counter)),
            }
        )

    beta_grid = [0.05, 0.1, 0.2]
    lambda_grid = [0.0, 0.5, 1.0]
    m_max = 1.1
    bound_rows: list[dict[str, Any]] = []
    for beta in beta_grid:
        for lam in lambda_grid:
            grad_bound = beta * (1.0 + lam * m_max)
            curvature_bound = (beta * beta / 4.0) * (1.0 + lam * m_max)
            bound_rows.append(
                {
                    "beta": beta,
                    "lambda": lam,
                    "max_grad_bound": grad_bound,
                    "max_curvature_bound": curvature_bound,
                }
            )

    return {
        "v2_exec_rows": v2_rows,
        "len_norm_alpha_rows": len_norm_rows,
        "margin_threshold_rows": threshold_rows,
        "beta_lambda_bound_rows": bound_rows,
    }


def build_markdown(report: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("# High-Level Fix Report")
    lines.append("")
    lines.append("## 1) Component Decoupling (High-1)")
    lines.append("")
    lines.append("| Setting | Guidance | Verifier | Search | Success | Planner Valid | Selected Valid | Search Selected |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for row in report["component_decoupling"]["settings"]:
        lines.append(
            "| {name} | {g} | {v} | {s} | {succ:.3f} | {pv:.3f} | {sv:.3f} | {ss:.3f} |".format(
                name=row["name"],
                g="1" if row["guidance"] else "0",
                v="1" if row["verifier_feedback"] else "0",
                s="1" if row["search"] else "0",
                succ=row["success_rate"],
                pv=row["planner_valid_rate"],
                sv=row["selected_valid_rate"],
                ss=row["search_selected_rate"],
            )
        )
    lines.append("")
    lines.append("| Right vs Left | n | Improved | Worse | p-value |")
    lines.append("| --- | ---: | ---: | ---: | ---: |")
    for row in report["component_decoupling"]["pairwise"]:
        lines.append(
            "| {right} vs {left} | {n} | {improved} | {worse} | {p:.6f} |".format(
                right=row["right"],
                left=row["left"],
                n=row["n"],
                improved=row["improved"],
                worse=row["worse"],
                p=row["p_value"],
            )
        )

    lines.append("")
    lines.append("## 2) MCTS Sensitivity (High-2/3)")
    lines.append("")
    lines.append("### Iteration Sweep (depth=18)")
    lines.append("| Iter | App-8 Success | App-8 Runtime(s) | App-40 Success | App-40 Runtime(s) |")
    lines.append("| ---: | ---: | ---: | ---: | ---: |")
    for row in report["mcts_sensitivity"]["fixed_depth18_iteration_sweep"]:
        lines.append(
            "| {it} | {s8:.3f} | {t8:.4f} | {s40:.3f} | {t40:.4f} |".format(
                it=row["iterations"],
                s8=row["app8_success"],
                t8=row["app8_runtime"],
                s40=row["app40_success"],
                t40=row["app40_runtime"],
            )
        )
    lines.append("")
    lines.append("### Depth Sweep (iter=96)")
    lines.append("| Depth | App-8 Success | App-8 Runtime(s) | App-40 Success | App-40 Runtime(s) |")
    lines.append("| ---: | ---: | ---: | ---: | ---: |")
    for row in report["mcts_sensitivity"]["fixed_iter96_depth_sweep"]:
        lines.append(
            "| {d} | {s8:.3f} | {t8:.4f} | {s40:.3f} | {t40:.4f} |".format(
                d=row["depth"],
                s8=row["app8_success"],
                t8=row["app8_runtime"],
                s40=row["app40_success"],
                t40=row["app40_runtime"],
            )
        )

    lines.append("")
    lines.append("## 3) Oracle Threshold Sensitivity (High-3)")
    lines.append("")
    lines.append("| max_cost_ratio | max_cost_gap | Success | Avg Cost | Avg Steps | Robot Entropy |")
    lines.append("| ---: | ---: | ---: | ---: | ---: | ---: |")
    for row in report["oracle_sensitivity"]["ratio_sweep_gap35"]:
        lines.append(
            "| {r:.2f} | {g:.1f} | {s:.3f} | {c:.3f} | {st:.3f} | {e:.4f} |".format(
                r=row["max_cost_ratio"],
                g=row["max_cost_gap"],
                s=row["success_rate"],
                c=row["avg_cost"],
                st=row["avg_steps"],
                e=row["robot_entropy"],
            )
        )

    lines.append("")
    lines.append("## 4) mDPO Sensitivity (High-3)")
    lines.append("")
    lines.append("### v2 Executable Metrics")
    lines.append("| Method | Balanced Exec | Exact Delivery | Avg Penalty |")
    lines.append("| --- | ---: | ---: | ---: |")
    for row in report["mdpo_sensitivity"]["v2_exec_rows"]:
        lines.append(
            "| {m} | {e:.3f} | {d:.3f} | {p:.2f} |".format(
                m=row["method"],
                e=row["balanced_executable_rate"],
                d=row["exact_delivery_rate"],
                p=row["avg_penalty"],
            )
        )
    if report["mdpo_sensitivity"]["len_norm_alpha_rows"]:
        lines.append("")
        lines.append("### token-mean / alpha-scale rows (locked summary)")
        lines.append("| Method | Balanced Exec | Exact Delivery | Avg Penalty |")
        lines.append("| --- | ---: | ---: | ---: |")
        for row in report["mdpo_sensitivity"]["len_norm_alpha_rows"]:
            lines.append(
                "| {m} | {e:.3f} | {d:.3f} | {p:.2f} |".format(
                    m=row["method"],
                    e=row["balanced_executable_rate"],
                    d=row["exact_delivery_rate"],
                    p=row["avg_penalty"],
                )
            )

    lines.append("")
    lines.append("### margin-bucket threshold sensitivity (train set)")
    lines.append("| Setting | Cuts | Mean Margin | Bucket Entropy |")
    lines.append("| --- | --- | ---: | ---: |")
    for row in report["mdpo_sensitivity"]["margin_threshold_rows"]:
        lines.append(
            "| {s} | {c} | {m:.4f} | {e:.4f} |".format(
                s=row["setting"],
                c=row["cuts"],
                m=row["mean_margin"],
                e=row["bucket_entropy"],
            )
        )

    lines.append("")
    lines.append("### beta/lambda analytic bounds")
    lines.append("| beta | lambda | max |dL/dDelta| | max d2L/dDelta2 |")
    lines.append("| ---: | ---: | ---: | ---: |")
    for row in report["mdpo_sensitivity"]["beta_lambda_bound_rows"]:
        lines.append(
            "| {b:.2f} | {l:.1f} | {g:.4f} | {h:.5f} |".format(
                b=row["beta"],
                l=row["lambda"],
                g=row["max_grad_bound"],
                h=row["max_curvature_bound"],
            )
        )
    return "\n".join(lines) + "\n"


def build_latex(report: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("\\begin{table}[t]")
    lines.append("\\centering")
    lines.append("\\caption{High-1 component-decoupling significance (graph-30).}")
    lines.append("\\label{tab:high-component-significance}")
    lines.append("\\small")
    lines.append("\\begin{tabular}{lcccc}")
    lines.append("\\toprule")
    lines.append("Right vs Left & $n$ & Improved & Worse & $p$ \\\\")
    lines.append("\\midrule")
    for row in report["component_decoupling"]["pairwise"]:
        lines.append(
            "{r} vs {l} & {n} & {i} & {w} & {p:.6f} \\\\".format(
                r=row["right"].replace("_", "\\_"),
                l=row["left"].replace("_", "\\_"),
                n=row["n"],
                i=row["improved"],
                w=row["worse"],
                p=row["p_value"],
            )
        )
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")
    lines.append("")

    lines.append("\\begin{table}[t]")
    lines.append("\\centering")
    lines.append("\\caption{MCTS sensitivity on symbolic app tasks (High-2/3).}")
    lines.append("\\label{tab:high-mcts-sensitivity}")
    lines.append("\\small")
    lines.append("\\begin{tabular}{ccccc}")
    lines.append("\\toprule")
    lines.append("Iter & Depth & App-8 success & App-40 success & App-40 runtime (s) \\\\")
    lines.append("\\midrule")
    for row in report["mcts_sensitivity"]["fixed_depth18_iteration_sweep"]:
        lines.append(
            "{it} & {d} & {s8:.3f} & {s40:.3f} & {t40:.4f} \\\\".format(
                it=row["iterations"],
                d=row["depth"],
                s8=row["app8_success"],
                s40=row["app40_success"],
                t40=row["app40_runtime"],
            )
        )
    lines.append("\\midrule")
    for row in report["mcts_sensitivity"]["fixed_iter96_depth_sweep"]:
        lines.append(
            "{it} & {d} & {s8:.3f} & {s40:.3f} & {t40:.4f} \\\\".format(
                it=row["iterations"],
                d=row["depth"],
                s8=row["app8_success"],
                s40=row["app40_success"],
                t40=row["app40_runtime"],
            )
        )
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")
    lines.append("")

    lines.append("\\begin{table}[t]")
    lines.append("\\centering")
    lines.append("\\caption{Oracle near-optimal threshold sensitivity (graph-30).}")
    lines.append("\\label{tab:high-oracle-sensitivity}")
    lines.append("\\small")
    lines.append("\\begin{tabular}{ccccc}")
    lines.append("\\toprule")
    lines.append("Ratio & Gap & Success & Avg cost & Robot entropy \\\\")
    lines.append("\\midrule")
    for row in report["oracle_sensitivity"]["ratio_sweep_gap35"]:
        lines.append(
            "{r:.2f} & {g:.1f} & {s:.3f} & {c:.3f} & {e:.4f} \\\\".format(
                r=row["max_cost_ratio"],
                g=row["max_cost_gap"],
                s=row["success_rate"],
                c=row["avg_cost"],
                e=row["robot_entropy"],
            )
        )
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")
    lines.append("")

    lines.append("\\begin{table}[t]")
    lines.append("\\centering")
    lines.append("\\caption{mDPO hyperparameter sensitivity diagnostics (High-3).}")
    lines.append("\\label{tab:high-mdpo-sensitivity}")
    lines.append("\\small")
    lines.append("\\begin{tabular}{p{5.0cm}ccc}")
    lines.append("\\toprule")
    lines.append("Method & Balanced exec & Exact delivery & Avg penalty \\\\")
    lines.append("\\midrule")
    for row in report["mdpo_sensitivity"]["v2_exec_rows"]:
        lines.append(
            "{m} & {e:.3f} & {d:.3f} & {p:.2f} \\\\".format(
                m=row["method"].replace("_", "\\_"),
                e=row["balanced_executable_rate"],
                d=row["exact_delivery_rate"],
                p=row["avg_penalty"],
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

    report = {
        "component_decoupling": build_component_decoupling(data_dir),
        "mcts_sensitivity": build_mcts_sensitivity(data_dir),
        "oracle_sensitivity": build_oracle_sensitivity(data_dir),
        "mdpo_sensitivity": build_mdpo_sensitivity(data_dir),
    }

    out_json = out_dir / "high_fix_report.json"
    out_md = out_dir / "high_fix_report.md"
    out_tex = out_dir / "high_fix_tables.tex"

    out_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    out_md.write_text(build_markdown(report), encoding="utf-8")
    out_tex.write_text(build_latex(report), encoding="utf-8")

    print(f"[OK] Wrote {out_json}")
    print(f"[OK] Wrote {out_md}")
    print(f"[OK] Wrote {out_tex}")


if __name__ == "__main__":
    main()
