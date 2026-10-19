from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Any


DATA_DIR = Path(__file__).resolve().parent
PAPER_DIR = DATA_DIR / "paper_artifacts"
Z95 = 1.959963984540054
MODES = ("greedy", "retry", "search")


def safe_div(numer: int, denom: int) -> float:
    return numer / denom if denom > 0 else 0.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build /app local-prior closure artifacts from benchmark run directories."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DATA_DIR,
        help="Project data directory.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=PAPER_DIR,
        help="Output directory for report artifacts.",
    )
    parser.add_argument(
        "--run",
        action="append",
        required=True,
        help=(
            "Run spec in the form label=relative_or_absolute_run_dir. "
            "Example: --run base=app_benchmark_local_base --run mdpo=app_benchmark_local_mdpo"
        ),
    )
    parser.add_argument(
        "--report-prefix",
        type=str,
        default="app_local_prior_closure",
        help="Output filename prefix.",
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


def wilson_ci(successes: int, total: int) -> tuple[float, float]:
    if total <= 0:
        return 0.0, 0.0
    p = successes / total
    zz = Z95 * Z95
    denom = 1.0 + zz / total
    center = (p + zz / (2.0 * total)) / denom
    half = Z95 * math.sqrt((p * (1.0 - p) + zz / (4.0 * total)) / total) / denom
    return max(0.0, center - half), min(1.0, center + half)


def parse_run_spec(spec: str, data_dir: Path) -> tuple[str, Path]:
    if "=" not in spec:
        raise ValueError(f"Invalid --run spec: {spec!r}. Expected label=path.")
    label, raw_path = spec.split("=", 1)
    label = label.strip()
    raw_path = raw_path.strip()
    if not label or not raw_path:
        raise ValueError(f"Invalid --run spec: {spec!r}.")
    path = Path(raw_path)
    if not path.is_absolute():
        path = data_dir / path
    return label, path.resolve()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def mode_rows(results: list[dict[str, Any]], mode: str) -> list[dict[str, Any]]:
    return [row for row in results if str(row.get("mode")) == mode]


def mode_success_map(results: list[dict[str, Any]], mode: str) -> dict[str, bool]:
    return {str(row["task_id"]): bool(row.get("success")) for row in mode_rows(results, mode)}


def mode_stats(results: list[dict[str, Any]], mode: str) -> dict[str, Any]:
    rows = mode_rows(results, mode)
    n = len(rows)
    success_count = sum(1 for row in rows if bool(row.get("success")))
    planner_valid_count = sum(1 for row in rows if bool(row.get("planner_valid")))
    selected_valid_count = sum(1 for row in rows if bool(row.get("selected_valid")))
    ci_low, ci_high = wilson_ci(success_count, n)

    runtimes = [float(row.get("runtime_sec") or 0.0) for row in rows]
    lengths = [int(row.get("executed_plan_length") or 0) for row in rows]
    handovers = [int(row.get("handover_count") or 0) for row in rows]
    sources = [str(row.get("selected_plan_source") or "") for row in rows]

    search_selected = sum(1 for source in sources if source.startswith("search"))
    planner_fallback = sum(1 for source in sources if source == "planner_fallback")
    planner_direct = sum(1 for source in sources if source == "planner")

    return {
        "mode": mode,
        "n": n,
        "success_count": success_count,
        "success_rate": success_count / max(n, 1),
        "success_ci_low": ci_low,
        "success_ci_high": ci_high,
        "planner_valid_rate": planner_valid_count / max(n, 1),
        "selected_valid_rate": selected_valid_count / max(n, 1),
        "avg_runtime_sec": statistics.mean(runtimes) if runtimes else 0.0,
        "avg_plan_length": statistics.mean(lengths) if lengths else 0.0,
        "avg_handover_count": statistics.mean(handovers) if handovers else 0.0,
        "search_selected_rate": search_selected / max(n, 1),
        "planner_fallback_rate": planner_fallback / max(n, 1),
        "planner_direct_rate": planner_direct / max(n, 1),
    }


def paired_delta(left_map: dict[str, bool], right_map: dict[str, bool]) -> dict[str, Any]:
    keys = sorted(set(left_map) & set(right_map))
    improved = 0
    worse = 0
    same = 0
    for key in keys:
        left_ok = bool(left_map[key])
        right_ok = bool(right_map[key])
        if right_ok and not left_ok:
            improved += 1
        elif left_ok and not right_ok:
            worse += 1
        else:
            same += 1
    return {
        "n": len(keys),
        "improved": improved,
        "worse": worse,
        "same": same,
        "p_value": mcnemar_exact_pvalue(improved, worse),
    }


def search_repair_metrics(results: list[dict[str, Any]]) -> dict[str, Any]:
    greedy = {str(row["task_id"]): row for row in mode_rows(results, "greedy")}
    retry = {str(row["task_id"]): row for row in mode_rows(results, "retry")}
    search = {str(row["task_id"]): row for row in mode_rows(results, "search")}
    keys = sorted(set(greedy) & set(search))

    greedy_fail = [task_id for task_id in keys if not bool(greedy[task_id].get("success"))]
    retry_fail = [task_id for task_id in keys if task_id in retry and not bool(retry[task_id].get("success"))]

    repaired_from_greedy = [
        task_id
        for task_id in greedy_fail
        if bool(search[task_id].get("success"))
    ]
    repaired_from_retry = [
        task_id
        for task_id in retry_fail
        if bool(search[task_id].get("success"))
    ]

    search_rows = [search[task_id] for task_id in keys]
    planner_invalid_rows = [row for row in search_rows if not bool(row.get("planner_valid"))]
    planner_invalid_repaired = sum(1 for row in planner_invalid_rows if bool(row.get("selected_valid")))
    greedy_success_count = sum(1 for task_id in keys if bool(greedy[task_id].get("success")))
    final_success_count = greedy_success_count + len(repaired_from_greedy)

    return {
        "n_paired": len(keys),
        "greedy_success_count": greedy_success_count,
        "greedy_fail_count": len(greedy_fail),
        "retry_fail_count": len(retry_fail),
        "repaired_from_greedy_count": len(repaired_from_greedy),
        "repaired_from_retry_count": len(repaired_from_retry),
        "repair_rate_from_greedy_fail": safe_div(len(repaired_from_greedy), len(greedy_fail)),
        "repair_rate_from_retry_fail": safe_div(len(repaired_from_retry), len(retry_fail)),
        "final_success_count_from_greedy_and_search": final_success_count,
        "final_success_rate_from_greedy_and_search": safe_div(final_success_count, len(keys)),
        "planner_invalid_count_search": len(planner_invalid_rows),
        "planner_invalid_repaired_count_search": planner_invalid_repaired,
        "planner_invalid_repair_rate_search": safe_div(planner_invalid_repaired, len(planner_invalid_rows)),
        "greedy_failed_task_ids": greedy_fail,
        "repaired_from_greedy_task_ids": repaired_from_greedy,
        "paired_greedy_to_search": paired_delta(
            {k: bool(v.get("success")) for k, v in greedy.items()},
            {k: bool(v.get("success")) for k, v in search.items()},
        ),
        "paired_retry_to_search": paired_delta(
            {k: bool(v.get("success")) for k, v in retry.items()},
            {k: bool(v.get("success")) for k, v in search.items()},
        ) if retry else None,
    }


def build_common_failure_slice(runs: list[dict[str, Any]]) -> dict[str, Any]:
    if not runs:
        return {"n": 0, "task_ids": [], "per_model": []}

    greedy_key_sets = [set(run["maps"]["greedy"].keys()) for run in runs]
    common_keys = sorted(set.intersection(*greedy_key_sets)) if greedy_key_sets else []
    common_fail_ids = [
        task_id
        for task_id in common_keys
        if all(not bool(run["maps"]["greedy"].get(task_id, False)) for run in runs)
    ]

    per_model: list[dict[str, Any]] = []
    for run in runs:
        search_map = run["maps"]["search"]
        repaired_count = sum(1 for task_id in common_fail_ids if bool(search_map.get(task_id, False)))
        per_model.append(
            {
                "label": run["label"],
                "common_fail_count": len(common_fail_ids),
                "repaired_count": repaired_count,
                "repair_rate": safe_div(repaired_count, len(common_fail_ids)),
            }
        )

    return {"n": len(common_fail_ids), "task_ids": common_fail_ids, "per_model": per_model}


def build_run_report(label: str, run_dir: Path) -> dict[str, Any]:
    result_path = run_dir / "benchmark_results.json"
    if not result_path.exists():
        raise FileNotFoundError(f"Missing benchmark_results.json: {result_path}")
    results = load_json(result_path)
    if not isinstance(results, list):
        raise ValueError(f"Unexpected benchmark JSON structure: {result_path}")

    modes = {mode: mode_stats(results, mode) for mode in MODES}
    repair = search_repair_metrics(results)
    maps = {mode: mode_success_map(results, mode) for mode in MODES}
    return {
        "label": label,
        "run_dir": str(run_dir),
        "num_rows": len(results),
        "modes": modes,
        "repair": repair,
        "maps": maps,
    }


def build_cross_run_search_pairs(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    pairs: list[dict[str, Any]] = []
    for i in range(len(runs)):
        for j in range(i + 1, len(runs)):
            left = runs[i]
            right = runs[j]
            pair = paired_delta(left["maps"]["search"], right["maps"]["search"])
            pairs.append(
                {
                    "left": left["label"],
                    "right": right["label"],
                    **pair,
                }
            )
    return pairs


def build_markdown(report: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("# /app Local-Prior Closure Report")
    lines.append("")
    lines.append("## Aggregate by Model and Mode")
    lines.append("")
    lines.append(
        "| Model | Mode | n | Success | 95% CI | Planner Valid | Selected Valid | Search Selected | Avg Plan Len | Avg Runtime (s) |"
    )
    lines.append("| --- | --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: |")
    for run in report["runs"]:
        label = run["label"]
        for mode in MODES:
            row = run["modes"][mode]
            lines.append(
                "| {label} | {mode} | {n} | {succ:.3f} ({ok}/{n}) | [{lo:.3f}, {hi:.3f}] | {pv:.3f} | {sv:.3f} | {ss:.3f} | {plen:.3f} | {rt:.3f} |".format(
                    label=label,
                    mode=mode,
                    n=row["n"],
                    succ=row["success_rate"],
                    ok=row["success_count"],
                    lo=row["success_ci_low"],
                    hi=row["success_ci_high"],
                    pv=row["planner_valid_rate"],
                    sv=row["selected_valid_rate"],
                    ss=row["search_selected_rate"],
                    plen=row["avg_plan_length"],
                    rt=row["avg_runtime_sec"],
                )
            )

    lines.append("")
    lines.append("## Search Repair Gain (within each model)")
    lines.append("")
    lines.append(
        "| Model | Paired n | Prior Success@Greedy | Greedy Fail | Search Repairs | Repair@GreedyFail | Final Success (Prior+Repair) | Retry Fail | Search Repairs@Retry | Repair@RetryFail | Planner Invalid (search) | Repaired Planner-Invalid | Planner-Invalid Repair |"
    )
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for run in report["runs"]:
        repair = run["repair"]
        lines.append(
            "| {label} | {n} | {gs} | {gf} | {rg} | {rrg:.3f} | {fs:.3f} ({fsc}/{n}) | {rf} | {rr} | {rrr:.3f} | {pi} | {pir} | {pirr:.3f} |".format(
                label=run["label"],
                n=repair["n_paired"],
                gs=repair["greedy_success_count"],
                gf=repair["greedy_fail_count"],
                rg=repair["repaired_from_greedy_count"],
                rrg=repair["repair_rate_from_greedy_fail"],
                fs=repair["final_success_rate_from_greedy_and_search"],
                fsc=repair["final_success_count_from_greedy_and_search"],
                rf=repair["retry_fail_count"],
                rr=repair["repaired_from_retry_count"],
                rrr=repair["repair_rate_from_retry_fail"],
                pi=repair["planner_invalid_count_search"],
                pir=repair["planner_invalid_repaired_count_search"],
                pirr=repair["planner_invalid_repair_rate_search"],
            )
        )

    lines.append("")
    lines.append("## Common-Failure Slice (all models fail in planner-only greedy)")
    lines.append("")
    lines.append(
        "| Model | Common Fail Count | Repaired by Search | Repair Rate on Common-Fail |"
    )
    lines.append("| --- | ---: | ---: | ---: |")
    for row in report.get("common_failure_slice", {}).get("per_model", []):
        lines.append(
            "| {label} | {n} | {r} | {rr:.3f} |".format(
                label=row["label"],
                n=row["common_fail_count"],
                r=row["repaired_count"],
                rr=row["repair_rate"],
            )
        )

    lines.append("")
    lines.append("## Paired Significance (search success across models)")
    lines.append("")
    lines.append("| Right vs Left | Paired n | Improved | Worse | p-value |")
    lines.append("| --- | ---: | ---: | ---: | ---: |")
    for row in report["cross_run_search_pairs"]:
        lines.append(
            "| {right} vs {left} | {n} | {improved} | {worse} | {p:.12g} |".format(
                right=row["right"],
                left=row["left"],
                n=row["n"],
                improved=row["improved"],
                worse=row["worse"],
                p=row["p_value"],
            )
        )

    lines.append("")
    return "\n".join(lines) + "\n"


def build_latex(report: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("\\begin{table}[t]")
    lines.append("\\centering")
    lines.append("\\caption{/app local-prior closure benchmark under identical verifier-guided protocol.}")
    lines.append("\\label{tab:app-local-prior-closure}")
    lines.append("\\small")
    lines.append("\\begin{tabular}{llccccc}")
    lines.append("\\toprule")
    lines.append("Model & Mode & Success & Planner valid & Selected valid & Search selected & Avg len \\\\")
    lines.append("\\midrule")
    for run in report["runs"]:
        for mode in MODES:
            row = run["modes"][mode]
            lines.append(
                "{label} & {mode} & {succ:.3f} ({ok}/{n}) & {pv:.3f} & {sv:.3f} & {ss:.3f} & {plen:.2f} \\\\".format(
                    label=run["label"].replace("_", "\\_"),
                    mode=mode,
                    succ=row["success_rate"],
                    ok=row["success_count"],
                    n=row["n"],
                    pv=row["planner_valid_rate"],
                    sv=row["selected_valid_rate"],
                    ss=row["search_selected_rate"],
                    plen=row["avg_plan_length"],
                )
            )
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")
    lines.append("")

    lines.append("\\begin{table}[t]")
    lines.append("\\centering")
    lines.append("\\caption{Failure-subset and end-to-end repair diagnostics for local priors on /app tasks.}")
    lines.append("\\label{tab:app-local-prior-repair-gain}")
    lines.append("\\small")
    lines.append("\\begin{tabular}{lcccccc}")
    lines.append("\\toprule")
    lines.append("Model & Greedy success & Greedy fail & Repair@GreedyFail & Final success & Planner-invalid repair & Paired $n$ \\\\")
    lines.append("\\midrule")
    for run in report["runs"]:
        repair = run["repair"]
        lines.append(
            "{label} & {gs} & {gf} & {rg:.3f} & {fs:.3f} ({fsc}/{n}) & {pi:.3f} & {n} \\\\".format(
                label=run["label"].replace("_", "\\_"),
                gs=repair["greedy_success_count"],
                gf=repair["greedy_fail_count"],
                rg=repair["repair_rate_from_greedy_fail"],
                fs=repair["final_success_rate_from_greedy_and_search"],
                fsc=repair["final_success_count_from_greedy_and_search"],
                pi=repair["planner_invalid_repair_rate_search"],
                n=repair["n_paired"],
            )
        )
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")
    lines.append("")

    common_rows = report.get("common_failure_slice", {}).get("per_model", [])
    if common_rows:
        lines.append("\\begin{table}[t]")
        lines.append("\\centering")
        lines.append("\\caption{Common-failure repair slice (tasks where all models fail in greedy).}")
        lines.append("\\label{tab:app-local-prior-common-fail}")
        lines.append("\\small")
        lines.append("\\begin{tabular}{lccc}")
        lines.append("\\toprule")
        lines.append("Model & Common fail count & Repaired by search & Repair rate \\\\")
        lines.append("\\midrule")
        for row in common_rows:
            lines.append(
                "{label} & {n} & {r} & {rr:.3f} \\\\".format(
                    label=row["label"].replace("_", "\\_"),
                    n=row["common_fail_count"],
                    r=row["repaired_count"],
                    rr=row["repair_rate"],
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

    run_specs: list[tuple[str, Path]] = [parse_run_spec(spec, data_dir) for spec in args.run]
    run_reports = [build_run_report(label, run_dir) for label, run_dir in run_specs]
    cross_pairs = build_cross_run_search_pairs(run_reports)
    common_failure_slice = build_common_failure_slice(run_reports)

    report = {
        "runs": [{k: v for k, v in run.items() if k != "maps"} for run in run_reports],
        "cross_run_search_pairs": cross_pairs,
        "common_failure_slice": common_failure_slice,
    }

    out_json = out_dir / f"{args.report_prefix}_report.json"
    out_md = out_dir / f"{args.report_prefix}_report.md"
    out_tex = out_dir / f"{args.report_prefix}_tables.tex"

    out_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    out_md.write_text(build_markdown(report), encoding="utf-8")
    out_tex.write_text(build_latex(report), encoding="utf-8")

    print(f"[OK] Wrote {out_json}")
    print(f"[OK] Wrote {out_md}")
    print(f"[OK] Wrote {out_tex}")


if __name__ == "__main__":
    main()
