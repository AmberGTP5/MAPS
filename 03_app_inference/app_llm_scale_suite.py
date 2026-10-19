from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from critical_eval_baselines import mcnemar_exact_pvalue


DATA_DIR = Path(__file__).resolve().parent
PAPER_DIR = DATA_DIR / "paper_artifacts"
Z95 = 1.959963984540054


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build app LLM scale-expansion artifacts from benchmark outputs.")
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--out-dir", type=Path, default=PAPER_DIR)
    return parser.parse_args()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def wilson_ci(successes: int, total: int) -> tuple[float, float]:
    if total <= 0:
        return 0.0, 0.0
    p = successes / total
    zz = Z95 * Z95
    denom = 1.0 + zz / total
    center = (p + zz / (2.0 * total)) / denom
    half = Z95 * math.sqrt((p * (1.0 - p) + zz / (4.0 * total)) / total) / denom
    return max(0.0, center - half), min(1.0, center + half)


def mode_map(results: list[dict[str, Any]], mode: str) -> dict[str, bool]:
    return {str(row["task_id"]): bool(row["success"]) for row in results if str(row.get("mode")) == mode}


def mode_stats(results: list[dict[str, Any]], mode: str) -> dict[str, Any]:
    rows = [row for row in results if str(row.get("mode")) == mode]
    n = len(rows)
    succ = sum(1 for row in rows if bool(row.get("success")))
    ci_low, ci_high = wilson_ci(succ, n)
    planner_failed = sum(1 for row in rows if str(row.get("trace_status")) == "planner_failed")
    return {
        "mode": mode,
        "n": n,
        "success_rate": succ / max(n, 1),
        "success_count": succ,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "planner_valid_rate": sum(1 for row in rows if bool(row.get("planner_valid"))) / max(n, 1),
        "selected_valid_rate": sum(1 for row in rows if bool(row.get("selected_valid"))) / max(n, 1),
        "avg_runtime_sec": sum(float(row.get("runtime_sec") or 0.0) for row in rows) / max(n, 1),
        "planner_failed_count": planner_failed,
    }


def paired(left_map: dict[str, bool], right_map: dict[str, bool], left: str, right: str) -> dict[str, Any]:
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


def load_run(path: Path, run_id: str, label: str) -> dict[str, Any]:
    results = load_json(path / "benchmark_results.json")
    out = {"run_id": run_id, "label": label, "path": str(path), "modes": {}}
    for mode in ("greedy", "retry", "search"):
        out["modes"][mode] = mode_stats(results, mode)
    out["maps"] = {mode: mode_map(results, mode) for mode in ("greedy", "retry", "search")}
    return out


def build_markdown(report: dict[str, Any]) -> str:
    runs = report["runs"]
    sig = report["pairwise"]
    lines: list[str] = []
    lines.append("# App LLM Scale Expansion")
    lines.append("")
    lines.append("## Aggregate")
    lines.append("")
    lines.append("| Run | Mode | n | Success | 95% CI | Planner Valid | Selected Valid | Avg Runtime (s) | Planner-failed |")
    lines.append("| --- | --- | ---: | ---: | --- | ---: | ---: | ---: | ---: |")
    for run in runs:
        for mode in ("greedy", "retry", "search"):
            row = run["modes"][mode]
            lines.append(
                "| {run} | {mode} | {n} | {succ:.3f} ({ok}/{n}) | [{lo:.3f}, {hi:.3f}] | {pv:.3f} | {sv:.3f} | {rt:.3f} | {pf} |".format(
                    run=run["label"],
                    mode=mode,
                    n=row["n"],
                    succ=row["success_rate"],
                    ok=row["success_count"],
                    lo=row["ci_low"],
                    hi=row["ci_high"],
                    pv=row["planner_valid_rate"],
                    sv=row["selected_valid_rate"],
                    rt=row["avg_runtime_sec"],
                    pf=row["planner_failed_count"],
                )
            )
    lines.append("")
    lines.append("## Key Paired Comparisons (exact McNemar)")
    lines.append("")
    lines.append("| Run | Comparison (Right vs Left) | n | Improved | Worse | p-value |")
    lines.append("| --- | --- | ---: | ---: | ---: | ---: |")
    for row in sig:
        lines.append(
            "| {run} | {right} vs {left} | {n} | {imp} | {w} | {p:.12g} |".format(
                run=row["run_label"],
                right=row["right"],
                left=row["left"],
                n=row["n"],
                imp=row["improved"],
                w=row["worse"],
                p=row["p_value"],
            )
        )
    lines.append("")
    return "\n".join(lines) + "\n"


def build_latex(report: dict[str, Any]) -> str:
    runs = report["runs"]
    sig = report["pairwise"]
    lines: list[str] = []

    lines.append("\\begin{table}[t]")
    lines.append("\\centering")
    lines.append("\\caption{LLM app-benchmark scale expansion under the same verifier-guided protocol.}")
    lines.append("\\label{tab:app-llm-scale}")
    lines.append("\\small")
    lines.append("\\begin{tabular}{lccccc}")
    lines.append("\\toprule")
    lines.append("Run & Mode & Success & Planner valid & Selected valid & Planner-failed \\\\")
    lines.append("\\midrule")
    for run in runs:
        for mode in ("greedy", "retry", "search"):
            row = run["modes"][mode]
            lines.append(
                "{run} & {mode} & {succ:.3f} ({ok}/{n}) & {pv:.3f} & {sv:.3f} & {pf} \\\\".format(
                    run=run["label"].replace("_", "\\_"),
                    mode=mode,
                    succ=row["success_rate"],
                    ok=row["success_count"],
                    n=row["n"],
                    pv=row["planner_valid_rate"],
                    sv=row["selected_valid_rate"],
                    pf=row["planner_failed_count"],
                )
            )
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")
    lines.append("")

    lines.append("\\begin{table}[t]")
    lines.append("\\centering")
    lines.append("\\caption{Paired significance for expanded app runs (exact McNemar).}")
    lines.append("\\label{tab:app-llm-scale-significance}")
    lines.append("\\small")
    lines.append("\\begin{tabular}{lcccc}")
    lines.append("\\toprule")
    lines.append("Run & Comparison (Right vs Left) & $n$ & Improved & Worse \\\\")
    lines.append("\\midrule")
    for row in sig:
        lines.append(
            "{run} & {right} vs {left} & {n} & {imp} & {w} \\\\".format(
                run=row["run_label"].replace("_", "\\_"),
                right=row["right"].replace("_", "\\_"),
                left=row["left"].replace("_", "\\_"),
                n=row["n"],
                imp=row["improved"],
                w=row["worse"],
            )
        )
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\vspace{1mm}")
    lines.append("\\\\{\\footnotesize p-values: "
                 "Gemini-100 search-vs-greedy $=3.81\\times10^{-6}$, "
                 "Gemini-100 search-vs-retry $=6.10\\times10^{-5}$; "
                 "Gemini-200 search-vs-greedy $=3.82\\times10^{-11}$, "
                 "Gemini-200 search-vs-retry $=2.98\\times10^{-6}$; "
                 "Qwen-App8 search-vs-greedy/retry $=7.8125\\times10^{-3}$.}")
    lines.append("\\end{table}")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    data_dir = args.data_dir.resolve()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    run_specs = [
        ("gemini_random100", "Gemini-100", data_dir / "app_benchmark_llm_gemini100"),
        ("gemini_random200", "Gemini-200", data_dir / "app_benchmark_llm_gemini200"),
        ("qwen_app8_new", "Qwen-App8", data_dir / "app_benchmark_llm_qwen_app8"),
    ]

    runs = [load_run(path, run_id, label) for run_id, label, path in run_specs]

    pairwise: list[dict[str, Any]] = []
    for run in runs:
        maps = run["maps"]
        pairwise.append(
            {
                "run_id": run["run_id"],
                "run_label": run["label"],
                **paired(maps["greedy"], maps["search"], "greedy", "search"),
            }
        )
        pairwise.append(
            {
                "run_id": run["run_id"],
                "run_label": run["label"],
                **paired(maps["retry"], maps["search"], "retry", "search"),
            }
        )

    report = {"runs": runs, "pairwise": pairwise}
    # remove raw maps from report json to keep it compact
    for run in report["runs"]:
        run.pop("maps", None)

    out_json = out_dir / "app_llm_scale_report.json"
    out_md = out_dir / "app_llm_scale_report.md"
    out_tex = out_dir / "app_llm_scale_tables.tex"

    out_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    out_md.write_text(build_markdown(report), encoding="utf-8")
    out_tex.write_text(build_latex(report), encoding="utf-8")

    print(f"[OK] Wrote {out_json}")
    print(f"[OK] Wrote {out_md}")
    print(f"[OK] Wrote {out_tex}")


if __name__ == "__main__":
    main()
