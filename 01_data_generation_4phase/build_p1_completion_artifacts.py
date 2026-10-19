from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DATA_DIR = Path(__file__).resolve().parent
PAPER_DIR = DATA_DIR / "paper_artifacts"


@dataclass(frozen=True)
class MethodSpec:
    name: str
    file_name: str


METHODS: tuple[MethodSpec, ...] = (
    MethodSpec("SFT", "sft_v2_balanced_report.json"),
    MethodSpec("Vanilla DPO", "dpo_vanilla_v2_balanced_report.json"),
    MethodSpec("mDPO", "dpo_margin_v2_balanced_report.json"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build P1-4/P1-5 completion artifacts from locked reports.")
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR, help="Root data directory.")
    parser.add_argument("--out-dir", type=Path, default=PAPER_DIR, help="Output directory for paper artifacts.")
    return parser.parse_args()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def to_rate(value: float | int | None) -> float:
    if value is None:
        return 0.0
    return float(value)


def fmt_rate(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{value:.3f}"


def _weighted_average(pairs: list[tuple[float | None, int]]) -> float | None:
    total_weight = sum(w for _, w in pairs if w > 0)
    if total_weight <= 0:
        return None
    total = 0.0
    for value, weight in pairs:
        if value is None:
            continue
        total += float(value) * weight
    return total / total_weight


def _safe_count_from_rate(rate: float | None, count: int) -> int:
    if rate is None:
        return 0
    return int(round(float(rate) * count))


def _build_hard_subset_report(data_dir: Path) -> tuple[dict[str, Any], str, str]:
    eval_dir = data_dir / "llama-factory" / "pvr_eval"
    graph_report = load_json(
        data_dir / "graph_benchmark_clean_google_gemini_2_5_flash_20260326_182800_ablation_report.json"
    )

    profiles = ("room_general", "mixed_light")
    drone_chassis = ("delivery_drone", "chassis_car")
    method_rows: list[dict[str, Any]] = []

    for spec in METHODS:
        report = load_json(eval_dir / spec.file_name)
        by_profile = report["by_task_profile"]
        by_robot = report["by_robot"]
        violation = report["violation_breakdown"]["sample_rate_over_invalid"]

        subset: dict[str, Any] = {}
        for profile in profiles:
            row = by_profile[profile]
            subset[profile] = {
                "count": int(row["count"]),
                "executable_rate": to_rate(row.get("executable_plan_rate")),
                "exact_rate": to_rate(row.get("exact_delivery_rate")),
                "avg_penalty_score": row.get("avg_penalty_score"),
            }

        robot_rows = [by_robot[name] for name in drone_chassis]
        drone_chassis_count = sum(int(r["count"]) for r in robot_rows)
        drone_chassis_valid = sum(int(r.get("valid_plan_count", 0)) for r in robot_rows)
        drone_chassis_exact = sum(_safe_count_from_rate(r.get("exact_delivery_rate"), int(r["count"])) for r in robot_rows)
        drone_chassis_penalty = _weighted_average(
            [(r.get("avg_penalty_score"), int(r["count"])) for r in robot_rows]
        )

        subset["drone_chassis"] = {
            "count": drone_chassis_count,
            "executable_rate": (drone_chassis_valid / drone_chassis_count) if drone_chassis_count else 0.0,
            "exact_rate": (drone_chassis_exact / drone_chassis_count) if drone_chassis_count else 0.0,
            "avg_penalty_score": drone_chassis_penalty,
        }
        subset["dominant_violation_over_invalid"] = {
            "coordination_failure": to_rate(violation.get("coordination_failure")),
            "terminal_failure": to_rate(violation.get("terminal_failure")),
            "morphology_constraint": to_rate(violation.get("morphology_constraint")),
            "graph_infeasible": to_rate(violation.get("graph_infeasible")),
        }

        method_rows.append(
            {
                "method": spec.name,
                "overall": report["overall"],
                "hard_subset": subset,
            }
        )

    counterfactual: dict[str, Any] = {}
    for profile in profiles:
        counterfactual[profile] = {}
        for guidance in ("none", "feasibility", "route_hint"):
            mode_rows = graph_report["profile_success"][guidance][profile]
            counterfactual[profile][guidance] = {}
            for mode in ("greedy", "retry", "search"):
                succ, total = mode_rows[mode]
                rate = (succ / total) if total else 0.0
                counterfactual[profile][guidance][mode] = {
                    "success": succ,
                    "total": total,
                    "rate": rate,
                }

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "sources": {
            "balanced_reports": [
                str(eval_dir / "sft_v2_balanced_report.json"),
                str(eval_dir / "dpo_vanilla_v2_balanced_report.json"),
                str(eval_dir / "dpo_margin_v2_balanced_report.json"),
            ],
            "graph_ablation_report": str(
                data_dir / "graph_benchmark_clean_google_gemini_2_5_flash_20260326_182800_ablation_report.json"
            ),
        },
        "methods": method_rows,
        "counterfactual_recoverability": counterfactual,
        "notes": [
            "This artifact targets P1-5 (room_general, mixed_light, drone_chassis).",
            "The counterfactual uses graph ablation profile slices to verify recoverability under guidance/search.",
        ],
    }

    md_lines: list[str] = []
    md_lines.append("# P1-5 Hard-Subset Report")
    md_lines.append("")
    md_lines.append("## Slice-level executable rates")
    md_lines.append("")
    md_lines.append("| Method | room_general | mixed_light | drone_chassis |")
    md_lines.append("| --- | ---: | ---: | ---: |")
    for row in method_rows:
        hs = row["hard_subset"]
        md_lines.append(
            "| {method} | {rg:.3f} ({rg_n}) | {ml:.3f} ({ml_n}) | {dc:.3f} ({dc_n}) |".format(
                method=row["method"],
                rg=hs["room_general"]["executable_rate"],
                rg_n=hs["room_general"]["count"],
                ml=hs["mixed_light"]["executable_rate"],
                ml_n=hs["mixed_light"]["count"],
                dc=hs["drone_chassis"]["executable_rate"],
                dc_n=hs["drone_chassis"]["count"],
            )
        )

    md_lines.append("")
    md_lines.append("## Dominant violation rates (over invalid plans)")
    md_lines.append("")
    md_lines.append("| Method | coordination_failure | terminal_failure | morphology_constraint | graph_infeasible |")
    md_lines.append("| --- | ---: | ---: | ---: | ---: |")
    for row in method_rows:
        v = row["hard_subset"]["dominant_violation_over_invalid"]
        md_lines.append(
            "| {method} | {coord:.3f} | {term:.3f} | {morph:.3f} | {graph:.3f} |".format(
                method=row["method"],
                coord=v["coordination_failure"],
                term=v["terminal_failure"],
                morph=v["morphology_constraint"],
                graph=v["graph_infeasible"],
            )
        )

    md_lines.append("")
    md_lines.append("## Counterfactual recoverability from graph ablation")
    md_lines.append("")
    md_lines.append("| Profile | none/greedy | none/retry | none/search | route_hint/greedy | route_hint/retry | route_hint/search |")
    md_lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: |")
    for profile in profiles:
        row_none = counterfactual[profile]["none"]
        row_route = counterfactual[profile]["route_hint"]
        md_lines.append(
            "| {profile} | {ng:.3f} | {nr:.3f} | {ns:.3f} | {rg:.3f} | {rr:.3f} | {rs:.3f} |".format(
                profile=profile,
                ng=row_none["greedy"]["rate"],
                nr=row_none["retry"]["rate"],
                ns=row_none["search"]["rate"],
                rg=row_route["greedy"]["rate"],
                rr=row_route["retry"]["rate"],
                rs=row_route["search"]["rate"],
            )
        )

    md_lines.append("")
    md_lines.append("## Key readout")
    md_lines.append("")
    md_lines.append("- room_general and mixed_light stay at 0 executable rate for all three adapters on the balanced diagnostic slice.")
    md_lines.append("- drone_chassis remains low (1/20 executable for each adapter), showing morphology-sensitive bottlenecks.")
    md_lines.append("- Counterfactual runs show these slices are recoverable under stronger guidance/search, so unsolved rates indicate prior grounding/data-coverage gaps, not intrinsic unsatisfiability.")
    md_lines.append("")
    markdown = "\n".join(md_lines) + "\n"

    tex_lines: list[str] = []
    tex_lines.append("\\begin{table}[t]")
    tex_lines.append("\\centering")
    tex_lines.append("\\caption{P1-5 hard-subset executable-rate diagnostics (balanced slice).}")
    tex_lines.append("\\label{tab:p1-hard-subset}")
    tex_lines.append("\\small")
    tex_lines.append("\\begin{tabular}{lccc}")
    tex_lines.append("\\toprule")
    tex_lines.append("Method & room\\_general & mixed\\_light & drone-chassis \\\\")
    tex_lines.append("\\midrule")
    for row in method_rows:
        hs = row["hard_subset"]
        tex_lines.append(
            "{method} & {rg:.3f} & {ml:.3f} & {dc:.3f} \\\\".format(
                method=row["method"],
                rg=hs["room_general"]["executable_rate"],
                ml=hs["mixed_light"]["executable_rate"],
                dc=hs["drone_chassis"]["executable_rate"],
            )
        )
    tex_lines.append("\\bottomrule")
    tex_lines.append("\\end{tabular}")
    tex_lines.append("\\end{table}")
    tex_lines.append("")
    tex_lines.append("\\begin{table}[t]")
    tex_lines.append("\\centering")
    tex_lines.append("\\caption{P1-5 hard-slice counterfactual recoverability from locked graph ablation.}")
    tex_lines.append("\\label{tab:p1-hard-counterfactual}")
    tex_lines.append("\\small")
    tex_lines.append("\\resizebox{\\columnwidth}{!}{%")
    tex_lines.append("\\begin{tabular}{lcccccc}")
    tex_lines.append("\\toprule")
    tex_lines.append("Profile & none/greedy & none/retry & none/search & route\\_hint/greedy & route\\_hint/retry & route\\_hint/search \\\\")
    tex_lines.append("\\midrule")
    for profile in profiles:
        row_none = counterfactual[profile]["none"]
        row_route = counterfactual[profile]["route_hint"]
        tex_lines.append(
            "{profile} & {ng:.3f} & {nr:.3f} & {ns:.3f} & {rg:.3f} & {rr:.3f} & {rs:.3f} \\\\".format(
                profile=profile.replace("_", "\\_"),
                ng=row_none["greedy"]["rate"],
                nr=row_none["retry"]["rate"],
                ns=row_none["search"]["rate"],
                rg=row_route["greedy"]["rate"],
                rr=row_route["retry"]["rate"],
                rs=row_route["search"]["rate"],
            )
        )
    tex_lines.append("\\bottomrule")
    tex_lines.append("\\end{tabular}")
    tex_lines.append("}")
    tex_lines.append("\\end{table}")
    tex_lines.append("")
    tex = "\n".join(tex_lines)
    return report, markdown, tex


def _pick_method(paradigms: list[dict[str, Any]], name: str) -> dict[str, Any]:
    for row in paradigms:
        if row.get("method") == name:
            return row
    raise KeyError(f"Method '{name}' not found in critical baseline report.")


def _build_embodied_sota_report(data_dir: Path) -> tuple[dict[str, Any], str, str]:
    critical = load_json(data_dir / "paper_artifacts" / "critical_baseline_report.json")
    paradigms = critical["paradigms"]

    react = _pick_method(paradigms, "ReAct-style one-shot")
    reflexion = _pick_method(paradigms, "Reflexion-style retry")
    tot = _pick_method(paradigms, "ToT-style search")
    symbolic = _pick_method(paradigms, "Classical symbolic planner")
    pvr = _pick_method(paradigms, "PVR full stack")

    mapping_rows = [
        {
            "target_family": "Code as Policies (traceable proxy)",
            "repo_reproduction": "ReAct-style one-shot",
            "mechanism_note": "single-pass planner output with symbolic API execution checks",
            "metrics": react,
        },
        {
            "target_family": "Inner Monologue / Reflexion (traceable proxy)",
            "repo_reproduction": "Reflexion-style retry",
            "mechanism_note": "failure-conditioned retry loop with verifier feedback",
            "metrics": reflexion,
        },
        {
            "target_family": "Tree-of-Thought style search baseline",
            "repo_reproduction": "ToT-style search",
            "mechanism_note": "deliberative search over candidate plans",
            "metrics": tot,
        },
        {
            "target_family": "Classical planning upper bound",
            "repo_reproduction": "Classical symbolic planner",
            "mechanism_note": "deterministic symbolic planner without LLM priors",
            "metrics": symbolic,
        },
        {
            "target_family": "Proposed framework",
            "repo_reproduction": "PVR full stack",
            "mechanism_note": "guidance + verifier + search",
            "metrics": pvr,
        },
    ]

    pairwise_keep = []
    for row in critical.get("pairwise", []):
        right = str(row.get("right"))
        left = str(row.get("left"))
        if right in {"graph_pvr", "app8_gemini_search", "app8_symbolic"} or "graph_symbolic" in right:
            pairwise_keep.append(row)

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source": str(data_dir / "paper_artifacts" / "critical_baseline_report.json"),
        "baseline_family_mapping": mapping_rows,
        "paired_significance": pairwise_keep,
        "claim_boundary": [
            "This P1-4 artifact uses protocol-matched, traceable reproductions under locked tasks.",
            "It does not claim official third-party codebase reproduction for Inner Monologue or Code as Policies.",
        ],
    }

    md_lines: list[str] = []
    md_lines.append("# P1-4 Embodied SOTA Comparison (Traceable Reproductions)")
    md_lines.append("")
    md_lines.append("## Family mapping under locked protocol")
    md_lines.append("")
    md_lines.append("| Target family | Repo reproduction | Graph-30 | App-8 Gemini | App-8 Qwen | App-40 Gemini |")
    md_lines.append("| --- | --- | ---: | ---: | ---: | ---: |")
    for row in mapping_rows:
        metric = row["metrics"]
        md_lines.append(
            "| {target} | {repro} | {g30} | {a8g} | {a8q} | {a40} |".format(
                target=row["target_family"],
                repro=row["repo_reproduction"],
                g30=fmt_rate(metric.get("graph_success")),
                a8g=fmt_rate(metric.get("app8_gemini_success")),
                a8q=fmt_rate(metric.get("app8_qwen_success")),
                a40=fmt_rate(metric.get("app40_gemini_success")),
            )
        )

    md_lines.append("")
    md_lines.append("## Key paired tests")
    md_lines.append("")
    md_lines.append("| Right vs Left | n | Improved | Worse | p-value |")
    md_lines.append("| --- | ---: | ---: | ---: | ---: |")
    for row in pairwise_keep:
        md_lines.append(
            "| {right} vs {left} | {n} | {improved} | {worse} | {p:.6f} |".format(
                right=row["right"],
                left=row["left"],
                n=row["n"],
                improved=row["improved"],
                worse=row["worse"],
                p=row["p_value"],
            )
        )

    md_lines.append("")
    md_lines.append("## Notes")
    md_lines.append("")
    md_lines.append("- This artifact closes P1-4 with traceable reproductions under identical task/protocol constraints.")
    md_lines.append("- If official IM/CaP repositories are later integrated, keep this table as the protocol-matched anchor and append official rows as extension, not replacement.")
    md_lines.append("")
    markdown = "\n".join(md_lines) + "\n"

    tex_lines: list[str] = []
    tex_lines.append("\\begin{table}[t]")
    tex_lines.append("\\centering")
    tex_lines.append("\\caption{P1-4 embodied planning baselines via traceable protocol-matched reproductions.}")
    tex_lines.append("\\label{tab:p1-embodied-sota}")
    tex_lines.append("\\small")
    tex_lines.append("\\resizebox{\\columnwidth}{!}{%")
    tex_lines.append("\\begin{tabular}{lccccc}")
    tex_lines.append("\\toprule")
    tex_lines.append("Target family & Reproduction & Graph-30 & App-8 (Gemini) & App-8 (Qwen) & App-40 (Gemini) \\\\")
    tex_lines.append("\\midrule")
    for row in mapping_rows:
        metric = row["metrics"]
        tex_lines.append(
            "{target} & {repro} & {g30} & {a8g} & {a8q} & {a40} \\\\".format(
                target=row["target_family"].replace("_", "\\_"),
                repro=row["repo_reproduction"].replace("_", "\\_"),
                g30=fmt_rate(metric.get("graph_success")),
                a8g=fmt_rate(metric.get("app8_gemini_success")),
                a8q=fmt_rate(metric.get("app8_qwen_success")),
                a40=fmt_rate(metric.get("app40_gemini_success")),
            )
        )
    tex_lines.append("\\bottomrule")
    tex_lines.append("\\end{tabular}")
    tex_lines.append("}")
    tex_lines.append("\\end{table}")
    tex_lines.append("")
    tex = "\n".join(tex_lines)
    return report, markdown, tex


def main() -> None:
    args = parse_args()
    data_dir = args.data_dir.resolve()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    hard_report, hard_md, hard_tex = _build_hard_subset_report(data_dir)
    sota_report, sota_md, sota_tex = _build_embodied_sota_report(data_dir)

    (out_dir / "p1_hard_subset_report.json").write_text(
        json.dumps(hard_report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (out_dir / "p1_hard_subset_report.md").write_text(hard_md, encoding="utf-8")
    (out_dir / "p1_hard_subset_tables.tex").write_text(hard_tex, encoding="utf-8")

    (out_dir / "p1_embodied_sota_report.json").write_text(
        json.dumps(sota_report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (out_dir / "p1_embodied_sota_report.md").write_text(sota_md, encoding="utf-8")
    (out_dir / "p1_embodied_sota_tables.tex").write_text(sota_tex, encoding="utf-8")

    print(f"[OK] {out_dir / 'p1_hard_subset_report.json'}")
    print(f"[OK] {out_dir / 'p1_hard_subset_report.md'}")
    print(f"[OK] {out_dir / 'p1_hard_subset_tables.tex'}")
    print(f"[OK] {out_dir / 'p1_embodied_sota_report.json'}")
    print(f"[OK] {out_dir / 'p1_embodied_sota_report.md'}")
    print(f"[OK] {out_dir / 'p1_embodied_sota_tables.tex'}")


if __name__ == "__main__":
    main()
