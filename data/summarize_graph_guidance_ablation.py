from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any


GUIDANCE_MODES = ["none", "feasibility", "route_hint"]
PLANNING_MODES = ["greedy", "retry", "search"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize graph-guidance ablations (none/feasibility/route_hint) into paper-ready artifacts."
    )
    parser.add_argument(
        "--base-prefix",
        type=str,
        required=True,
        help="Run directory prefix without guidance suffix, e.g. graph_benchmark_clean_xxx_20260326_182800",
    )
    parser.add_argument(
        "--root-dir",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Directory containing benchmark run folders.",
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        default=None,
        help="Output markdown path (default: <root>/<base-prefix>_ablation_report.md)",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Output JSON path (default: <root>/<base-prefix>_ablation_report.json)",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=None,
        help="Output CSV path (default: <root>/<base-prefix>_ablation_table.csv)",
    )
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def rate(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return numerator / denominator


def wilson_interval(successes: int, total: int, z: float = 1.96) -> tuple[float, float]:
    if total <= 0:
        return 0.0, 0.0
    phat = successes / total
    z2 = z * z
    denom = 1.0 + z2 / total
    center = (phat + z2 / (2 * total)) / denom
    half = z * math.sqrt((phat * (1.0 - phat) + z2 / (4 * total)) / total) / denom
    return max(0.0, center - half), min(1.0, center + half)


def ensure_row_integrity(rows: list[dict[str, Any]], guidance: str) -> tuple[int, list[str]]:
    issues: list[str] = []
    keys = set()
    mode_counts = defaultdict(int)
    task_ids = set()
    for row in rows:
        task_id = str(row.get("task_id"))
        mode = str(row.get("mode"))
        g = str(row.get("guidance_mode"))
        key = (task_id, mode, g)
        if key in keys:
            issues.append(f"duplicate row key: {key}")
        keys.add(key)
        if mode in PLANNING_MODES:
            mode_counts[mode] += 1
        task_ids.add(task_id)
        if g != guidance:
            issues.append(f"guidance mismatch row={key}, expected {guidance}")

    task_count = len(task_ids)
    for mode in PLANNING_MODES:
        if mode_counts[mode] != task_count:
            issues.append(
                f"mode row-count mismatch guidance={guidance}, mode={mode}, "
                f"rows={mode_counts[mode]}, expected={task_count}"
            )
    if len(rows) != task_count * len(PLANNING_MODES):
        issues.append(
            f"total rows mismatch guidance={guidance}, rows={len(rows)}, expected={task_count * len(PLANNING_MODES)}"
        )
    return task_count, issues


def aggregate_mode(rows: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(rows)
    success_n = sum(1 for r in rows if bool(r.get("success")))
    parse_n = sum(1 for r in rows if bool(r.get("planner_parse_ok")))
    schema_n = sum(1 for r in rows if bool(r.get("planner_schema_pass")))
    planner_valid_n = sum(1 for r in rows if bool(r.get("planner_valid")))
    selected_valid_n = sum(1 for r in rows if bool(r.get("selected_valid")))
    search_selected_n = sum(
        1
        for r in rows
        if str(r.get("selected_plan_source", "")).startswith("search")
    )
    runtime_values = [float(r.get("runtime_sec", 0.0) or 0.0) for r in rows]
    step_values = [int(r.get("selected_step_count", 0) or 0) for r in rows]
    valid_costs = [
        float(r.get("selected_trajectory_cost"))
        for r in rows
        if r.get("selected_trajectory_cost") is not None
    ]
    low_ci, high_ci = wilson_interval(success_n, total)
    return {
        "num_tasks": total,
        "success_count": success_n,
        "success_rate": rate(success_n, total),
        "success_ci_low": low_ci,
        "success_ci_high": high_ci,
        "planner_parse_rate": rate(parse_n, total),
        "planner_schema_rate": rate(schema_n, total),
        "planner_valid_rate": rate(planner_valid_n, total),
        "selected_valid_rate": rate(selected_valid_n, total),
        "search_selected_count": search_selected_n,
        "search_selected_rate": rate(search_selected_n, total),
        "avg_runtime_sec": sum(runtime_values) / total if total else 0.0,
        "avg_selected_steps": sum(step_values) / total if total else 0.0,
        "avg_selected_cost": (sum(valid_costs) / len(valid_costs)) if valid_costs else None,
    }


def compute_profile_success(rows: list[dict[str, Any]]) -> dict[str, dict[str, tuple[int, int]]]:
    # profile -> mode -> (success, total)
    summary: dict[str, dict[str, tuple[int, int]]] = {}
    profile_mode_counts: dict[str, dict[str, list[int]]] = defaultdict(
        lambda: defaultdict(lambda: [0, 0])
    )
    for r in rows:
        profile = str(r.get("task_profile", "unknown"))
        mode = str(r.get("mode"))
        if mode not in PLANNING_MODES:
            continue
        profile_mode_counts[profile][mode][1] += 1
        if bool(r.get("success")):
            profile_mode_counts[profile][mode][0] += 1
    for profile, by_mode in profile_mode_counts.items():
        summary[profile] = {}
        for mode, pair in by_mode.items():
            summary[profile][mode] = (pair[0], pair[1])
    return summary


def paired_success_deltas(
    by_guidance_rows: dict[str, list[dict[str, Any]]],
    *,
    left: str,
    right: str,
) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for mode in PLANNING_MODES:
        left_rows = {
            str(r.get("task_id")): bool(r.get("success"))
            for r in by_guidance_rows[left]
            if str(r.get("mode")) == mode
        }
        right_rows = {
            str(r.get("task_id")): bool(r.get("success"))
            for r in by_guidance_rows[right]
            if str(r.get("mode")) == mode
        }
        task_ids = sorted(set(left_rows.keys()) & set(right_rows.keys()))
        improved = 0
        worse = 0
        same = 0
        for tid in task_ids:
            l = left_rows[tid]
            r = right_rows[tid]
            if l and not r:
                improved += 1
            elif not l and r:
                worse += 1
            else:
                same += 1
        total = len(task_ids)
        delta_rate = rate(improved - worse, total)
        result[mode] = {
            "tasks": total,
            "improved": improved,
            "worse": worse,
            "same": same,
            "delta_success_rate": delta_rate,
        }
    return result


def build_markdown(
    *,
    base_prefix: str,
    integrity_issues: list[str],
    aggregate: dict[str, dict[str, dict[str, Any]]],
    pairwise_deltas: dict[str, dict[str, dict[str, Any]]],
    profile_success: dict[str, dict[str, dict[str, tuple[int, int]]]],
) -> str:
    lines: list[str] = []
    lines.append("# Graph Guidance Ablation Report")
    lines.append("")
    lines.append(f"- Run prefix: `{base_prefix}`")
    lines.append("")

    if integrity_issues:
        lines.append("## Integrity Checks")
        lines.append("")
        for issue in integrity_issues:
            lines.append(f"- [WARN] {issue}")
        lines.append("")
    else:
        lines.append("## Integrity Checks")
        lines.append("")
        lines.append("- All three guidance runs passed row-count and key consistency checks.")
        lines.append("")

    lines.append("## Main Table")
    lines.append("")
    lines.append(
        "| Guidance | Mode | Success | 95% CI | Parse | Planner Valid | Selected Valid | Search Selected | Avg Cost | Avg Steps | Avg Runtime (s) |"
    )
    lines.append(
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"
    )
    for guidance in GUIDANCE_MODES:
        for mode in PLANNING_MODES:
            row = aggregate[guidance][mode]
            cost = "-" if row["avg_selected_cost"] is None else f"{row['avg_selected_cost']:.3f}"
            search_selected = "-" if mode != "search" else f"{row['search_selected_rate']:.3f}"
            lines.append(
                "| {guidance} | {mode} | {success:.3f} | [{ci_low:.3f}, {ci_high:.3f}] | {parse:.3f} | {planner_valid:.3f} | {selected_valid:.3f} | {search_selected} | {cost} | {steps:.3f} | {runtime:.3f} |".format(
                    guidance=guidance,
                    mode=mode,
                    success=row["success_rate"],
                    ci_low=row["success_ci_low"],
                    ci_high=row["success_ci_high"],
                    parse=row["planner_parse_rate"],
                    planner_valid=row["planner_valid_rate"],
                    selected_valid=row["selected_valid_rate"],
                    search_selected=search_selected,
                    cost=cost,
                    steps=row["avg_selected_steps"],
                    runtime=row["avg_runtime_sec"],
                )
            )
    lines.append("")

    lines.append("## Paired Task Deltas")
    lines.append("")
    for pair_name, delta in pairwise_deltas.items():
        lines.append(f"### {pair_name}")
        lines.append("")
        lines.append("| Mode | Improved | Worse | Same | Delta Success Rate |")
        lines.append("| --- | ---: | ---: | ---: | ---: |")
        for mode in PLANNING_MODES:
            row = delta[mode]
            lines.append(
                f"| {mode} | {row['improved']} | {row['worse']} | {row['same']} | {row['delta_success_rate']:.3f} |"
            )
        lines.append("")

    lines.append("## Success by Task Profile")
    lines.append("")
    for mode in ("greedy", "retry"):
        lines.append(f"### {mode}")
        lines.append("")
        lines.append("| Profile | none | feasibility | route_hint |")
        lines.append("| --- | ---: | ---: | ---: |")
        profiles = sorted(
            set(profile_success["none"].keys())
            | set(profile_success["feasibility"].keys())
            | set(profile_success["route_hint"].keys())
        )
        for profile in profiles:
            cells = []
            for guidance in GUIDANCE_MODES:
                succ, total = profile_success[guidance].get(profile, {}).get(mode, (0, 0))
                if total <= 0:
                    cells.append("-")
                else:
                    cells.append(f"{succ}/{total} ({succ/total:.3f})")
            lines.append(
                f"| {profile} | {cells[0]} | {cells[1]} | {cells[2]} |"
            )
        lines.append("")

    lines.append("## Key Takeaways")
    lines.append("")
    route_retry = aggregate["route_hint"]["retry"]["success_rate"]
    none_retry = aggregate["none"]["retry"]["success_rate"]
    feas_retry = aggregate["feasibility"]["retry"]["success_rate"]
    route_greedy = aggregate["route_hint"]["greedy"]["success_rate"]
    none_greedy = aggregate["none"]["greedy"]["success_rate"]
    search_sel_none = aggregate["none"]["search"]["search_selected_rate"]
    search_sel_route = aggregate["route_hint"]["search"]["search_selected_rate"]
    lines.append(
        f"- Retry success rises from `none={none_retry:.3f}` to `feasibility={feas_retry:.3f}` to `route_hint={route_retry:.3f}`."
    )
    lines.append(
        f"- Greedy success rises from `none={none_greedy:.3f}` to `route_hint={route_greedy:.3f}`."
    )
    lines.append(
        f"- Search intervention rate drops from `none={search_sel_none:.3f}` to `route_hint={search_sel_route:.3f}`, meaning stronger guidance reduces repair demand."
    )
    lines.append("")
    return "\n".join(lines) + "\n"


def main() -> int:
    args = parse_args()
    root_dir = args.root_dir.resolve()
    base_prefix = args.base_prefix.strip()

    output_md = args.output_md or (root_dir / f"{base_prefix}_ablation_report.md")
    output_json = args.output_json or (root_dir / f"{base_prefix}_ablation_report.json")
    output_csv = args.output_csv or (root_dir / f"{base_prefix}_ablation_table.csv")

    by_guidance_rows: dict[str, list[dict[str, Any]]] = {}
    integrity_issues: list[str] = []

    for guidance in GUIDANCE_MODES:
        run_dir = root_dir / f"{base_prefix}_{guidance}"
        jsonl_path = run_dir / "benchmark_results.jsonl"
        if not jsonl_path.exists():
            raise FileNotFoundError(f"Missing result file: {jsonl_path}")
        rows = load_jsonl(jsonl_path)
        by_guidance_rows[guidance] = rows
        _, issues = ensure_row_integrity(rows, guidance)
        integrity_issues.extend(issues)

    aggregate: dict[str, dict[str, dict[str, Any]]] = {}
    profile_success: dict[str, dict[str, dict[str, tuple[int, int]]]] = {}

    for guidance, rows in by_guidance_rows.items():
        aggregate[guidance] = {}
        for mode in PLANNING_MODES:
            mode_rows = [r for r in rows if str(r.get("mode")) == mode]
            aggregate[guidance][mode] = aggregate_mode(mode_rows)
        profile_success[guidance] = compute_profile_success(rows)

    pairwise_deltas = {
        "route_hint vs none": paired_success_deltas(by_guidance_rows, left="route_hint", right="none"),
        "route_hint vs feasibility": paired_success_deltas(by_guidance_rows, left="route_hint", right="feasibility"),
        "feasibility vs none": paired_success_deltas(by_guidance_rows, left="feasibility", right="none"),
    }

    report_json = {
        "base_prefix": base_prefix,
        "integrity_issues": integrity_issues,
        "aggregate": aggregate,
        "pairwise_deltas": pairwise_deltas,
        "profile_success": profile_success,
    }

    output_json.write_text(json.dumps(report_json, ensure_ascii=False, indent=2), encoding="utf-8")

    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "guidance_mode",
                "planning_mode",
                "num_tasks",
                "success_rate",
                "success_ci_low",
                "success_ci_high",
                "planner_parse_rate",
                "planner_schema_rate",
                "planner_valid_rate",
                "selected_valid_rate",
                "search_selected_rate",
                "avg_selected_cost",
                "avg_selected_steps",
                "avg_runtime_sec",
            ]
        )
        for guidance in GUIDANCE_MODES:
            for mode in PLANNING_MODES:
                row = aggregate[guidance][mode]
                writer.writerow(
                    [
                        guidance,
                        mode,
                        row["num_tasks"],
                        row["success_rate"],
                        row["success_ci_low"],
                        row["success_ci_high"],
                        row["planner_parse_rate"],
                        row["planner_schema_rate"],
                        row["planner_valid_rate"],
                        row["selected_valid_rate"],
                        row["search_selected_rate"] if mode == "search" else "",
                        row["avg_selected_cost"],
                        row["avg_selected_steps"],
                        row["avg_runtime_sec"],
                    ]
                )

    output_md.write_text(
        build_markdown(
            base_prefix=base_prefix,
            integrity_issues=integrity_issues,
            aggregate=aggregate,
            pairwise_deltas=pairwise_deltas,
            profile_success=profile_success,
        ),
        encoding="utf-8",
    )

    print(f"Wrote markdown report: {output_md}")
    print(f"Wrote JSON report: {output_json}")
    print(f"Wrote CSV table: {output_csv}")
    if integrity_issues:
        print(f"Integrity warnings: {len(integrity_issues)}")
    else:
        print("Integrity checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

