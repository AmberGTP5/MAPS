from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


Z95 = 1.959963984540054


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate multiple RWARE pilot reports into one suite report.")
    parser.add_argument(
        "--reports",
        nargs="+",
        required=True,
        help="Paths to rware_generalization_report.json files.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "paper_artifacts",
        help="Output directory for suite report artifacts.",
    )
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


def build_markdown(report: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("# RWARE Generalization Suite (Critical Expansion)")
    lines.append("")
    lines.append("| Environment | Episodes | Random Success | Heuristic Success | Success McNemar p | Random Early@H | Heuristic Early@H | Early@H McNemar p |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for row in report["environments"]:
        lines.append(
            "| {env} | {episodes} | {rs:.3f} ({rsc}/{episodes}) | {hs:.3f} ({hsc}/{episodes}) | {sp:.6f} | {re:.3f} ({rec}/{episodes}) | {he:.3f} ({hec}/{episodes}) | {ep:.6f} |".format(
                env=row["env_id"],
                episodes=row["episodes"],
                rs=row["random"]["success_rate"],
                rsc=row["random"]["success_count"],
                hs=row["heuristic"]["success_rate"],
                hsc=row["heuristic"]["success_count"],
                sp=row["paired_success_test"]["p_value"],
                re=row["random"]["early_reward_rate"],
                rec=row["random"]["early_reward_count"],
                he=row["heuristic"]["early_reward_rate"],
                hec=row["heuristic"]["early_reward_count"],
                ep=row["paired_early_reward_test"]["p_value"],
            )
        )
    lines.append("")
    pooled = report["pooled"]
    lines.append("## Pooled Across Environments")
    lines.append("")
    lines.append(
        "- Success: random={rs:.3f} ({rsc}/{n}), heuristic={hs:.3f} ({hsc}/{n}), paired exact McNemar p={p:.6f} (improved={imp}, worse={wor}).".format(
            rs=pooled["random"]["success_rate"],
            rsc=pooled["random"]["success_count"],
            hs=pooled["heuristic"]["success_rate"],
            hsc=pooled["heuristic"]["success_count"],
            n=pooled["episodes_total"],
            p=pooled["paired_success_test"]["p_value"],
            imp=pooled["paired_success_test"]["improved"],
            wor=pooled["paired_success_test"]["worse"],
        )
    )
    lines.append(
        "- Early-Reward@H: random={rs:.3f} ({rsc}/{n}), heuristic={hs:.3f} ({hsc}/{n}), paired exact McNemar p={p:.6f} (improved={imp}, worse={wor}).".format(
            rs=pooled["random"]["early_reward_rate"],
            rsc=pooled["random"]["early_reward_count"],
            hs=pooled["heuristic"]["early_reward_rate"],
            hsc=pooled["heuristic"]["early_reward_count"],
            n=pooled["episodes_total"],
            p=pooled["paired_early_reward_test"]["p_value"],
            imp=pooled["paired_early_reward_test"]["improved"],
            wor=pooled["paired_early_reward_test"]["worse"],
        )
    )
    return "\n".join(lines) + "\n"


def build_latex_table(report: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("\\begin{table}[t]")
    lines.append("\\centering")
    lines.append("\\caption{RWARE multi-setting generalization suite (critical expansion).}")
    lines.append("\\label{tab:rware-suite}")
    lines.append("\\small")
    lines.append("\\resizebox{\\columnwidth}{!}{%")
    lines.append("\\begin{tabular}{lcccc}")
    lines.append("\\toprule")
    lines.append("Environment & Random success & Heuristic success & $p$ (success) & $p$ (early@H) \\\\")
    lines.append("\\midrule")
    for row in report["environments"]:
        lines.append(
            "{env} & {rs:.3f} ({rsc}/{n}) & {hs:.3f} ({hsc}/{n}) & {sp:.6f} & {ep:.6f} \\\\".format(
                env=row["env_id"].replace("_", "\\_"),
                rs=row["random"]["success_rate"],
                rsc=row["random"]["success_count"],
                hs=row["heuristic"]["success_rate"],
                hsc=row["heuristic"]["success_count"],
                n=row["episodes"],
                sp=row["paired_success_test"]["p_value"],
                ep=row["paired_early_reward_test"]["p_value"],
            )
        )
    lines.append("\\midrule")
    pooled = report["pooled"]
    lines.append(
        "pooled & {rs:.3f} ({rsc}/{n}) & {hs:.3f} ({hsc}/{n}) & {sp:.6f} & {ep:.6f} \\\\".format(
            rs=pooled["random"]["success_rate"],
            rsc=pooled["random"]["success_count"],
            hs=pooled["heuristic"]["success_rate"],
            hsc=pooled["heuristic"]["success_count"],
            n=pooled["episodes_total"],
            sp=pooled["paired_success_test"]["p_value"],
            ep=pooled["paired_early_reward_test"]["p_value"],
        )
    )
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("}")
    lines.append("\\end{table}")
    lines.append("")
    return "\n".join(lines)


def compact_policy(row: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "episodes",
        "success_count",
        "success_rate",
        "success_ci_low",
        "success_ci_high",
        "mean_team_return",
        "mean_episode_steps",
        "mean_first_reward_step",
        "early_reward_horizon",
        "early_reward_count",
        "early_reward_rate",
        "early_reward_ci_low",
        "early_reward_ci_high",
    )
    return {k: row[k] for k in keys if k in row}


def main() -> None:
    args = parse_args()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    env_rows: list[dict[str, Any]] = []

    total_eps = 0
    random_success = 0
    heuristic_success = 0
    random_early = 0
    heuristic_early = 0

    success_improved = 0
    success_worse = 0
    early_improved = 0
    early_worse = 0

    for path_str in args.reports:
        path = Path(path_str).resolve()
        row = load_json(path)
        random_row = row["policies"]["random"]
        heuristic_row = row["policies"]["verifier_heuristic"]
        random_compact = compact_policy(random_row)
        heuristic_compact = compact_policy(heuristic_row)
        episodes = int(row["episodes"])

        env_rows.append(
            {
                "env_id": row["env_id"],
                "episodes": episodes,
                "random": random_compact,
                "heuristic": heuristic_compact,
                "paired_success_test": row["paired_success_test"],
                "paired_early_reward_test": row["paired_early_reward_test"],
            }
        )

        total_eps += episodes
        random_success += int(random_row["success_count"])
        heuristic_success += int(heuristic_row["success_count"])
        random_early += int(random_row["early_reward_count"])
        heuristic_early += int(heuristic_row["early_reward_count"])

        success_improved += int(row["paired_success_test"]["improved"])
        success_worse += int(row["paired_success_test"]["worse"])
        early_improved += int(row["paired_early_reward_test"]["improved"])
        early_worse += int(row["paired_early_reward_test"]["worse"])

    rs_low, rs_high = wilson_ci(random_success, total_eps)
    hs_low, hs_high = wilson_ci(heuristic_success, total_eps)
    re_low, re_high = wilson_ci(random_early, total_eps)
    he_low, he_high = wilson_ci(heuristic_early, total_eps)

    report = {
        "suite_name": "rware_critical_multi_setting",
        "environments": env_rows,
        "pooled": {
            "episodes_total": total_eps,
            "random": {
                "success_count": random_success,
                "success_rate": random_success / max(total_eps, 1),
                "success_ci_low": rs_low,
                "success_ci_high": rs_high,
                "early_reward_count": random_early,
                "early_reward_rate": random_early / max(total_eps, 1),
                "early_reward_ci_low": re_low,
                "early_reward_ci_high": re_high,
            },
            "heuristic": {
                "success_count": heuristic_success,
                "success_rate": heuristic_success / max(total_eps, 1),
                "success_ci_low": hs_low,
                "success_ci_high": hs_high,
                "early_reward_count": heuristic_early,
                "early_reward_rate": heuristic_early / max(total_eps, 1),
                "early_reward_ci_low": he_low,
                "early_reward_ci_high": he_high,
            },
            "paired_success_test": {
                "n": total_eps,
                "improved": success_improved,
                "worse": success_worse,
                "same": total_eps - (success_improved + success_worse),
                "p_value": mcnemar_exact_pvalue(success_improved, success_worse),
            },
            "paired_early_reward_test": {
                "n": total_eps,
                "improved": early_improved,
                "worse": early_worse,
                "same": total_eps - (early_improved + early_worse),
                "p_value": mcnemar_exact_pvalue(early_improved, early_worse),
            },
        },
    }

    out_json = out_dir / "rware_generalization_suite_report.json"
    out_md = out_dir / "rware_generalization_suite_report.md"
    out_tex = out_dir / "rware_generalization_suite_table.tex"

    out_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    out_md.write_text(build_markdown(report), encoding="utf-8")
    out_tex.write_text(build_latex_table(report), encoding="utf-8")

    print(f"[OK] Wrote {out_json}")
    print(f"[OK] Wrote {out_md}")
    print(f"[OK] Wrote {out_tex}")


if __name__ == "__main__":
    main()
