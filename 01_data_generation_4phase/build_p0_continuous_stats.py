from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import stats


@dataclass(frozen=True)
class ComparisonSpec:
    dataset_name: str
    csv_path: Path
    left_mode: str
    right_mode: str
    metrics: Tuple[str, ...]
    metric_direction: Dict[str, str]  # "higher" or "lower"


def _safe_float(value: object) -> Optional[float]:
    if value is None:
        return None
    text = str(value).strip()
    if text == "":
        return None
    try:
        return float(text)
    except ValueError:
        return None


def load_rows_by_mode(csv_path: Path) -> Dict[str, Dict[str, Dict[str, str]]]:
    by_mode: Dict[str, Dict[str, Dict[str, str]]] = {}
    with csv_path.open("r", encoding="utf-8-sig", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            mode = str(row.get("mode", "")).strip()
            task_id = str(row.get("task_id", "")).strip()
            if not mode or not task_id:
                continue
            by_mode.setdefault(mode, {})[task_id] = row
    return by_mode


def paired_continuous_test(
    left_vals: List[float], right_vals: List[float]
) -> Dict[str, Optional[float]]:
    left = np.array(left_vals, dtype=float)
    right = np.array(right_vals, dtype=float)
    if left.size == 0 or right.size == 0 or left.size != right.size:
        return {
            "n": 0,
            "mean_left": None,
            "mean_right": None,
            "mean_diff_right_minus_left": None,
            "p_value": None,
            "ci95_low": None,
            "ci95_high": None,
            "cohens_d_paired": None,
        }

    diff = right - left
    n = int(diff.size)
    mean_left = float(np.mean(left))
    mean_right = float(np.mean(right))
    mean_diff = float(np.mean(diff))

    if n < 2:
        return {
            "n": n,
            "mean_left": mean_left,
            "mean_right": mean_right,
            "mean_diff_right_minus_left": mean_diff,
            "p_value": None,
            "ci95_low": None,
            "ci95_high": None,
            "cohens_d_paired": None,
        }

    test = stats.ttest_rel(right, left, nan_policy="omit")
    p_value = float(test.pvalue) if test.pvalue is not None else None
    if p_value is not None and not np.isfinite(p_value):
        p_value = 1.0 if abs(mean_diff) < 1e-12 else 0.0

    sd_diff = float(np.std(diff, ddof=1))
    se = sd_diff / np.sqrt(n) if sd_diff > 0 else 0.0
    if se == 0.0:
        ci_low = mean_diff
        ci_high = mean_diff
        d = 0.0
    else:
        t_crit = float(stats.t.ppf(0.975, df=n - 1))
        margin = t_crit * se
        ci_low = mean_diff - margin
        ci_high = mean_diff + margin
        d = mean_diff / sd_diff

    return {
        "n": n,
        "mean_left": mean_left,
        "mean_right": mean_right,
        "mean_diff_right_minus_left": mean_diff,
        "p_value": p_value,
        "ci95_low": float(ci_low),
        "ci95_high": float(ci_high),
        "cohens_d_paired": float(d),
    }


def compute_for_spec(spec: ComparisonSpec) -> List[Dict[str, object]]:
    by_mode = load_rows_by_mode(spec.csv_path)
    left_rows = by_mode.get(spec.left_mode, {})
    right_rows = by_mode.get(spec.right_mode, {})
    shared_tasks = sorted(set(left_rows.keys()) & set(right_rows.keys()))

    out: List[Dict[str, object]] = []
    for metric in spec.metrics:
        left_vals: List[float] = []
        right_vals: List[float] = []
        for task_id in shared_tasks:
            lv = _safe_float(left_rows[task_id].get(metric))
            rv = _safe_float(right_rows[task_id].get(metric))
            if lv is None or rv is None:
                continue
            left_vals.append(lv)
            right_vals.append(rv)

        stats_row = paired_continuous_test(left_vals, right_vals)
        direction = spec.metric_direction.get(metric, "higher")
        mean_diff = stats_row["mean_diff_right_minus_left"]
        better_flag = None
        if mean_diff is not None:
            if direction == "higher":
                better_flag = bool(mean_diff > 0)
            elif direction == "lower":
                better_flag = bool(mean_diff < 0)

        out.append(
            {
                "dataset": spec.dataset_name,
                "csv_path": str(spec.csv_path),
                "left_mode": spec.left_mode,
                "right_mode": spec.right_mode,
                "metric": metric,
                "direction": direction,
                "right_better_by_direction": better_flag,
                **stats_row,
            }
        )
    return out


def render_markdown(rows: List[Dict[str, object]]) -> str:
    header = (
        "# Continuous-Metric Significance (Paired t-test)\n\n"
        "All tests are paired on matched `task_id` within the same dataset.\n"
        "Columns:\n"
        "- `mean_diff`: right - left\n"
        "- `CI95`: 95% confidence interval for mean diff\n"
        "- `d`: paired Cohen's d\n\n"
        "| Dataset | Comparison (right vs left) | Metric | n | mean_left | mean_right | "
        "mean_diff | CI95 | p-value | d | Direction | Right Better |\n"
        "| --- | --- | --- | ---: | ---: | ---: | ---: | --- | ---: | ---: | --- | --- |\n"
    )
    lines = [header]
    for r in rows:
        ci_low = r["ci95_low"]
        ci_high = r["ci95_high"]
        ci = "-"
        if ci_low is not None and ci_high is not None:
            ci = f"[{ci_low:.4f}, {ci_high:.4f}]"
        p = r["p_value"]
        p_text = "-" if p is None else f"{p:.6g}"
        d = r["cohens_d_paired"]
        d_text = "-" if d is None else f"{d:.4f}"
        rb = r["right_better_by_direction"]
        rb_text = "-" if rb is None else ("yes" if rb else "no")
        lines.append(
            "| "
            f"{r['dataset']} | "
            f"{r['right_mode']} vs {r['left_mode']} | "
            f"{r['metric']} | "
            f"{r['n']} | "
            f"{r['mean_left']:.4f} | "
            f"{r['mean_right']:.4f} | "
            f"{r['mean_diff_right_minus_left']:.4f} | "
            f"{ci} | "
            f"{p_text} | "
            f"{d_text} | "
            f"{r['direction']} | "
            f"{rb_text} |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    data_dir = Path(__file__).resolve().parent
    paper_dir = data_dir / "paper_artifacts"
    paper_dir.mkdir(parents=True, exist_ok=True)

    specs = [
        ComparisonSpec(
            dataset_name="graph_none_30",
            csv_path=data_dir / "graph_benchmark_none" / "benchmark_results.csv",
            left_mode="greedy",
            right_mode="search",
            metrics=("runtime_sec", "selected_step_count", "selected_trajectory_cost"),
            metric_direction={
                "runtime_sec": "lower",
                "selected_step_count": "lower",
                "selected_trajectory_cost": "lower",
            },
        ),
        ComparisonSpec(
            dataset_name="graph_route_hint_30",
            csv_path=data_dir
            / "graph_benchmark_clean_google_gemini_2_5_flash_20260326_182800_route_hint"
            / "benchmark_results.csv",
            left_mode="greedy",
            right_mode="search",
            metrics=("runtime_sec", "selected_step_count", "selected_trajectory_cost"),
            metric_direction={
                "runtime_sec": "lower",
                "selected_step_count": "lower",
                "selected_trajectory_cost": "lower",
            },
        ),
        ComparisonSpec(
            dataset_name="app8_gemini",
            csv_path=data_dir / "app_benchmark_results" / "benchmark_results.csv",
            left_mode="greedy",
            right_mode="search",
            metrics=("runtime_sec", "executed_plan_length", "handover_count"),
            metric_direction={
                "runtime_sec": "lower",
                "executed_plan_length": "lower",
                "handover_count": "lower",
            },
        ),
        ComparisonSpec(
            dataset_name="app40_gemini",
            csv_path=data_dir / "app_benchmark_results_random40" / "benchmark_results.csv",
            left_mode="greedy",
            right_mode="search",
            metrics=("runtime_sec", "executed_plan_length", "handover_count"),
            metric_direction={
                "runtime_sec": "lower",
                "executed_plan_length": "lower",
                "handover_count": "lower",
            },
        ),
    ]

    all_rows: List[Dict[str, object]] = []
    for spec in specs:
        if not spec.csv_path.exists():
            continue
        all_rows.extend(compute_for_spec(spec))

    out_json = paper_dir / "significance_continuous.json"
    out_md = paper_dir / "significance_continuous.md"
    out_json.write_text(json.dumps({"tests": all_rows}, ensure_ascii=False, indent=2), encoding="utf-8")
    out_md.write_text(render_markdown(all_rows), encoding="utf-8")

    print(f"[saved] {out_json}")
    print(f"[saved] {out_md}")
    print(f"[rows] {len(all_rows)}")


if __name__ == "__main__":
    main()
