from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


MODES = ("greedy", "retry", "search")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare two app benchmark runs.")
    parser.add_argument("--left-summary", type=Path, required=True, help="Left run benchmark_summary.json")
    parser.add_argument("--right-summary", type=Path, required=True, help="Right run benchmark_summary.json")
    parser.add_argument("--left-name", type=str, default="left")
    parser.add_argument("--right-name", type=str, default="right")
    parser.add_argument(
        "--output-md",
        type=Path,
        default=None,
        help="Output markdown path (default: <right-summary-stem>_vs_<left-name>.md)",
    )
    return parser.parse_args()


def load_summary(path: Path) -> dict[str, dict[str, Any]]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    return {str(row["mode"]): row for row in rows}


def _fmt(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def _delta(left: Any, right: Any) -> str:
    if left is None or right is None:
        return "-"
    try:
        return f"{(float(right) - float(left)):.3f}"
    except Exception:
        return "-"


def build_markdown(
    *,
    left_name: str,
    right_name: str,
    left: dict[str, dict[str, Any]],
    right: dict[str, dict[str, Any]],
) -> str:
    lines: list[str] = []
    lines.append("# App Benchmark Comparison")
    lines.append("")
    lines.append(f"- Left: `{left_name}`")
    lines.append(f"- Right: `{right_name}`")
    lines.append("")
    lines.append("## Aggregate Delta (Right - Left)")
    lines.append("")
    lines.append(
        "| Mode | Metric | Left | Right | Delta |"
    )
    lines.append("| --- | --- | ---: | ---: | ---: |")

    metrics = (
        "success_rate",
        "planner_valid_rate",
        "selected_valid_rate",
        "source_match_rate",
        "handover_match_rate",
        "avg_plan_length",
        "avg_runtime_sec",
        "search_selected_rate",
    )

    for mode in MODES:
        lrow = left.get(mode, {})
        rrow = right.get(mode, {})
        for metric in metrics:
            lval = lrow.get(metric)
            rval = rrow.get(metric)
            lines.append(
                f"| {mode} | {metric} | {_fmt(lval)} | {_fmt(rval)} | {_delta(lval, rval)} |"
            )

    lines.append("")
    return "\n".join(lines) + "\n"


def main() -> int:
    args = parse_args()
    left_summary = load_summary(args.left_summary.resolve())
    right_summary = load_summary(args.right_summary.resolve())

    output_md = args.output_md
    if output_md is None:
        safe_left = "".join(ch if ch.isalnum() else "_" for ch in args.left_name).strip("_").lower()
        output_md = args.right_summary.resolve().with_name(
            f"{args.right_summary.stem}_vs_{safe_left}.md"
        )

    content = build_markdown(
        left_name=args.left_name,
        right_name=args.right_name,
        left=left_summary,
        right=right_summary,
    )
    output_md.write_text(content, encoding="utf-8")
    print(f"Wrote comparison markdown: {output_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

