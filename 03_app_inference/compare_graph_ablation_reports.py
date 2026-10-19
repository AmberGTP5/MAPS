from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


GUIDANCE_MODES = ["none", "feasibility", "route_hint"]
PLANNING_MODES = ["greedy", "retry", "search"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare two graph ablation JSON reports.")
    parser.add_argument("--left-json", type=Path, required=True, help="Reference report JSON path.")
    parser.add_argument("--right-json", type=Path, required=True, help="Compared report JSON path.")
    parser.add_argument("--left-name", type=str, default="left", help="Label for reference run.")
    parser.add_argument("--right-name", type=str, default="right", help="Label for compared run.")
    parser.add_argument(
        "--output-md",
        type=Path,
        default=None,
        help="Output markdown path (default: <right-json-stem>_vs_<left-name>.md)",
    )
    return parser.parse_args()


def load_report(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def fmt(v: float) -> str:
    return f"{v:.3f}"


def build_markdown(
    *,
    left_name: str,
    right_name: str,
    left: dict[str, Any],
    right: dict[str, Any],
) -> str:
    lines: list[str] = []
    lines.append("# Graph Ablation Cross-Run Comparison")
    lines.append("")
    lines.append(f"- Left: `{left_name}` ({left.get('base_prefix', '')})")
    lines.append(f"- Right: `{right_name}` ({right.get('base_prefix', '')})")
    lines.append("")

    left_issues = left.get("integrity_issues", [])
    right_issues = right.get("integrity_issues", [])
    lines.append("## Integrity")
    lines.append("")
    lines.append(f"- Left issues: {len(left_issues)}")
    lines.append(f"- Right issues: {len(right_issues)}")
    lines.append("")

    lines.append("## Success Rate Delta (Right - Left)")
    lines.append("")
    lines.append("| Guidance | Mode | Left | Right | Delta |")
    lines.append("| --- | --- | ---: | ---: | ---: |")
    for guidance in GUIDANCE_MODES:
        for mode in PLANNING_MODES:
            l = float(left["aggregate"][guidance][mode]["success_rate"])
            r = float(right["aggregate"][guidance][mode]["success_rate"])
            lines.append(
                f"| {guidance} | {mode} | {fmt(l)} | {fmt(r)} | {fmt(r - l)} |"
            )
    lines.append("")

    lines.append("## Search Intervention Delta (Right - Left)")
    lines.append("")
    lines.append("| Guidance | Left Search Selected | Right Search Selected | Delta |")
    lines.append("| --- | ---: | ---: | ---: |")
    for guidance in GUIDANCE_MODES:
        l = float(left["aggregate"][guidance]["search"]["search_selected_rate"])
        r = float(right["aggregate"][guidance]["search"]["search_selected_rate"])
        lines.append(f"| {guidance} | {fmt(l)} | {fmt(r)} | {fmt(r - l)} |")
    lines.append("")

    lines.append("## Summary")
    lines.append("")
    for mode in ("greedy", "retry", "search"):
        l = float(left["aggregate"]["route_hint"][mode]["success_rate"])
        r = float(right["aggregate"]["route_hint"][mode]["success_rate"])
        lines.append(
            f"- route_hint/{mode}: `{left_name}={fmt(l)}`, `{right_name}={fmt(r)}`, delta=`{fmt(r - l)}`."
        )
    lines.append("")
    return "\n".join(lines) + "\n"


def main() -> int:
    args = parse_args()
    left = load_report(args.left_json.resolve())
    right = load_report(args.right_json.resolve())

    output_md = args.output_md
    if output_md is None:
        safe_left = "".join(ch if ch.isalnum() else "_" for ch in args.left_name).strip("_").lower()
        output_md = args.right_json.resolve().with_name(f"{args.right_json.stem}_vs_{safe_left}.md")

    content = build_markdown(
        left_name=args.left_name,
        right_name=args.right_name,
        left=left,
        right=right,
    )
    output_md.write_text(content, encoding="utf-8")
    print(f"Wrote comparison markdown: {output_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

