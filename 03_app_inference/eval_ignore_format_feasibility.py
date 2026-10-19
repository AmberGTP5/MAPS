from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List


PICK_PATTERNS = [
    re.compile(r"\bpick[_\s-]?up\b", re.IGNORECASE),
]

PUT_PATTERNS = [
    re.compile(r"\bput[_\s-]?down\b", re.IGNORECASE),
    re.compile(r"\bdrop[_\s-]?off\b", re.IGNORECASE),
]

NO_SOLUTION_PATTERNS = [
    re.compile(r"\bno[_\s-]?solution\b", re.IGNORECASE),
    re.compile(r"\bno\s+feasible\s+plan\b", re.IGNORECASE),
    re.compile(r"\bunsat\b", re.IGNORECASE),
    re.compile(r"\bcannot\s+deliver\b", re.IGNORECASE),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate task feasibility from generated text while ignoring strict output format."
    )
    parser.add_argument("--input-file", required=True, help="jsonl file from eval_generate_plans.py")
    parser.add_argument("--output-prefix", required=True, help="prefix for json/md reports")
    parser.add_argument(
        "--window",
        type=int,
        default=240,
        help="Character window for near-node checks around action tokens.",
    )
    return parser.parse_args()


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def find_near(text: str, patterns: List[re.Pattern[str]], node: str, window: int) -> bool:
    if not text or not node:
        return False
    for pat in patterns:
        for match in pat.finditer(text):
            start = max(0, match.start() - window)
            end = min(len(text), match.end() + window)
            if node in text[start:end]:
                return True
    return False


def contains_any(text: str, patterns: List[re.Pattern[str]]) -> bool:
    return any(p.search(text) for p in patterns)


def score_row(row: Dict[str, Any], window: int) -> Dict[str, Any]:
    text = str(row.get("generated_text") or "")
    start = str(row.get("start_node") or "")
    end = str(row.get("end_node") or "")
    item = str(row.get("target_item") or "")

    has_pick = contains_any(text, PICK_PATTERNS)
    has_put = contains_any(text, PUT_PATTERNS)
    pick_near_start = find_near(text, PICK_PATTERNS, start, window)
    put_near_end = find_near(text, PUT_PATTERNS, end, window)
    has_no_solution = contains_any(text, NO_SOLUTION_PATTERNS)
    mentions_start = start in text if start else False
    mentions_end = end in text if end else False
    mentions_item = item in text if item else False

    # Conservative: must have both action anchors and no no-solution declaration.
    strict_feasible = bool(has_pick and has_put and pick_near_start and put_near_end and not has_no_solution)

    # Looser: action tokens + mentions start/end/item, still excluding explicit no-solution.
    lenient_feasible = bool(
        has_pick and has_put and mentions_start and mentions_end and mentions_item and not has_no_solution
    )

    if strict_feasible:
        verdict = "feasible_strict"
    elif lenient_feasible:
        verdict = "feasible_lenient_only"
    else:
        verdict = "not_feasible_logic"

    return {
        "pair_id": row.get("pair_id"),
        "task_id": row.get("task_id"),
        "target_item": row.get("target_item"),
        "start_node": row.get("start_node"),
        "end_node": row.get("end_node"),
        "strict_feasible": strict_feasible,
        "lenient_feasible": lenient_feasible,
        "verdict": verdict,
        "signals": {
            "has_pick": has_pick,
            "has_put": has_put,
            "pick_near_start": pick_near_start,
            "put_near_end": put_near_end,
            "has_no_solution": has_no_solution,
            "mentions_start": mentions_start,
            "mentions_end": mentions_end,
            "mentions_item": mentions_item,
        },
        "evidence": {
            "head": text[:360],
            "tail": text[-360:],
        },
    }


def render_md(rows: List[Dict[str, Any]], summary: Dict[str, Any]) -> str:
    lines = [
        "# Ignore-Format Feasibility Audit",
        "",
        "## Summary",
        "",
        f"- Total: `{summary['count']}`",
        f"- Strict feasible: `{summary['strict_feasible_count']}`",
        f"- Lenient feasible: `{summary['lenient_feasible_count']}`",
        f"- Not feasible logic: `{summary['not_feasible_count']}`",
        "",
        "## Per-Sample",
        "",
        "| Pair ID | Task | Verdict | strict | lenient | no_solution | pick@start | put@end |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        signals = row["signals"]
        task = f"{row['target_item']} ({row['start_node']} -> {row['end_node']})"
        lines.append(
            f"| {row['pair_id']} | {task} | {row['verdict']} | "
            f"{row['strict_feasible']} | {row['lenient_feasible']} | {signals['has_no_solution']} | "
            f"{signals['pick_near_start']} | {signals['put_near_end']} |"
        )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    input_path = Path(args.input_file).resolve()
    output_prefix = Path(args.output_prefix).resolve()
    output_prefix.parent.mkdir(parents=True, exist_ok=True)

    rows = load_jsonl(input_path)
    scored = [score_row(row, args.window) for row in rows]

    strict_count = sum(1 for row in scored if row["strict_feasible"])
    lenient_count = sum(1 for row in scored if row["lenient_feasible"])
    not_feasible_count = len(scored) - strict_count

    summary = {
        "count": len(scored),
        "strict_feasible_count": strict_count,
        "lenient_feasible_count": lenient_count,
        "not_feasible_count": not_feasible_count,
    }

    with output_prefix.with_name(output_prefix.name + "_scored.jsonl").open("w", encoding="utf-8") as f:
        for row in scored:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    output_prefix.with_name(output_prefix.name + "_report.json").write_text(
        json.dumps({"summary": summary, "rows": scored}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    output_prefix.with_name(output_prefix.name + "_report.md").write_text(
        render_md(scored, summary),
        encoding="utf-8",
    )

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"[saved] {output_prefix.with_name(output_prefix.name + '_scored.jsonl')}")
    print(f"[saved] {output_prefix.with_name(output_prefix.name + '_report.json')}")
    print(f"[saved] {output_prefix.with_name(output_prefix.name + '_report.md')}")


if __name__ == "__main__":
    main()
