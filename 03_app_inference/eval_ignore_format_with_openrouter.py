from __future__ import annotations

import argparse
import json
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="LLM-based audit: ignore output format, judge task logic feasibility."
    )
    parser.add_argument("--input-file", required=True, help="Generated jsonl (from eval_generate_plans.py).")
    parser.add_argument("--output-prefix", required=True, help="Prefix of output files.")
    parser.add_argument("--model", default="google/gemini-2.5-pro")
    parser.add_argument("--base-url", default=os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"))
    parser.add_argument("--api-key", default=os.getenv("OPENROUTER_API_KEY", ""))
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--max-samples", type=int, default=0, help="0 means all rows.")
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--retry-sleep", type=float, default=2.0)
    return parser.parse_args()


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    return rows


def dump_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def extract_json_object(text: str) -> Optional[Dict[str, Any]]:
    if not isinstance(text, str) or not text.strip():
        return None
    text = text.strip()

    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

    fenced = re.search(r"```json\s*(\{.*\})\s*```", text, re.IGNORECASE | re.DOTALL)
    if fenced:
        try:
            obj = json.loads(fenced.group(1))
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass

    left = text.find("{")
    right = text.rfind("}")
    if left >= 0 and right > left:
        candidate = text[left : right + 1]
        try:
            obj = json.loads(candidate)
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass
    return None


def build_prompt(row: Dict[str, Any]) -> str:
    start = row.get("start_node")
    end = row.get("end_node")
    item = row.get("target_item")
    text = row.get("generated_text") or ""
    pair_id = row.get("pair_id")
    task_id = row.get("task_id")

    return f"""You are a strict evaluator for robot delivery planning.

Important:
- Ignore JSON/XML/format errors.
- Infer the intended plan logic from raw text.
- Judge whether the plan logic can complete the task.
- In this benchmark, tasks are intended to be feasible.

Task:
- pair_id: {pair_id}
- task_id: {task_id}
- start_node: {start}
- end_node: {end}
- target_item: {item}

Raw model output:
<raw_output>
{text}
</raw_output>

Return ONLY one JSON object in this exact schema:
{{
  "can_complete_task": "yes|no|uncertain",
  "pickup_at_start": "yes|no|uncertain",
  "delivery_at_end": "yes|no|uncertain",
  "declares_no_solution": "yes|no",
  "has_internal_contradiction": "yes|no",
  "major_errors": ["..."],
  "reason": "short reason"
}}
"""


def call_openrouter(client: Any, args: argparse.Namespace, prompt: str) -> Dict[str, Any]:
    response = client.chat.completions.create(
        model=args.model,
        messages=[{"role": "user", "content": prompt}],
        temperature=args.temperature,
        max_tokens=args.max_tokens,
    )
    content = response.choices[0].message.content
    if not isinstance(content, str) or not content.strip():
        raise RuntimeError("empty_llm_response")
    obj = extract_json_object(content)
    if not obj:
        raise RuntimeError("llm_json_parse_failed")
    return obj


def main() -> None:
    args = parse_args()
    if not args.api_key.strip():
        raise RuntimeError("Missing OPENROUTER_API_KEY (or --api-key).")

    try:
        from openai import OpenAI
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError("Please install openai package: pip install openai") from exc

    client = OpenAI(base_url=args.base_url, api_key=args.api_key.strip())

    input_path = Path(args.input_file).resolve()
    out_base = Path(args.output_prefix).resolve()
    out_base.parent.mkdir(parents=True, exist_ok=True)

    rows = load_jsonl(input_path)
    if args.max_samples > 0:
        rows = rows[: args.max_samples]

    scored_rows: List[Dict[str, Any]] = []
    summary = {
        "count": len(rows),
        "audited": 0,
        "can_complete_yes": 0,
        "can_complete_no": 0,
        "can_complete_uncertain": 0,
        "pickup_at_start_yes": 0,
        "delivery_at_end_yes": 0,
        "declares_no_solution_yes": 0,
        "has_internal_contradiction_yes": 0,
        "llm_errors": 0,
    }

    for idx, row in enumerate(rows, start=1):
        prompt = build_prompt(row)
        review = None
        err = ""
        for attempt in range(1, args.max_retries + 1):
            try:
                review = call_openrouter(client, args, prompt)
                break
            except Exception as exc:  # noqa: BLE001
                err = f"{type(exc).__name__}: {exc}"
                if attempt < args.max_retries:
                    time.sleep(args.retry_sleep * attempt)

        out_row = {
            "pair_id": row.get("pair_id"),
            "task_id": row.get("task_id"),
            "target_item": row.get("target_item"),
            "start_node": row.get("start_node"),
            "end_node": row.get("end_node"),
            "llm_review": review,
            "llm_error": err,
        }
        scored_rows.append(out_row)

        if review is None:
            summary["llm_errors"] += 1
        else:
            summary["audited"] += 1
            cc = str(review.get("can_complete_task", "")).lower()
            if cc == "yes":
                summary["can_complete_yes"] += 1
            elif cc == "no":
                summary["can_complete_no"] += 1
            else:
                summary["can_complete_uncertain"] += 1

            if str(review.get("pickup_at_start", "")).lower() == "yes":
                summary["pickup_at_start_yes"] += 1
            if str(review.get("delivery_at_end", "")).lower() == "yes":
                summary["delivery_at_end_yes"] += 1
            if str(review.get("declares_no_solution", "")).lower() == "yes":
                summary["declares_no_solution_yes"] += 1
            if str(review.get("has_internal_contradiction", "")).lower() == "yes":
                summary["has_internal_contradiction_yes"] += 1

        key = row.get("pair_id") or row.get("task_id") or f"row_{idx}"
        print(f"[audited {idx}/{len(rows)}] {key}")

    scored_path = out_base.with_name(out_base.name + "_scored.jsonl")
    report_path = out_base.with_name(out_base.name + "_report.json")
    report_md_path = out_base.with_name(out_base.name + "_report.md")

    dump_jsonl(scored_path, scored_rows)
    report = {"summary": summary}
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    md = [
        "# OpenRouter LLM Logic Audit (Ignore Format)",
        "",
        f"- Model: `{args.model}`",
        f"- Count: `{summary['count']}`",
        f"- Audited: `{summary['audited']}`",
        f"- can_complete=yes: `{summary['can_complete_yes']}`",
        f"- can_complete=no: `{summary['can_complete_no']}`",
        f"- can_complete=uncertain: `{summary['can_complete_uncertain']}`",
        f"- pickup_at_start=yes: `{summary['pickup_at_start_yes']}`",
        f"- delivery_at_end=yes: `{summary['delivery_at_end_yes']}`",
        f"- declares_no_solution=yes: `{summary['declares_no_solution_yes']}`",
        f"- has_internal_contradiction=yes: `{summary['has_internal_contradiction_yes']}`",
        f"- llm_errors: `{summary['llm_errors']}`",
        "",
    ]
    report_md_path.write_text("\n".join(md), encoding="utf-8")

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"[saved] {scored_path}")
    print(f"[saved] {report_path}")
    print(f"[saved] {report_md_path}")


if __name__ == "__main__":
    main()
