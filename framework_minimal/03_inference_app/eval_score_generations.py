from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from eval_utils import (
    DEFAULT_PARSE_SCHEMA_PENALTY,
    aggregate_by_group,
    aggregate_scored_rows,
    dump_jsonl,
    extract_task_context,
    load_jsonl,
    parse_response_text,
    render_report_markdown,
    summarize_plan_shape,
    validate_plan_schema,
)
from symbolic_validator import SymbolicValidator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score generated PVR plans with parser/schema/symbolic checks.")
    parser.add_argument("--input-file", required=True, help="Generated jsonl file or dataset jsonl.")
    parser.add_argument("--text-field", default="generated_text", help="Field containing model response text.")
    parser.add_argument(
        "--output-prefix",
        default="",
        help="Prefix for scored jsonl/json/md outputs. Defaults to input-file stem under the same directory.",
    )
    parser.add_argument(
        "--keep-response-text",
        action="store_true",
        help="Retain the raw response text in the scored per-sample jsonl.",
    )
    parser.add_argument(
        "--llm-second-pass",
        action="store_true",
        help="Enable LLM second-pass audit (applies equally to all models).",
    )
    parser.add_argument(
        "--llm-target",
        default="all_samples",
        choices=("all_samples", "parse_only", "parse_or_schema", "failed_samples"),
        help="Which rows trigger LLM second-pass audit.",
    )
    parser.add_argument(
        "--llm-max-rechecks",
        type=int,
        default=0,
        help="Max rows to recheck with LLM. 0 means no cap.",
    )
    parser.add_argument(
        "--llm-model",
        default=os.getenv("PVR_EVAL_LLM_MODEL", "google/gemini-2.5-flash"),
        help="LLM model id for second-pass.",
    )
    parser.add_argument(
        "--llm-api-key",
        default=os.getenv("OPENROUTER_API_KEY", ""),
        help="OpenRouter API key for second-pass.",
    )
    parser.add_argument(
        "--llm-base-url",
        default=os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
        help="OpenRouter-compatible base URL for second-pass.",
    )
    parser.add_argument("--llm-temperature", type=float, default=0.0)
    parser.add_argument("--llm-max-tokens", type=int, default=2048)
    return parser.parse_args()


def build_scored_row(
    row: Dict[str, Any],
    text_field: str,
    validator: SymbolicValidator,
) -> Dict[str, Any]:
    context = extract_task_context(row)
    response_text = row.get(text_field)

    parsed = parse_response_text(response_text)
    plan = parsed["plan"]
    plan_shape = summarize_plan_shape(plan)

    schema = validate_plan_schema(plan) if parsed["parse_ok"] else {"is_valid": False, "errors": []}
    validator_result: Dict[str, Any] = {}
    error_type: List[str] = []
    violation_breakdown: Dict[str, int] = {}

    if parsed["parse_ok"] and schema["is_valid"]:
        validator_result = validator.validate_plan(
            plan=plan,
            start_node=context["start_node"],
            end_node=context["end_node"],
            target_item=context["target_item"],
        )
        error_type = list(validator_result.get("error_type") or [])
        breakdown = validator_result.get("violation_breakdown") or {}
        if isinstance(breakdown, dict):
            violation_breakdown = {str(k): int(v) for k, v in breakdown.items()}

    executable_plan = bool(validator_result.get("is_valid")) if validator_result else False
    # Exact delivery must imply full symbolic executability, not just absence of
    # terminal_failure. Otherwise a plan can "end" at the right target while still
    # violating coordination/capability constraints mid-trajectory.
    exact_delivery = bool(
        parsed["parse_ok"]
        and schema["is_valid"]
        and executable_plan
        and "terminal_failure" not in error_type
    )
    penalty_score = (
        float(validator_result.get("penalty_score", 0.0) or 0.0)
        if validator_result
        else DEFAULT_PARSE_SCHEMA_PENALTY
    )

    scored_row = {
        "pair_id": context["pair_id"],
        "task_id": context["task_id"],
        "selected_robot": context["selected_robot"],
        "task_profile": context["task_profile"],
        "margin": context["margin"],
        "margin_label": context["margin_label"],
        "start_node": context["start_node"],
        "end_node": context["end_node"],
        "target_item": context["target_item"],
        "parse_ok": bool(parsed["parse_ok"]),
        "parse_error": parsed["parse_error"],
        "schema_pass": bool(schema["is_valid"]),
        "schema_errors": list(schema["errors"]),
        "executable_plan": executable_plan,
        "exact_delivery": exact_delivery,
        "rule_violation": not executable_plan,
        "penalty_score": penalty_score,
        "trajectory_cost": validator_result.get("trajectory_cost"),
        "step_count": plan_shape["step_count"],
        "handover_count": plan_shape["handover_count"],
        "single_robot": plan_shape["single_robot"],
        "robot_set": plan_shape["robot_set"],
        "error_type": error_type,
        "error_details": validator_result.get("error_details", ""),
        "failed_steps": list(validator_result.get("failed_steps") or []),
        "violation_breakdown": violation_breakdown,
        "validation_source": validator_result.get("validation_source", ""),
        "think": parsed["think"],
        "plan": plan,
    }

    return scored_row


def should_run_second_pass(scored_row: Dict[str, Any], llm_target: str) -> bool:
    if llm_target == "all_samples":
        return True
    if llm_target == "parse_only":
        return not scored_row.get("parse_ok", False)
    if llm_target == "failed_samples":
        return not scored_row.get("exact_delivery", False)
    if llm_target == "parse_or_schema":
        return (not scored_row.get("parse_ok", False)) or (
            scored_row.get("parse_ok", False) and not scored_row.get("schema_pass", False)
        )
    return (not scored_row.get("parse_ok", False)) or (
        scored_row.get("parse_ok", False) and not scored_row.get("schema_pass", False)
    )


def init_llm_client(args: argparse.Namespace):
    if not args.llm_second_pass:
        return None
    api_key = (args.llm_api_key or "").strip()
    if not api_key:
        raise RuntimeError("LLM second-pass is enabled but --llm-api-key / OPENROUTER_API_KEY is missing.")
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError("openai package is required for --llm-second-pass.") from exc
    return OpenAI(base_url=args.llm_base_url, api_key=api_key)


def build_llm_recheck_prompt(
    context: Dict[str, Any],
    response_text: Any,
    base_scored: Dict[str, Any],
    llm_target: str,
) -> str:
    text = response_text if isinstance(response_text, str) else ""
    base_failure = {
        "parse_ok": bool(base_scored.get("parse_ok")),
        "schema_pass": bool(base_scored.get("schema_pass")),
        "executable_plan": bool(base_scored.get("executable_plan")),
        "exact_delivery": bool(base_scored.get("exact_delivery")),
        "parse_error": base_scored.get("parse_error", ""),
        "schema_errors": base_scored.get("schema_errors", []),
        "error_type": base_scored.get("error_type", []),
        "error_details": base_scored.get("error_details", ""),
        "failed_steps": base_scored.get("failed_steps", []),
    }

    if llm_target == "all_samples":
        mode_hint = (
            "Audit every sample comprehensively: parse/schema integrity, reasoning consistency, "
            "goal reachability, and terminal delivery correctness."
        )
    elif llm_target == "failed_samples":
        mode_hint = (
            "Focus on full failure recheck: parse/schema, action ordering, goal reachability, and terminal delivery."
        )
    else:
        mode_hint = "Focus on parse/schema recheck while still reporting reasoning/goal/terminal verdict."

    return f"""You are an evaluator-side second-pass reviewer for robot plans.
{mode_hint}
Do not invent new tasks. Keep robots/actions within the original intent.

Task context:
- start_node: {context.get("start_node")}
- end_node: {context.get("end_node")}
- target_item: {context.get("target_item")}

Base validator failure snapshot:
{json.dumps(base_failure, ensure_ascii=False)}

Raw model output:
<raw_output>
{text}
</raw_output>

Return ONLY one JSON object:
{{
  "review": {{
    "llm_pass": "yes|no|uncertain",
    "reasoning_consistency": "good|weak|missing|contradictory",
    "goal_reachability": "yes|no|uncertain",
    "terminal_delivery": "yes|no|uncertain",
    "symbolic_risk": "low|medium|high",
    "notes": "short explanation"
  }}
}}
"""


def extract_json_object(text: str) -> Optional[Dict[str, Any]]:
    if not isinstance(text, str) or not text.strip():
        return None
    try:
        parsed = json.loads(text)
        return parsed if isinstance(parsed, dict) else None
    except json.JSONDecodeError:
        pass
    import re

    fenced = re.search(r"```json\s*(\{.*\})\s*```", text, re.DOTALL | re.IGNORECASE)
    if fenced:
        try:
            parsed = json.loads(fenced.group(1))
            return parsed if isinstance(parsed, dict) else None
        except json.JSONDecodeError:
            return None
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        candidate = text[start : end + 1]
        try:
            parsed = json.loads(candidate)
            return parsed if isinstance(parsed, dict) else None
        except json.JSONDecodeError:
            return None
    return None


def call_llm_recheck(
    *,
    client,
    args: argparse.Namespace,
    context: Dict[str, Any],
    response_text: Any,
    base_scored: Dict[str, Any],
) -> Tuple[Optional[Dict[str, Any]], str]:
    prompt = build_llm_recheck_prompt(context, response_text, base_scored, args.llm_target)
    try:
        response = client.chat.completions.create(
            model=args.llm_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=args.llm_temperature,
            max_tokens=args.llm_max_tokens,
        )
        content = response.choices[0].message.content
        if not isinstance(content, str) or not content.strip():
            return None, "empty_llm_response"
        parsed = extract_json_object(content)
        if not parsed:
            return None, "llm_json_parse_failed"
        return parsed, ""
    except Exception as exc:  # noqa: BLE001
        return None, f"{type(exc).__name__}: {exc}"


def main() -> None:
    args = parse_args()

    input_path = Path(args.input_file).resolve()
    prefix = args.output_prefix.strip()
    if prefix:
        output_base = Path(prefix).resolve()
    else:
        output_base = input_path.with_suffix("")

    rows = load_jsonl(input_path)
    validator = SymbolicValidator()
    llm_client = init_llm_client(args)

    base_rows: List[Dict[str, Any]] = []
    final_rows: List[Dict[str, Any]] = []
    llm_stats = {
        "enabled": bool(args.llm_second_pass),
        "attempted": 0,
        "returned_review": 0,
        "failed_calls": 0,
        "skipped_by_cap": 0,
        "llm_pass_yes": 0,
        "goal_reachability_yes": 0,
        "terminal_delivery_yes": 0,
        "symbolic_disagreement_count": 0,
    }
    total = len(rows)
    for index, row in enumerate(rows, start=1):
        base_row = build_scored_row(row, args.text_field, validator)
        base_rows.append(base_row)
        scored_row = dict(base_row)

        llm_meta: Dict[str, Any] = {"applied": False, "error": ""}
        can_recheck = should_run_second_pass(base_row, args.llm_target)
        max_rechecks = max(0, int(args.llm_max_rechecks))
        under_cap = (max_rechecks == 0) or (llm_stats["attempted"] < max_rechecks)

        if llm_client is not None and can_recheck:
            if not under_cap:
                llm_stats["skipped_by_cap"] += 1
            else:
                llm_stats["attempted"] += 1
                llm_meta["applied"] = True
                context = extract_task_context(row)
                review_payload, err = call_llm_recheck(
                    client=llm_client,
                    args=args,
                    context=context,
                    response_text=row.get(args.text_field),
                    base_scored=base_row,
                )
                if review_payload is None:
                    llm_stats["failed_calls"] += 1
                    llm_meta["error"] = err or "unknown_llm_error"
                else:
                    llm_stats["returned_review"] += 1
                    review = review_payload.get("review")
                    if isinstance(review, dict):
                        llm_meta["review"] = review
                        llm_pass = str(review.get("llm_pass", "")).lower()
                        goal = str(review.get("goal_reachability", "")).lower()
                        terminal = str(review.get("terminal_delivery", "")).lower()
                        if llm_pass == "yes":
                            llm_stats["llm_pass_yes"] += 1
                        if goal == "yes":
                            llm_stats["goal_reachability_yes"] += 1
                        if terminal == "yes":
                            llm_stats["terminal_delivery_yes"] += 1
                        if terminal == "yes" and not bool(base_row.get("exact_delivery")):
                            llm_stats["symbolic_disagreement_count"] += 1
                    else:
                        llm_meta["error"] = "llm_response_missing_review_object"

        if args.llm_second_pass:
            scored_row["llm_second_pass"] = llm_meta

        if args.keep_response_text:
            scored_row["response_text"] = row.get(args.text_field)
        final_rows.append(scored_row)
        key = scored_row.get("pair_id") or scored_row.get("task_id") or f"row_{index}"
        print(f"[scored {index}/{total}] {key}")

    base_aggregate = aggregate_scored_rows(base_rows)
    final_aggregate = aggregate_scored_rows(final_rows)
    report = {
        "input_file": str(input_path),
        "text_field": args.text_field,
        "overall": base_aggregate["overall"],
        "failure_breakdown": base_aggregate["failure_breakdown"],
        "violation_breakdown": base_aggregate["violation_breakdown"],
        "by_robot": aggregate_by_group(base_rows, "selected_robot"),
        "by_task_profile": aggregate_by_group(base_rows, "task_profile"),
        "by_margin": aggregate_by_group(base_rows, "margin_label"),
    }
    if args.llm_second_pass:
        report["llm_second_pass"] = {
            "mode": "check_only_no_score_override",
            "config": {
                "model": args.llm_model,
                "target": args.llm_target,
                "max_rechecks": args.llm_max_rechecks,
            },
            "stats": llm_stats,
            "overall_reference_same_as_primary": final_aggregate["overall"],
            "failure_breakdown_reference_same_as_primary": final_aggregate["failure_breakdown"],
        }

    scored_path = output_base.with_name(output_base.name + "_scored.jsonl")
    scored_llm_path = output_base.with_name(output_base.name + "_scored_llm_second_pass.jsonl")
    report_json_path = output_base.with_name(output_base.name + "_report.json")
    report_md_path = output_base.with_name(output_base.name + "_report.md")

    dump_jsonl(scored_path, base_rows)
    if args.llm_second_pass:
        dump_jsonl(scored_llm_path, final_rows)
    report_json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    report_md_path.write_text(render_report_markdown(report), encoding="utf-8")

    print(json.dumps(report["overall"], ensure_ascii=False, indent=2))
    if args.llm_second_pass:
        print(json.dumps(report["llm_second_pass"]["stats"], ensure_ascii=False, indent=2))
    print(f"[saved] {scored_path}")
    if args.llm_second_pass:
        print(f"[saved] {scored_llm_path}")
    print(f"[saved] {report_json_path}")
    print(f"[saved] {report_md_path}")


if __name__ == "__main__":
    main()
