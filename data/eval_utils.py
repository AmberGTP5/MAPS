from __future__ import annotations

import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from planning_schema import ACTION_SPECS, AGENT_PROFILES


DEFAULT_PARSE_SCHEMA_PENALTY = 5.0
CANONICAL_VIOLATION_TYPES = (
    "capability_mismatch",
    "morphology_constraint",
    "graph_infeasible",
    "coordination_failure",
    "resource_conflict",
    "terminal_failure",
)


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def dump_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def safe_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        if math.isnan(value) or math.isinf(value):
            return None
        return float(value)
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            number = float(text)
        except ValueError:
            return None
        if math.isnan(number) or math.isinf(number):
            return None
        return number
    return None


def canonical_margin_label(value: Any) -> str:
    margin = safe_float(value)
    if margin is None:
        return "unknown"
    rounded = round(margin, 3)
    if rounded.is_integer():
        return str(int(rounded))
    return f"{rounded:.3f}".rstrip("0").rstrip(".")


def extract_tagged_block(text: str, tag: str) -> str:
    if not isinstance(text, str):
        return ""
    match = re.search(rf"<{tag}>(.*?)</{tag}>", text, re.DOTALL | re.IGNORECASE)
    return match.group(1).strip() if match else ""


def extract_json_fence(text: str) -> str:
    if not isinstance(text, str):
        return ""
    fenced = re.search(r"```json\s*(.*?)\s*```", text, re.DOTALL | re.IGNORECASE)
    return fenced.group(1).strip() if fenced else ""


def find_json_array(text: str) -> Optional[str]:
    if not isinstance(text, str) or not text:
        return None

    starts = [idx for idx, ch in enumerate(text) if ch == "["]
    for start in starts:
        depth = 0
        in_string = False
        escape = False
        for idx in range(start, len(text)):
            ch = text[idx]
            if in_string:
                if escape:
                    escape = False
                elif ch == "\\":
                    escape = True
                elif ch == '"':
                    in_string = False
                continue

            if ch == '"':
                in_string = True
            elif ch == "[":
                depth += 1
            elif ch == "]":
                depth -= 1
                if depth == 0:
                    candidate = text[start : idx + 1].strip()
                    try:
                        parsed = json.loads(candidate)
                    except json.JSONDecodeError:
                        continue
                    if isinstance(parsed, list):
                        return candidate
    return None


def find_json_arrays(text: str) -> List[str]:
    if not isinstance(text, str) or not text:
        return []

    arrays: List[str] = []
    starts = [idx for idx, ch in enumerate(text) if ch == "["]
    for start in starts:
        depth = 0
        in_string = False
        escape = False
        for idx in range(start, len(text)):
            ch = text[idx]
            if in_string:
                if escape:
                    escape = False
                elif ch == "\\":
                    escape = True
                elif ch == '"':
                    in_string = False
                continue

            if ch == '"':
                in_string = True
            elif ch == "[":
                depth += 1
            elif ch == "]":
                depth -= 1
                if depth == 0:
                    candidate = text[start : idx + 1].strip()
                    try:
                        parsed = json.loads(candidate)
                    except json.JSONDecodeError:
                        continue
                    if isinstance(parsed, list):
                        arrays.append(candidate)
                    break
    return arrays


def score_plan_candidate(plan: Any) -> float:
    if not isinstance(plan, list):
        return float("-inf")
    if not plan:
        return -50.0

    dict_steps = 0
    required_hits = 0
    for step in plan:
        if not isinstance(step, dict):
            continue
        dict_steps += 1
        required_hits += sum(1 for field in ("id", "robot", "action", "args", "depends_on") if field in step)

    if dict_steps == 0:
        # Avoid picking incidental arrays like depends_on: []
        return -20.0 - min(float(len(plan)), 10.0)

    field_density = required_hits / max(1.0, dict_steps * 5.0)
    return dict_steps * 10.0 + field_density * 5.0 + min(float(len(plan)), 100.0) * 0.1


def parse_response_text(response_text: Any) -> Dict[str, Any]:
    if not isinstance(response_text, str) or not response_text.strip():
        return {
            "parse_ok": False,
            "parse_error": "missing_or_non_string_response",
            "think": "",
            "answer_block": "",
            "plan": None,
        }

    think = extract_tagged_block(response_text, "think")
    answer_block = extract_tagged_block(response_text, "answer")

    candidates = []
    if answer_block:
        candidates.append(answer_block)
        fenced_answer = extract_json_fence(answer_block)
        if fenced_answer:
            candidates.append(fenced_answer)

    fenced_response = extract_json_fence(response_text)
    if fenced_response:
        candidates.append(fenced_response)

    candidates.append(response_text)

    best_plan: Any = None
    best_score = float("-inf")
    for candidate in candidates:
        for json_array in find_json_arrays(candidate):
            try:
                plan = json.loads(json_array)
            except json.JSONDecodeError:
                continue
            score = score_plan_candidate(plan)
            if score > best_score:
                best_score = score
                best_plan = plan

    if isinstance(best_plan, list):
        return {
            "parse_ok": True,
            "parse_error": "",
            "think": think,
            "answer_block": answer_block,
            "plan": best_plan,
        }

    return {
        "parse_ok": False,
        "parse_error": "json_array_not_found",
        "think": think,
        "answer_block": answer_block,
        "plan": None,
    }


def validate_plan_schema(plan: Any) -> Dict[str, Any]:
    errors: List[str] = []

    if not isinstance(plan, list):
        return {"is_valid": False, "errors": ["plan_is_not_a_list"]}
    if not plan:
        return {"is_valid": False, "errors": ["plan_is_empty"]}

    for index, step in enumerate(plan, start=1):
        prefix = f"step_{index}"
        if not isinstance(step, dict):
            errors.append(f"{prefix}_not_an_object")
            continue

        required_fields = ("id", "robot", "action", "args", "depends_on")
        for field_name in required_fields:
            if field_name not in step:
                errors.append(f"{prefix}_missing_{field_name}")

        step_id = step.get("id")
        if "id" in step and (not isinstance(step_id, str) or not step_id.strip()):
            errors.append(f"{prefix}_invalid_id")

        robot = step.get("robot")
        if "robot" in step and (not isinstance(robot, str) or robot not in AGENT_PROFILES):
            errors.append(f"{prefix}_invalid_robot")

        action = step.get("action")
        if "action" in step and (not isinstance(action, str) or action not in ACTION_SPECS):
            errors.append(f"{prefix}_invalid_action")

        args = step.get("args")
        if "args" in step and not isinstance(args, dict):
            errors.append(f"{prefix}_invalid_args")
            args = None

        depends_on = step.get("depends_on")
        if "depends_on" in step and (
            not isinstance(depends_on, list)
            or any(not isinstance(dep, str) or not dep.strip() for dep in depends_on)
        ):
            errors.append(f"{prefix}_invalid_depends_on")

        if isinstance(action, str) and action in ACTION_SPECS and isinstance(args, dict):
            for arg_name in ACTION_SPECS[action].get("required_args", ()):
                arg_value = args.get(arg_name)
                if not isinstance(arg_value, str) or not arg_value.strip():
                    errors.append(f"{prefix}_missing_or_invalid_arg_{arg_name}")

    return {"is_valid": not errors, "errors": errors}


def extract_task_context(row: Dict[str, Any]) -> Dict[str, Any]:
    task = row.get("task") if isinstance(row.get("task"), dict) else {}
    task_metadata = task.get("task_metadata") if isinstance(task.get("task_metadata"), dict) else {}

    selected_robot = row.get("selected_robot")
    if not selected_robot:
        chosen_plan = row.get("chosen_plan")
        if isinstance(chosen_plan, list) and chosen_plan and isinstance(chosen_plan[0], dict):
            selected_robot = chosen_plan[0].get("robot")

    return {
        "pair_id": row.get("pair_id"),
        "task_id": row.get("task_id"),
        "selected_robot": selected_robot or "unknown",
        "task_profile": task_metadata.get("task_profile", "unknown"),
        "margin": safe_float(row.get("margin")),
        "margin_label": canonical_margin_label(row.get("margin")),
        "prompt": row.get("prompt", ""),
        "start_node": task.get("start_node") or row.get("start_node"),
        "end_node": task.get("end_node") or row.get("end_node"),
        "target_item": task.get("target_item") or row.get("target_item"),
        "task": task,
        "task_metadata": task_metadata,
    }


def summarize_plan_shape(plan: Optional[List[Dict[str, Any]]]) -> Dict[str, Any]:
    if not isinstance(plan, list):
        return {
            "step_count": 0,
            "handover_count": 0,
            "single_robot": False,
            "robot_set": [],
        }

    robots = []
    handovers = 0
    for step in plan:
        if not isinstance(step, dict):
            continue
        robot = step.get("robot")
        if isinstance(robot, str) and robot:
            robots.append(robot)
        if step.get("action") == "handover":
            handovers += 1

    robot_set = sorted(set(robots))
    return {
        "step_count": len(plan),
        "handover_count": handovers,
        "single_robot": len(robot_set) == 1 and len(plan) > 0,
        "robot_set": robot_set,
    }


def aggregate_scored_rows(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    total = len(rows)
    violation_sample_counts = Counter()
    violation_instance_counts = Counter()
    failure_breakdown = Counter()

    valid_costs: List[float] = []
    parse_sum = 0
    schema_sum = 0
    executable_sum = 0
    exact_delivery_sum = 0
    penalty_sum = 0.0
    step_sum = 0
    handover_sum = 0
    single_robot_sum = 0

    for row in rows:
        parse_ok = bool(row.get("parse_ok"))
        schema_pass = bool(row.get("schema_pass"))
        executable = bool(row.get("executable_plan"))
        exact_delivery = bool(row.get("exact_delivery"))

        parse_sum += int(parse_ok)
        schema_sum += int(schema_pass)
        executable_sum += int(executable)
        exact_delivery_sum += int(exact_delivery)
        penalty_sum += float(row.get("penalty_score", 0.0) or 0.0)
        step_sum += int(row.get("step_count", 0) or 0)
        handover_sum += int(row.get("handover_count", 0) or 0)
        single_robot_sum += int(bool(row.get("single_robot")))

        if executable:
            cost = safe_float(row.get("trajectory_cost"))
            if cost is not None:
                valid_costs.append(cost)

        if not parse_ok:
            failure_breakdown["parse_failure"] += 1
        elif not schema_pass:
            failure_breakdown["schema_failure"] += 1
        elif not executable:
            failure_breakdown["symbolic_invalid"] += 1
        else:
            failure_breakdown["success"] += 1

        error_types = row.get("error_type") or []
        if not isinstance(error_types, list):
            error_types = []
        for name in set(error_types):
            violation_sample_counts[name] += 1

        violation_breakdown = row.get("violation_breakdown") or {}
        if isinstance(violation_breakdown, dict):
            for name, count in violation_breakdown.items():
                try:
                    violation_instance_counts[name] += int(count)
                except (TypeError, ValueError):
                    continue

    def rate(numerator: float) -> float:
        return round(numerator / total, 6) if total else 0.0

    invalid_total = total - executable_sum
    overall = {
        "count": total,
        "parse_rate": rate(parse_sum),
        "schema_pass_rate": rate(schema_sum),
        "executable_plan_rate": rate(executable_sum),
        "exact_delivery_rate": rate(exact_delivery_sum),
        "rule_violation_rate": round(1.0 - rate(executable_sum), 6) if total else 0.0,
        "avg_penalty_score": round(penalty_sum / total, 6) if total else 0.0,
        "avg_valid_trajectory_cost": round(sum(valid_costs) / len(valid_costs), 6) if valid_costs else None,
        "avg_step_count": round(step_sum / total, 6) if total else 0.0,
        "avg_handover_count": round(handover_sum / total, 6) if total else 0.0,
        "single_robot_rate": rate(single_robot_sum),
        "valid_plan_count": executable_sum,
        "invalid_plan_count": invalid_total,
    }

    sample_rate_over_all = {
        name: round(violation_sample_counts.get(name, 0) / total, 6) if total else 0.0
        for name in CANONICAL_VIOLATION_TYPES
    }
    sample_rate_over_invalid = {
        name: round(violation_sample_counts.get(name, 0) / invalid_total, 6) if invalid_total else 0.0
        for name in CANONICAL_VIOLATION_TYPES
    }
    total_instances = sum(violation_instance_counts.values())
    instance_share = {
        name: round(violation_instance_counts.get(name, 0) / total_instances, 6) if total_instances else 0.0
        for name in CANONICAL_VIOLATION_TYPES
    }

    return {
        "overall": overall,
        "failure_breakdown": dict(failure_breakdown),
        "violation_breakdown": {
            "sample_counts": {name: violation_sample_counts.get(name, 0) for name in CANONICAL_VIOLATION_TYPES},
            "sample_rate_over_all": sample_rate_over_all,
            "sample_rate_over_invalid": sample_rate_over_invalid,
            "instance_counts": {name: violation_instance_counts.get(name, 0) for name in CANONICAL_VIOLATION_TYPES},
            "instance_share": instance_share,
        },
    }


def aggregate_by_group(rows: List[Dict[str, Any]], key_name: str) -> Dict[str, Any]:
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for row in rows:
        key = row.get(key_name, "unknown")
        if key is None:
            key = "unknown"
        key = str(key)
        grouped.setdefault(key, []).append(row)

    return {
        key: aggregate_scored_rows(group_rows)["overall"]
        for key, group_rows in sorted(grouped.items(), key=lambda item: item[0])
    }


def render_report_markdown(report: Dict[str, Any]) -> str:
    def metric_row(name: str, value: Any) -> str:
        return f"| {name} | {value} |"

    lines = [
        "# Evaluation Report",
        "",
        f"- Input file: `{report.get('input_file', '')}`",
        f"- Text field: `{report.get('text_field', '')}`",
        f"- Total samples: `{report.get('overall', {}).get('count', 0)}`",
        "",
        "## Overall",
        "",
        "| Metric | Value |",
        "| --- | --- |",
    ]

    overall = report.get("overall", {})
    for key in (
        "parse_rate",
        "schema_pass_rate",
        "executable_plan_rate",
        "exact_delivery_rate",
        "rule_violation_rate",
        "avg_penalty_score",
        "avg_valid_trajectory_cost",
        "avg_step_count",
        "avg_handover_count",
        "single_robot_rate",
    ):
        lines.append(metric_row(key, overall.get(key)))

    lines.extend(
        [
            "",
            "## Failure Breakdown",
            "",
            "| Type | Count |",
            "| --- | --- |",
        ]
    )
    for key, value in sorted((report.get("failure_breakdown") or {}).items()):
        lines.append(metric_row(key, value))

    lines.extend(
        [
            "",
            "## Canonical Violation Rates",
            "",
            "| Violation | Rate over all samples | Rate over invalid samples |",
            "| --- | --- | --- |",
        ]
    )
    violation_block = report.get("violation_breakdown") or {}
    over_all = violation_block.get("sample_rate_over_all") or {}
    over_invalid = violation_block.get("sample_rate_over_invalid") or {}
    for name in CANONICAL_VIOLATION_TYPES:
        lines.append(f"| {name} | {over_all.get(name, 0.0)} | {over_invalid.get(name, 0.0)} |")

    for section_key, title in (
        ("by_robot", "By Robot"),
        ("by_task_profile", "By Task Profile"),
        ("by_margin", "By Margin"),
    ):
        section = report.get(section_key) or {}
        if not section:
            continue
        lines.extend(
            [
                "",
                f"## {title}",
                "",
                "| Group | Count | Parse | Schema | Executable | Exact Delivery | Margin/Violation |",
                "| --- | --- | --- | --- | --- | --- | --- |",
            ]
        )
        for key, metrics in section.items():
            lines.append(
                "| "
                + " | ".join(
                    [
                        str(key),
                        str(metrics.get("count")),
                        str(metrics.get("parse_rate")),
                        str(metrics.get("schema_pass_rate")),
                        str(metrics.get("executable_plan_rate")),
                        str(metrics.get("exact_delivery_rate")),
                        str(metrics.get("rule_violation_rate")),
                    ]
                )
                + " |"
            )

    return "\n".join(lines) + "\n"
