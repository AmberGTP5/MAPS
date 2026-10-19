"""
Phase 3: strict filtering for high-confidence preference data.
"""

from __future__ import annotations

import json
import os
from collections import Counter
from datetime import datetime
from typing import Any, Dict, List, Tuple

from preference_utils import build_preference_pair_id, compute_preference_strength, render_plan_response
from symbolic_validator import SymbolicValidator


INPUT_FILE = "dpo_dataset_validated.jsonl"
OUTPUT_FILE = "dpo_dataset_filtered.jsonl"
REJECTED_FILE = "dpo_dataset_rejected_by_filter.jsonl"
REPORT_FILE = "filter_report.json"

validator = SymbolicValidator()
VALID_PAIR_TYPES = {"hard_invalid", "soft_suboptimal"}


def is_nonempty_text(value):
    return isinstance(value, str) and len(value.strip()) > 0


def parse_single_step(s_val, max_len):
    if isinstance(s_val, int):
        return s_val if 1 <= s_val <= max_len else None

    if isinstance(s_val, str):
        s = s_val.strip().lower()
        if s.startswith("t") and s[1:].isdigit():
            step_num = int(s[1:])
            return step_num if 1 <= step_num <= max_len else None

        s = s.replace("step", "").strip()
        if s.isdigit():
            step_num = int(s)
            return step_num if 1 <= step_num <= max_len else None
    return None


def normalize_failed_steps(feedback, rejected_plan):
    steps = feedback.get("failed_steps", [])
    if not isinstance(steps, list):
        if "failed_step" in feedback:
            steps = [feedback["failed_step"]]
        else:
            return None

    max_len = len(rejected_plan) if isinstance(rejected_plan, list) else 0
    valid_steps = []
    for step in steps:
        parsed = parse_single_step(step, max_len)
        if parsed is not None:
            valid_steps.append(parsed)
    return valid_steps if valid_steps else None


def plans_equal(plan_a, plan_b):
    try:
        return json.dumps(plan_a, sort_keys=True, ensure_ascii=False) == json.dumps(
            plan_b, sort_keys=True, ensure_ascii=False
        )
    except Exception:
        return plan_a == plan_b


def validate_chosen_plan(sample: Dict) -> Tuple[bool, Dict]:
    feedback = validator.validate_plan(
        sample["chosen"]["plan"],
        start_node=sample["start_node"],
        end_node=sample["end_node"],
        target_item=sample["target_item"],
    )
    return feedback.get("is_valid", False), feedback


def normalize_pair_type(sample: Dict[str, Any]) -> str:
    pair_type = sample.get("pair_type", "hard_invalid")
    if not isinstance(pair_type, str) or pair_type not in VALID_PAIR_TYPES:
        return "hard_invalid"
    return pair_type


def to_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def filter_sample(sample):
    reasons: List[str] = []
    pair_type = normalize_pair_type(sample)
    sample["pair_type"] = pair_type
    chosen = sample.get("chosen")
    rejected = sample.get("rejected")
    target_errors = sample.get("target_errors", [])
    prompt = sample.get("prompt")
    local_graph = sample.get("local_graph")

    if not isinstance(chosen, dict):
        return False, ["missing_chosen"], sample
    if not isinstance(rejected, dict):
        return False, ["missing_rejected"], sample

    chosen_plan = chosen.get("plan")
    rejected_plan = rejected.get("plan")
    if not isinstance(chosen_plan, list) or not chosen_plan:
        reasons.append("invalid_chosen_plan")
    if not isinstance(rejected_plan, list) or not rejected_plan:
        reasons.append("invalid_rejected_plan")

    if not is_nonempty_text(prompt):
        reasons.append("missing_prompt")

    if not isinstance(local_graph, dict):
        reasons.append("missing_local_graph")
    else:
        if not isinstance(local_graph.get("local_nodes"), list) or not isinstance(
            local_graph.get("local_edges"), list
        ):
            reasons.append("invalid_local_graph")

    if is_nonempty_text(chosen.get("thought")) and isinstance(chosen_plan, list):
        chosen["response_text"] = chosen.get("response_text") or render_plan_response(
            chosen["thought"], chosen_plan
        )
    if is_nonempty_text(rejected.get("thought")) and isinstance(rejected_plan, list):
        rejected["response_text"] = rejected.get("response_text") or render_plan_response(
            rejected["thought"], rejected_plan
        )

    if not is_nonempty_text(chosen.get("response_text")):
        reasons.append("missing_chosen_response_text")
    if not is_nonempty_text(rejected.get("response_text")):
        reasons.append("missing_rejected_response_text")

    chosen_ok, chosen_feedback = validate_chosen_plan(sample)
    sample["chosen"]["feedback"] = chosen_feedback
    if not chosen_ok:
        reasons.append("chosen_failed_rule_validation")

    feedback = rejected.get("feedback")
    if not isinstance(feedback, dict):
        reasons.append("missing_feedback")
        return False, reasons, sample

    rejected_valid = bool(feedback.get("is_valid", False))
    sample["chosen_valid"] = bool(chosen_feedback.get("is_valid", False))
    sample["rejected_valid"] = rejected_valid

    if pair_type == "hard_invalid":
        if rejected_valid:
            reasons.append("feedback_not_invalid")

        detected_errors = feedback.get("error_type", [])
        if not isinstance(detected_errors, list):
            detected_errors = [detected_errors]
        if target_errors and set(target_errors) - set(detected_errors):
            reasons.append("error_type_mismatch_or_missed")

        if "penalty_score" not in feedback or not isinstance(feedback["penalty_score"], (int, float)):
            reasons.append("missing_penalty_score")

        if not is_nonempty_text(feedback.get("error_details")):
            reasons.append("missing_error_details")

        normalized_steps = normalize_failed_steps(feedback, rejected_plan)
        if not normalized_steps:
            reasons.append("invalid_failed_steps")
        else:
            feedback["failed_steps"] = normalized_steps

    elif pair_type == "soft_suboptimal":
        if not rejected_valid:
            reasons.append("soft_rejected_not_valid")
        if target_errors:
            reasons.append("soft_pair_should_not_have_target_errors")

        chosen_utility = to_float(
            sample.get("chosen_utility"),
            to_float(chosen_feedback.get("trajectory_cost"), 0.0),
        )
        rejected_utility = to_float(
            sample.get("rejected_utility"),
            to_float(feedback.get("trajectory_cost"), chosen_utility),
        )
        utility_gap = round(rejected_utility - chosen_utility, 6)
        sample["chosen_utility"] = round(chosen_utility, 6)
        sample["rejected_utility"] = round(rejected_utility, 6)
        sample["utility_gap"] = utility_gap

        if utility_gap <= 0:
            reasons.append("soft_nonpositive_utility_gap")

        # For soft pairs we only require both plans to be valid; no invalid-step fields.
        feedback["error_type"] = []
        feedback["failed_steps"] = []
        if not is_nonempty_text(feedback.get("error_details")):
            feedback["error_details"] = "valid_but_suboptimal_soft_pair"

    if "penalty_score" not in feedback or not isinstance(feedback["penalty_score"], (int, float)):
        feedback["penalty_score"] = to_float(feedback.get("penalty_score"), 0.0)

    if plans_equal(chosen_plan, rejected_plan):
        reasons.append("chosen_rejected_identical")

    validation_source = feedback.get("validation_source")
    if not validation_source and pair_type == "soft_suboptimal":
        validation_source = "rule_engine"
        feedback["validation_source"] = "rule_engine"
    if validation_source not in ["rule_engine", "claude", "gpt", "llm"]:
        reasons.append("invalid_validation_source")

    if pair_type == "hard_invalid" and is_nonempty_text(feedback.get("error_details")) and len(feedback["error_details"].strip()) < 20:
        reasons.append("error_details_too_short")

    if is_nonempty_text(prompt) and isinstance(chosen_plan, list) and isinstance(rejected_plan, list):
        sample["pair_id"] = sample.get("pair_id") or build_preference_pair_id(
            prompt,
            chosen_plan,
            rejected_plan,
        )

    if isinstance(chosen_feedback, dict) and isinstance(feedback, dict):
        sample["preference_strength"] = compute_preference_strength(
            chosen_feedback,
            feedback,
            pair_type=pair_type,
            chosen_utility=sample.get("chosen_utility"),
            rejected_utility=sample.get("rejected_utility"),
        )

    return len(reasons) == 0, reasons, sample


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)

    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        samples = [json.loads(line) for line in f]

    kept = []
    rejected = []
    drop_reasons = Counter()
    source_stats = Counter()
    error_stats = Counter()
    pair_type_before = Counter()
    pair_type_kept = Counter()

    for sample in samples:
        pair_type_before[normalize_pair_type(sample)] += 1
        keep, reasons, processed_sample = filter_sample(sample)
        feedback = processed_sample.get("rejected", {}).get("feedback", {})
        source = feedback.get("validation_source", "unknown")
        source_stats[source] += 1
        for error_name in processed_sample.get("target_errors", []):
            error_stats[error_name] += 1

        if keep:
            kept.append(processed_sample)
            pair_type_kept[processed_sample.get("pair_type", "unknown")] += 1
        else:
            rejected.append({"sample": processed_sample, "filter_reasons": reasons})
            for reason in reasons:
                drop_reasons[reason] += 1

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for sample in kept:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")

    with open(REJECTED_FILE, "w", encoding="utf-8") as f:
        for item in rejected:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    report = {
        "timestamp": datetime.now().isoformat(),
        "input_file": INPUT_FILE,
        "total": len(samples),
        "kept_strict": len(kept),
        "dropped": len(rejected),
        "keep_rate": round(len(kept) / len(samples) * 100, 2) if samples else 0.0,
        "drop_reasons": dict(drop_reasons),
        "source_stats_before_filter": dict(source_stats),
        "pair_type_before_filter": dict(pair_type_before),
        "pair_type_kept": dict(pair_type_kept),
        "target_error_distribution": dict(error_stats),
        "output_file": OUTPUT_FILE,
        "rejected_file": REJECTED_FILE,
    }

    with open(REPORT_FILE, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print("=" * 70)
    print("Phase 3 Filter Complete")
    print("=" * 70)
    print(f"Input total: {len(samples)}")
    print(f"Kept (strict): {len(kept)}")
    print(f"Dropped: {len(rejected)}")
    print(f"Keep rate: {report['keep_rate']}%")
    print("\nDrop reasons:")
    for reason, count in drop_reasons.items():
        print(f"  - {reason}: {count}")
    print("\nPair types (before -> kept):")
    all_pair_types = sorted(set(pair_type_before.keys()) | set(pair_type_kept.keys()))
    for pair_type in all_pair_types:
        print(f"  - {pair_type}: {pair_type_before.get(pair_type, 0)} -> {pair_type_kept.get(pair_type, 0)}")
    print("\nOutputs:")
    print(f"  - {OUTPUT_FILE}")
    print(f"  - {REJECTED_FILE}")
    print(f"  - {REPORT_FILE}")
    print("=" * 70)


if __name__ == "__main__":
    main()
