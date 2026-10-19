"""
Phase 4: export margin-aware DPO training files and leakage-aware splits.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

from preference_utils import (
    build_planning_prompt,
    build_preference_pair_id,
    compute_preference_strength,
    derive_task_metadata,
    render_plan_response,
    render_oracle_reasoning_trace,
)


DEFAULT_INPUT_FILE = "dpo_dataset_filtered.jsonl"
DEFAULT_OUTPUT_PREFIX = "dpo_margin_dataset"
FORMAT_NEGATIVE_VARIANTS = (
    "missing_answer_block",
    "empty_plan",
    "missing_depends_on",
    "missing_pick_up",
    "missing_put_down",
)
VALID_PAIR_TYPES = {"hard_invalid", "soft_suboptimal"}


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def dump_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def ensure_response_text(block: Dict[str, Any]) -> str:
    response_text = block.get("response_text")
    if isinstance(response_text, str) and response_text.strip():
        return response_text
    return render_plan_response(block.get("thought", ""), block.get("plan", []))


def build_normalized_prompt(sample: Dict[str, Any]) -> str:
    local_graph = sample.get("local_graph")
    if isinstance(local_graph, dict):
        return build_planning_prompt(
            instruction=sample["instruction"],
            scenario_json=local_graph,
            target_item=sample["target_item"],
            item_profile=sample.get("item_profile", {}),
            start_node=sample["start_node"],
            end_node=sample["end_node"],
            reference_paths=sample.get("reference_paths", []),
        )
    return sample["prompt"]


def build_concise_chosen_response(sample: Dict[str, Any]) -> str:
    chosen = sample["chosen"]
    reasoning_trace = chosen.get("reasoning_trace")
    if isinstance(reasoning_trace, dict) and reasoning_trace:
        thought = render_oracle_reasoning_trace(reasoning_trace)
    else:
        thought = (chosen.get("thought") or "").strip()
    return render_plan_response(thought, chosen.get("plan", []))


def _stable_variant_offset(record: Dict[str, Any]) -> int:
    key = record.get("pair_id") or record.get("task_id") or "record"
    digest = hashlib.sha1(str(key).encode("utf-8")).hexdigest()
    return int(digest[:8], 16)


def _synthetic_pref_strength(
    chosen_feedback: Dict[str, Any],
    margin: float,
    severity_level: str,
    penalty_score: float,
    error_types: List[str],
    chosen_utility: float,
    rejected_utility: float,
) -> Dict[str, Any]:
    chosen_penalty = float(chosen_feedback.get("penalty_score", 0.0) or 0.0)
    chosen_cost = float(chosen_feedback.get("trajectory_cost", 0.0) or 0.0)
    utility_gap = max(0.0, float(rejected_utility) - float(chosen_utility))
    breakdown = {error_name: 1 for error_name in error_types}
    return {
        "pair_type": "hard_invalid",
        "chosen_penalty_score": chosen_penalty,
        "rejected_penalty_score": chosen_penalty + float(penalty_score),
        "severity_score": float(penalty_score),
        "severity_source": "penalty_gap",
        "severity_level": severity_level,
        "severity_margin": margin,
        "major_violation_count": len(error_types),
        "minor_violation_count": 0,
        "total_violation_count": len(error_types),
        "chosen_trajectory_cost": chosen_cost,
        "rejected_trajectory_cost": chosen_cost,
        "trajectory_cost_gap": 0.0,
        "chosen_utility": round(float(chosen_utility), 6),
        "rejected_utility": round(float(rejected_utility), 6),
        "utility_gap": round(utility_gap, 6),
        "error_types": list(error_types),
        "violation_breakdown": breakdown,
    }


def normalize_pair_type(value: Any) -> str:
    if isinstance(value, str) and value in VALID_PAIR_TYPES:
        return value
    return "hard_invalid"


def to_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _structural_negative_thought(variant_name: str, start_node: str, end_node: str) -> str:
    if variant_name == "missing_answer_block":
        return (
            "[Task]\n"
            f"- Same transport task from {start_node} to {end_node}.\n"
            "[Edit]\n"
            "- Corrupt the response format by omitting the <answer> block."
        )
    if variant_name == "empty_plan":
        return (
            "[Task]\n"
            f"- Same transport task from {start_node} to {end_node}.\n"
            "[Edit]\n"
            "- Return an empty JSON plan."
        )
    if variant_name == "missing_depends_on":
        return (
            "[Task]\n"
            f"- Same transport task from {start_node} to {end_node}.\n"
            "[Edit]\n"
            "- Keep the plan steps but remove all depends_on fields."
        )
    if variant_name == "missing_pick_up":
        return (
            "[Task]\n"
            f"- Same transport task from {start_node} to {end_node}.\n"
            "[Edit]\n"
            "- Remove the required initial pick_up action."
        )
    return (
        "[Task]\n"
        f"- Same transport task from {start_node} to {end_node}.\n"
        "[Edit]\n"
        "- Remove the required final put_down action."
    )


def _build_structural_negative(record: Dict[str, Any], variant_name: str) -> Dict[str, Any] | None:
    chosen_plan = record.get("chosen_plan") or []
    if not isinstance(chosen_plan, list) or not chosen_plan:
        return None

    rejected_text = ""
    rejected_plan: List[Dict[str, Any]] = []
    margin = 1.1
    severity_level = "critical"
    penalty_score = 18.0
    error_types = ["format_failure"]
    start_node = record["task"]["start_node"]
    end_node = record["task"]["end_node"]
    thought = _structural_negative_thought(variant_name, start_node, end_node)

    if variant_name == "missing_answer_block":
        rejected_text = f"<think>\n{thought}\n</think>"
        rejected_plan = []
        penalty_score = 22.0
    elif variant_name == "empty_plan":
        rejected_plan = []
        rejected_text = render_plan_response(thought, rejected_plan)
        penalty_score = 20.0
    elif variant_name == "missing_depends_on":
        rejected_plan = [{k: v for k, v in step.items() if k != "depends_on"} for step in chosen_plan]
        rejected_text = render_plan_response(thought, rejected_plan)
        margin = 0.7
        severity_level = "severe"
        penalty_score = 14.0
    elif variant_name == "missing_pick_up":
        rejected_plan = chosen_plan[1:] if len(chosen_plan) > 1 else []
        rejected_text = render_plan_response(thought, rejected_plan)
        penalty_score = 18.0
    elif variant_name == "missing_put_down":
        rejected_plan = chosen_plan[:-1] if len(chosen_plan) > 1 else []
        rejected_text = render_plan_response(thought, rejected_plan)
        penalty_score = 18.0
    else:
        return None

    augmented = dict(record)
    chosen_utility = to_float(record.get("chosen_utility"), to_float(record["chosen_feedback"].get("trajectory_cost"), 0.0))
    rejected_utility = chosen_utility + max(1.0, penalty_score * 0.25)

    augmented["pair_id"] = f"{record['pair_id']}__fmt__{variant_name}"
    augmented["pair_type"] = "hard_invalid"
    augmented["chosen_valid"] = True
    augmented["rejected_valid"] = False
    augmented["chosen_utility"] = round(chosen_utility, 6)
    augmented["rejected_utility"] = round(rejected_utility, 6)
    augmented["utility_gap"] = round(rejected_utility - chosen_utility, 6)
    augmented["rejected"] = rejected_text
    augmented["rejected_plan"] = rejected_plan
    augmented["margin"] = margin
    augmented["preference_strength"] = _synthetic_pref_strength(
        chosen_feedback=record["chosen_feedback"],
        margin=margin,
        severity_level=severity_level,
        penalty_score=penalty_score,
        error_types=error_types,
        chosen_utility=chosen_utility,
        rejected_utility=rejected_utility,
    )
    augmented["rejected_feedback"] = {
        "is_valid": False,
        "error_type": error_types,
        "error_details": f"synthetic structural negative: {variant_name}",
        "failed_steps": [],
        "penalty_score": penalty_score,
        "trajectory_cost": record["chosen_feedback"].get("trajectory_cost"),
        "violation_breakdown": {error_name: 1 for error_name in error_types},
        "validation_source": "synthetic_format_augmentation",
    }
    augmented["target_errors"] = sorted(set(list(record.get("target_errors", [])) + error_types))
    generation_metadata = dict(record.get("generation_metadata", {}))
    generation_metadata["phase4_augmentation"] = variant_name
    augmented["generation_metadata"] = generation_metadata
    augmented["augmentation_type"] = variant_name
    augmented["exclude_from_sft"] = True
    return augmented


def augment_train_records(records: List[Dict[str, Any]], copies_per_record: int) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    if copies_per_record <= 0:
        return list(records), {}

    augmented_records = list(records)
    variant_counts: Counter[str] = Counter()

    for record in records:
        offset = _stable_variant_offset(record)
        used = 0
        for variant_index in range(len(FORMAT_NEGATIVE_VARIANTS)):
            if used >= copies_per_record:
                break
            variant_name = FORMAT_NEGATIVE_VARIANTS[(offset + variant_index) % len(FORMAT_NEGATIVE_VARIANTS)]
            augmented = _build_structural_negative(record, variant_name)
            if not augmented:
                continue
            augmented_records.append(augmented)
            variant_counts[variant_name] += 1
            used += 1

    return augmented_records, dict(variant_counts)


def top_counts(counter: Counter, limit: int = 20) -> Dict[str, int]:
    return dict(counter.most_common(limit))


def extract_selected_robot(record: Dict[str, Any]) -> str:
    selected_robot = record.get("selected_robot")
    if isinstance(selected_robot, str) and selected_robot:
        return selected_robot

    chosen_plan = record.get("chosen_plan", [])
    if chosen_plan and isinstance(chosen_plan, list):
        first_step = chosen_plan[0]
        if isinstance(first_step, dict):
            robot = first_step.get("robot")
            if isinstance(robot, str) and robot:
                return robot

    reasoning_trace = record.get("chosen_reasoning_trace") or {}
    selection = reasoning_trace.get("selection") or {}
    robot = selection.get("selected_robot")
    if isinstance(robot, str) and robot:
        return robot

    return "unknown"


def build_training_record(sample: Dict[str, Any]) -> Dict[str, Any]:
    prompt = build_normalized_prompt(sample)
    chosen = sample["chosen"]
    rejected = sample["rejected"]
    chosen_feedback = chosen["feedback"]
    rejected_feedback = rejected["feedback"]
    pair_type = normalize_pair_type(sample.get("pair_type"))
    chosen_utility = to_float(
        sample.get("chosen_utility"),
        to_float(chosen_feedback.get("trajectory_cost"), 0.0),
    )
    rejected_utility = to_float(
        sample.get("rejected_utility"),
        to_float(rejected_feedback.get("trajectory_cost"), chosen_utility),
    )
    utility_gap = round(rejected_utility - chosen_utility, 6)

    task_metadata = sample.get("task_metadata") or derive_task_metadata(
        sample["start_node"],
        sample["end_node"],
        sample["target_item"],
    )
    preference_strength = compute_preference_strength(
        chosen_feedback,
        rejected_feedback,
        pair_type=pair_type,
        chosen_utility=chosen_utility,
        rejected_utility=rejected_utility,
    )

    pair_id = sample.get("pair_id") or build_preference_pair_id(
        prompt,
        chosen["plan"],
        rejected["plan"],
    )
    selected_robot = (
        (chosen.get("reasoning_trace") or {}).get("selection", {}).get("selected_robot")
        or (chosen["plan"][0].get("robot") if chosen.get("plan") else "unknown")
    )

    return {
        "pair_id": pair_id,
        "task_id": sample.get("task_id"),
        "pair_type": pair_type,
        "dataset_schema_version": sample.get("dataset_schema_version", "pvr_dpo_v2"),
        "prompt_version": sample.get("prompt_version", "graph_prompt_v2"),
        "selected_robot": selected_robot,
        "chosen_valid": bool(sample.get("chosen_valid", chosen_feedback.get("is_valid", False))),
        "rejected_valid": bool(sample.get("rejected_valid", rejected_feedback.get("is_valid", False))),
        "chosen_utility": round(chosen_utility, 6),
        "rejected_utility": round(rejected_utility, 6),
        "utility_gap": utility_gap,
        "prompt": prompt,
        "chosen": build_concise_chosen_response(sample),
        "rejected": ensure_response_text(rejected),
        "margin": preference_strength["severity_margin"],
        "preference_strength": preference_strength,
        "instruction": sample["instruction"],
        "target_errors": sample.get("target_errors", []),
        "task": {
            "start_node": sample["start_node"],
            "end_node": sample["end_node"],
            "target_item": sample["target_item"],
            "item_profile": sample.get("item_profile", {}),
            "task_metadata": task_metadata,
            "local_graph": sample.get("local_graph"),
            "reference_paths": sample.get("reference_paths", []),
        },
        "chosen_plan": chosen["plan"],
        "rejected_plan": rejected["plan"],
        "chosen_feedback": chosen_feedback,
        "chosen_reasoning_trace": chosen.get("reasoning_trace"),
        "rejected_feedback": rejected_feedback,
        "generation_metadata": sample.get("generation_metadata", {}),
        "augmentation_type": None,
        "exclude_from_sft": False,
    }


def choose_group_key(record: Dict[str, Any], split_by: str) -> str:
    metadata = record["task"]["task_metadata"]
    if split_by == "random":
        return record["pair_id"]
    if split_by == "task":
        return metadata["task_signature"]
    if split_by == "item":
        return metadata["target_item"]
    if split_by == "building_pair":
        return metadata["building_pair"]
    if split_by == "start_building":
        return metadata["start_building"]
    if split_by == "end_building":
        return metadata["end_building"]
    raise ValueError(f"Unsupported split_by={split_by}")


def choose_stratify_key(record: Dict[str, Any], stratify_by: str) -> str:
    metadata = record["task"]["task_metadata"]
    selected_robot = extract_selected_robot(record)
    task_profile = metadata.get("task_profile", "unknown")

    if stratify_by == "none":
        return "__all__"
    if stratify_by == "robot":
        return selected_robot
    if stratify_by == "task_profile":
        return task_profile
    if stratify_by == "robot_profile":
        return f"{selected_robot}|{task_profile}"
    raise ValueError(f"Unsupported stratify_by={stratify_by}")


def choose_balanced_eval_key(record: Dict[str, Any], balanced_eval_by: str) -> str:
    if balanced_eval_by == "none":
        return "__none__"

    selected_robot = extract_selected_robot(record)
    if balanced_eval_by == "robot":
        return selected_robot
    if balanced_eval_by == "robot_severity":
        severity = record["preference_strength"]["severity_level"]
        return f"{selected_robot}|{severity}"
    raise ValueError(f"Unsupported balanced_eval_by={balanced_eval_by}")


def _dominant_stratify_key(rows: List[Dict[str, Any]], stratify_by: str) -> str:
    labels = Counter(choose_stratify_key(row, stratify_by) for row in rows)
    return labels.most_common(1)[0][0]


def _balanced_eval_diversity_score(
    record: Dict[str, Any],
    selected_profile_counts: Counter,
    selected_severity_counts: Counter,
    selected_item_counts: Counter,
    selected_pair_counts: Counter,
    group_profile_counts: Counter,
    group_severity_counts: Counter,
    group_item_counts: Counter,
    group_pair_counts: Counter,
) -> Tuple[int, int, int, int, int, int, int, int]:
    metadata = record["task"]["task_metadata"]
    profile = metadata.get("task_profile", "unknown")
    severity = record["preference_strength"]["severity_level"]
    item = metadata["target_item"]
    building_pair = metadata["building_pair"]

    return (
        selected_profile_counts[profile],
        selected_severity_counts[severity],
        selected_item_counts[item],
        selected_pair_counts[building_pair],
        group_profile_counts[profile],
        group_severity_counts[severity],
        group_item_counts[item],
        group_pair_counts[building_pair],
    )


def _select_diverse_group_subset(
    records: List[Dict[str, Any]],
    k: int,
    seed: int,
) -> List[Dict[str, Any]]:
    rows = list(records)
    rng = random.Random(seed)
    rng.shuffle(rows)

    group_profile_counts = Counter(record["task"]["task_metadata"].get("task_profile", "unknown") for record in rows)
    group_severity_counts = Counter(record["preference_strength"]["severity_level"] for record in rows)
    group_item_counts = Counter(record["task"]["task_metadata"]["target_item"] for record in rows)
    group_pair_counts = Counter(record["task"]["task_metadata"]["building_pair"] for record in rows)

    selected: List[Dict[str, Any]] = []
    selected_profile_counts = Counter()
    selected_severity_counts = Counter()
    selected_item_counts = Counter()
    selected_pair_counts = Counter()

    while rows and len(selected) < k:
        best_row = min(
            rows,
            key=lambda record: _balanced_eval_diversity_score(
                record,
                selected_profile_counts,
                selected_severity_counts,
                selected_item_counts,
                selected_pair_counts,
                group_profile_counts,
                group_severity_counts,
                group_item_counts,
                group_pair_counts,
            ),
        )
        rows.remove(best_row)
        selected.append(best_row)

        metadata = best_row["task"]["task_metadata"]
        selected_profile_counts[metadata.get("task_profile", "unknown")] += 1
        selected_severity_counts[best_row["preference_strength"]["severity_level"]] += 1
        selected_item_counts[metadata["target_item"]] += 1
        selected_pair_counts[metadata["building_pair"]] += 1

    return selected


def reserve_balanced_eval_records(
    records: List[Dict[str, Any]],
    balanced_eval_by: str,
    balanced_eval_per_group: int,
    seed: int,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    if balanced_eval_by == "none":
        return [], list(records), {"enabled": False}

    groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for record in records:
        groups[choose_balanced_eval_key(record, balanced_eval_by)].append(record)

    if not groups:
        return [], list(records), {"enabled": False}

    min_available = min(len(rows) for rows in groups.values())
    if balanced_eval_per_group <= 0:
        if balanced_eval_by == "robot":
            # Keep a balanced held-out set without consuming every scarce robot sample.
            resolved_per_group = min(10, max(4, min_available // 2))
        else:
            resolved_per_group = min_available
    else:
        resolved_per_group = min(balanced_eval_per_group, min_available)
    if resolved_per_group <= 0:
        return [], list(records), {"enabled": False}

    selected: List[Dict[str, Any]] = []
    selected_pair_ids = set()

    for index, group_key in enumerate(sorted(groups)):
        subset = _select_diverse_group_subset(
            groups[group_key],
            k=resolved_per_group,
            seed=seed + index,
        )
        selected.extend(subset)
        for record in subset:
            selected_pair_ids.add(record["pair_id"])

    remaining = [record for record in records if record["pair_id"] not in selected_pair_ids]
    return selected, remaining, {
        "enabled": True,
        "balanced_eval_by": balanced_eval_by,
        "requested_per_group": balanced_eval_per_group,
        "resolved_per_group": resolved_per_group,
        "group_count": len(groups),
        "selected_count": len(selected),
    }


def _assign_group(
    grouped_item: Dict[str, Any],
    split_name: str,
    assigned: Dict[str, List[Dict[str, Any]]],
    assigned_counts: Dict[str, int],
    split_strata: Dict[str, set[str]],
) -> None:
    assigned[split_name].extend(grouped_item["rows"])
    assigned_counts[split_name] += grouped_item["size"]
    split_strata[split_name].add(grouped_item["stratum"])


def _choose_seed_split(
    grouped_item: Dict[str, Any],
    candidate_splits: List[str],
    assigned_counts: Dict[str, int],
    target_sizes: Dict[str, float],
    tolerance: Dict[str, float],
) -> str | None:
    viable = []
    for split_name in candidate_splits:
        projected = assigned_counts[split_name] + grouped_item["size"]
        if projected <= target_sizes[split_name] + tolerance[split_name]:
            viable.append(split_name)

    if not viable:
        return None

    return max(viable, key=lambda split_name: target_sizes[split_name] - assigned_counts[split_name])


def _best_split_for_group(
    grouped_item: Dict[str, Any],
    assigned_counts: Dict[str, int],
    target_sizes: Dict[str, float],
    split_strata: Dict[str, set[str]],
    stratum_group_counts: Counter,
    tolerance: Dict[str, float],
) -> str:
    best_split = "train"
    best_score = float("-inf")

    for split_name in ("train", "val", "test"):
        size_after = assigned_counts[split_name] + grouped_item["size"]
        deficit = target_sizes[split_name] - assigned_counts[split_name]
        overflow = max(0.0, size_after - target_sizes[split_name])

        coverage_bonus = 0.0
        if split_name in {"val", "test"} and grouped_item["stratum"] not in split_strata[split_name]:
            rarity = max(1, stratum_group_counts[grouped_item["stratum"]])
            coverage_bonus = 20.0 + (4.0 / rarity)

        tolerance_penalty = 0.0
        if size_after > target_sizes[split_name] + tolerance[split_name]:
            tolerance_penalty = 50.0

        score = coverage_bonus + deficit - (overflow * 3.0) - tolerance_penalty
        if split_name == "train":
            score += 1.0

        if score > best_score:
            best_score = score
            best_split = split_name

    return best_split


def assign_grouped_splits(
    records: List[Dict[str, Any]],
    split_by: str,
    stratify_by: str,
    seed: int,
    train_ratio: float,
    val_ratio: float,
) -> Dict[str, List[Dict[str, Any]]]:
    groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for record in records:
        groups[choose_group_key(record, split_by)].append(record)

    grouped_items: List[Dict[str, Any]] = []
    stratum_group_counts = Counter()
    for group_key, group_rows in groups.items():
        grouped_item = {
            "group_key": group_key,
            "rows": group_rows,
            "size": len(group_rows),
            "stratum": _dominant_stratify_key(group_rows, stratify_by),
        }
        grouped_items.append(grouped_item)
        stratum_group_counts[grouped_item["stratum"]] += 1

    rng = random.Random(seed)
    rng.shuffle(grouped_items)
    grouped_items.sort(key=lambda item: (stratum_group_counts[item["stratum"]], item["size"]))

    total = len(records)
    target_sizes = {
        "train": total * train_ratio,
        "val": total * val_ratio,
        "test": total * max(0.0, 1.0 - train_ratio - val_ratio),
    }
    assigned = {"train": [], "val": [], "test": []}
    assigned_counts = {"train": 0, "val": 0, "test": 0}
    split_strata = {"train": set(), "val": set(), "test": set()}
    tolerance = {
        "train": max(2.0, total * 0.02),
        "val": max(1.0, total * 0.02),
        "test": max(1.0, total * 0.02),
    }

    if stratify_by != "none":
        groups_by_stratum: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for grouped_item in grouped_items:
            groups_by_stratum[grouped_item["stratum"]].append(grouped_item)

        assigned_group_keys = set()
        for stratum, stratum_groups in sorted(groups_by_stratum.items(), key=lambda item: (len(item[1]), item[0])):
            available = [group for group in sorted(stratum_groups, key=lambda item: item["size"]) if group["group_key"] not in assigned_group_keys]
            if not available:
                continue

            if len(available) >= 2:
                for split_name in ("val", "test"):
                    chosen_split = _choose_seed_split(
                        available[0],
                        [split_name],
                        assigned_counts,
                        target_sizes,
                        tolerance,
                    )
                    if chosen_split is None:
                        continue
                    selected_group = available.pop(0)
                    _assign_group(selected_group, chosen_split, assigned, assigned_counts, split_strata)
                    assigned_group_keys.add(selected_group["group_key"])
                    if not available:
                        break
            elif len(available) == 1:
                selected_group = available[0]
                chosen_split = _choose_seed_split(
                    selected_group,
                    ["val", "test"],
                    assigned_counts,
                    target_sizes,
                    tolerance,
                )
                if chosen_split is not None:
                    _assign_group(selected_group, chosen_split, assigned, assigned_counts, split_strata)
                    assigned_group_keys.add(selected_group["group_key"])

        remaining_grouped_items = [item for item in grouped_items if item["group_key"] not in assigned_group_keys]
    else:
        remaining_grouped_items = grouped_items

    remaining_grouped_items.sort(key=lambda item: item["size"], reverse=True)
    for grouped_item in remaining_grouped_items:
        split_name = _best_split_for_group(
            grouped_item,
            assigned_counts,
            target_sizes,
            split_strata,
            stratum_group_counts,
            tolerance,
        )
        _assign_group(grouped_item, split_name, assigned, assigned_counts, split_strata)

    return assigned


def summarize_records(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    severity_counter = Counter()
    margin_counter = Counter()
    error_counter = Counter()
    pair_type_counter = Counter()
    item_counter = Counter()
    building_pair_counter = Counter()
    robot_counter = Counter()
    task_profile_counter = Counter()

    for record in records:
        pair_type_counter[record.get("pair_type", "hard_invalid")] += 1
        pref = record["preference_strength"]
        severity_counter[pref["severity_level"]] += 1
        margin_counter[str(pref["severity_margin"])] += 1
        for error_name in pref.get("error_types", []):
            error_counter[error_name] += 1

        metadata = record["task"]["task_metadata"]
        item_counter[metadata["target_item"]] += 1
        building_pair_counter[metadata["building_pair"]] += 1
        task_profile_counter[metadata.get("task_profile", "unknown")] += 1
        robot_counter[extract_selected_robot(record)] += 1

    return {
        "count": len(records),
        "pair_type_distribution": dict(pair_type_counter),
        "severity_distribution": dict(severity_counter),
        "margin_distribution": dict(margin_counter),
        "error_type_distribution": dict(error_counter),
        "robot_distribution": dict(robot_counter),
        "task_profile_distribution": dict(task_profile_counter),
        "top_items": top_counts(item_counter),
        "top_building_pairs": top_counts(building_pair_counter),
    }


def prepare_records(samples: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    prepared = []
    dropped = []

    for sample in samples:
        try:
            if not sample.get("prompt"):
                raise ValueError("missing_prompt")
            if not isinstance(sample.get("local_graph"), dict):
                raise ValueError("missing_local_graph")
            prepared.append(build_training_record(sample))
        except Exception as exc:
            dropped.append(
                {
                    "pair_id": sample.get("pair_id"),
                    "reason": str(exc),
                    "sample": sample,
                }
            )

    return prepared, dropped


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare margin-aware DPO training files.")
    parser.add_argument("--input", default=DEFAULT_INPUT_FILE, help="Filtered jsonl input file.")
    parser.add_argument(
        "--output-prefix",
        default=DEFAULT_OUTPUT_PREFIX,
        help="Prefix for exported all/train/val/test jsonl files.",
    )
    parser.add_argument(
        "--split-by",
        default="task",
        choices=["random", "task", "item", "building_pair", "start_building", "end_building"],
        help="Grouping key used to avoid leakage across train/val/test.",
    )
    parser.add_argument(
        "--stratify-by",
        default="robot_profile",
        choices=["none", "robot", "task_profile", "robot_profile"],
        help="Distribution key balanced across splits after leakage-aware grouping.",
    )
    parser.add_argument(
        "--balanced-eval-by",
        default="robot",
        choices=["none", "robot", "robot_severity"],
        help="Reserve an additional held-out balanced evaluation subset before train/val/test splitting.",
    )
    parser.add_argument(
        "--balanced-eval-per-group",
        type=int,
        default=0,
        help="Samples per balanced-eval group. Use 0 to auto-resolve to the minimum available group count.",
    )
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--train-format-negative-copies",
        type=int,
        default=1,
        help="Number of synthetic structural hard negatives to add per training record. Applied to train only.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base_dir = Path(__file__).resolve().parent
    input_path = base_dir / args.input
    output_prefix = base_dir / args.output_prefix

    samples = load_jsonl(input_path)
    prepared_records, dropped_records = prepare_records(samples)
    balanced_eval_records, split_input_records, balanced_eval_meta = reserve_balanced_eval_records(
        prepared_records,
        balanced_eval_by=args.balanced_eval_by,
        balanced_eval_per_group=args.balanced_eval_per_group,
        seed=args.seed,
    )
    split_records = assign_grouped_splits(
        split_input_records,
        split_by=args.split_by,
        stratify_by=args.stratify_by,
        seed=args.seed,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
    )
    base_train_count = len(split_records["train"])
    augmented_train_records, augmentation_meta = augment_train_records(
        split_records["train"],
        copies_per_record=args.train_format_negative_copies,
    )
    split_records["train"] = augmented_train_records

    dump_jsonl(output_prefix.with_name(f"{output_prefix.name}_all.jsonl"), prepared_records)
    dump_jsonl(output_prefix.with_name(f"{output_prefix.name}_balanced_eval.jsonl"), balanced_eval_records)
    dump_jsonl(output_prefix.with_name(f"{output_prefix.name}_train.jsonl"), split_records["train"])
    dump_jsonl(output_prefix.with_name(f"{output_prefix.name}_val.jsonl"), split_records["val"])
    dump_jsonl(output_prefix.with_name(f"{output_prefix.name}_test.jsonl"), split_records["test"])
    dump_jsonl(output_prefix.with_name(f"{output_prefix.name}_dropped.jsonl"), dropped_records)

    report = {
        "input_file": str(input_path.name),
        "output_prefix": output_prefix.name,
        "split_by": args.split_by,
        "stratify_by": args.stratify_by,
        "balanced_eval_by": args.balanced_eval_by,
        "balanced_eval_per_group": args.balanced_eval_per_group,
        "seed": args.seed,
        "train_ratio": args.train_ratio,
        "val_ratio": args.val_ratio,
        "train_format_negative_copies": args.train_format_negative_copies,
        "test_ratio": round(max(0.0, 1.0 - args.train_ratio - args.val_ratio), 4),
        "total_input": len(samples),
        "prepared": len(prepared_records),
        "dropped": len(dropped_records),
        "balanced_eval": {
            **balanced_eval_meta,
            "summary": summarize_records(balanced_eval_records) if balanced_eval_records else {"count": 0},
        },
        "split_pool_count": len(split_input_records),
        "train_augmentation": {
            "enabled": args.train_format_negative_copies > 0,
            "base_train_count": base_train_count,
            "augmented_train_count": len(split_records["train"]),
            "added_count": len(split_records["train"]) - base_train_count,
            "variant_counts": augmentation_meta,
        },
        "overall": summarize_records(prepared_records),
        "train": summarize_records(split_records["train"]),
        "val": summarize_records(split_records["val"]),
        "test": summarize_records(split_records["test"]),
    }

    report_path = output_prefix.with_name(f"{output_prefix.name}_report.json")
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print("=" * 72)
    print("Phase 4 Training Export Complete")
    print("=" * 72)
    print(f"Input samples: {len(samples)}")
    print(f"Prepared records: {len(prepared_records)}")
    print(f"Dropped records: {len(dropped_records)}")
    print(f"Split mode: {args.split_by}")
    if balanced_eval_records:
        print(
            f"Balanced eval: {len(balanced_eval_records)} "
            f"(by={args.balanced_eval_by}, per_group={balanced_eval_meta.get('resolved_per_group')})"
        )
    print(f"Train/Val/Test: {len(split_records['train'])}/{len(split_records['val'])}/{len(split_records['test'])}")
    if args.train_format_negative_copies > 0:
        print(f"Train structural negatives added: {sum(augmentation_meta.values())} -> {augmentation_meta}")
    print(f"Export prefix: {output_prefix.name}")
    print("=" * 72)


if __name__ == "__main__":
    main()
