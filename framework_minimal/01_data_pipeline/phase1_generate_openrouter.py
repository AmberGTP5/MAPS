"""
Phase 1: generate chosen/rejected preference pairs for graph-grounded planning.
"""

from __future__ import annotations

import json
import os
import random
import re
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from openai import OpenAI
from tqdm import tqdm

from graph_engine import build_default_graph_engine
from negative_plan_mutator import NegativePlanMutator
from oracle_planner import OraclePlanner
from preference_utils import build_preference_pair_id, render_plan_response
from symbolic_validator import SymbolicValidator
from task_generator_english import EnglishTaskGenerator as TaskGenerator


OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_BASE_URL = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
GEMINI_MODEL = os.getenv("PVR_PHASE1_MODEL", "google/gemini-2.5-flash")

NUM_SAMPLES = int(os.getenv("PVR_PHASE1_NUM_SAMPLES", "500"))
COOLDOWN_SECONDS = int(os.getenv("PVR_PHASE1_COOLDOWN_SECONDS", "5"))
MAX_REJECTED_ATTEMPTS = int(os.getenv("PVR_PHASE1_REJECTED_ATTEMPTS", "3"))
MAX_TASK_DEDUP_ATTEMPTS = int(os.getenv("PVR_PHASE1_TASK_DEDUP_ATTEMPTS", "80"))
MAX_GLOBAL_ATTEMPTS = int(os.getenv("PVR_PHASE1_MAX_GLOBAL_ATTEMPTS", str(max(2000, NUM_SAMPLES * 20))))
RANDOM_SEED = int(os.getenv("PVR_PHASE1_SEED", "42"))
INCLUDE_RESOURCE_CONFLICT = os.getenv("PVR_PHASE1_INCLUDE_RESOURCE_CONFLICT", "0") == "1"
USE_LLM_REJECTED_FALLBACK = os.getenv("PVR_PHASE1_USE_LLM_REJECTED_FALLBACK", "1") == "1"
TOPK_CANDIDATES = int(os.getenv("PVR_PHASE1_TOPK_CANDIDATES", "4"))
TOPK_PATHS_PER_ROBOT = int(os.getenv("PVR_PHASE1_TOPK_PATHS_PER_ROBOT", "3"))
MIN_SOFT_UTILITY_GAP = float(os.getenv("PVR_PHASE1_MIN_SOFT_UTILITY_GAP", "2.0"))
ENABLE_SOFT_SUBOPTIMAL = os.getenv("PVR_PHASE1_ENABLE_SOFT_SUBOPTIMAL", "1") == "1"
ALLOW_HARD_TARGET_MISMATCH = os.getenv("PVR_PHASE1_ALLOW_HARD_TARGET_MISMATCH", "0") == "1"

OUTPUT_FILE = "dpo_dataset_phase1.jsonl"
LOG_FILE = "generation_log_phase1.txt"
DATASET_SCHEMA_VERSION = "pvr_dpo_v2"

ALL_ERROR_DEFINITIONS = {
    "capability_mismatch": {
        "name": "Capability Mismatch",
        "description": "A robot uses an action or edge that exceeds its capabilities.",
    },
    "morphology_constraint": {
        "name": "Morphology Constraint Violation",
        "description": "A robot violates node-type, payload, stair, elevator, or aerial restrictions.",
    },
    "coordination_failure": {
        "name": "Coordination Failure",
        "description": "Robots fail to coordinate locations, handovers, or item ownership correctly.",
    },
    "resource_conflict": {
        "name": "Resource Conflict",
        "description": "Robots compete for the same elevator/resource without dependency ordering.",
    },
}

ERROR_DEFINITIONS = dict(ALL_ERROR_DEFINITIONS)
if not INCLUDE_RESOURCE_CONFLICT:
    # With the current schema only `chassis_car` can legally traverse elevator edges,
    # so true resource-conflict negatives are effectively unattainable without inventing
    # unsupported extra robot identities.
    ERROR_DEFINITIONS.pop("resource_conflict", None)

client = OpenAI(base_url=OPENROUTER_BASE_URL, api_key=OPENROUTER_API_KEY or "MISSING")
validator: Optional[SymbolicValidator] = None
negative_mutator: Optional[NegativePlanMutator] = None

log_file = None
stats = {
    "total_attempts": 0,
    "successful_generations": 0,
    "api_failures": 0,
    "json_parse_failures": 0,
    "chosen_planner_failures": 0,
    "rejected_validation_failures": 0,
    "rejected_target_misses": 0,
    "symbolic_rejected_successes": 0,
    "llm_rejected_successes": 0,
    "duplicate_task_skips": 0,
    "hard_target_mismatch_drops": 0,
    "error_distribution": Counter(),
    "pair_type_distribution": Counter(),
    "hard_invalid_successes": 0,
    "soft_suboptimal_successes": 0,
}


def log(message: str, level: str = "INFO") -> None:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_message = f"[{timestamp}] [{level}] {message}"
    print(log_message)
    if log_file:
        log_file.write(log_message + "\n")
        log_file.flush()


def call_model(
    prompt: str,
    temperature: float = 0.4,
    max_tokens: int = 3000,
    max_retries: int = 3,
) -> Optional[str]:
    if not OPENROUTER_API_KEY:
        raise RuntimeError("OPENROUTER_API_KEY is not set.")

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=GEMINI_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return response.choices[0].message.content
        except Exception as exc:
            log(f"API call failed ({attempt + 1}/{max_retries}): {exc}", "WARNING")
            if attempt < max_retries - 1:
                time.sleep(5 * (attempt + 1))
            else:
                stats["api_failures"] += 1
                return None


def _extract_tagged_block(text: str, tag: str) -> str:
    match = re.search(rf"<{tag}>(.*?)</{tag}>", text, re.DOTALL | re.IGNORECASE)
    return match.group(1).strip() if match else ""


def _extract_json_fence(text: str) -> str:
    fenced = re.search(r"```json\s*(.*?)\s*```", text, re.DOTALL | re.IGNORECASE)
    return fenced.group(1).strip() if fenced else ""


def _find_json_array(text: str) -> Optional[str]:
    if not text:
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


def extract_think_and_plan(response: str) -> Tuple[str, Optional[List[Dict]]]:
    think = _extract_tagged_block(response, "think")
    answer = _extract_tagged_block(response, "answer")

    candidates = []
    if answer:
        candidates.append(answer)
        fenced_answer = _extract_json_fence(answer)
        if fenced_answer:
            candidates.append(fenced_answer)

    fenced_response = _extract_json_fence(response)
    if fenced_response:
        candidates.append(fenced_response)

    candidates.append(response)

    for candidate in candidates:
        json_array_str = _find_json_array(candidate)
        if not json_array_str:
            continue
        try:
            plan = json.loads(json_array_str)
        except json.JSONDecodeError:
            continue
        if isinstance(plan, list):
            return think, plan

    stats["json_parse_failures"] += 1
    return think, None


def generate_error_prompt(base_prompt: str, chosen_plan: List[Dict], target_errors: List[str]) -> str:
    error_details = []
    for index, error_name in enumerate(target_errors, 1):
        error_def = ERROR_DEFINITIONS[error_name]
        error_details.append(f"{index}. {error_def['name']}: {error_def['description']}")

    return f"""{base_prompt}

[Reference Valid Plan]
This plan is valid and executable. Use it as the starting point for your edit:
```json
{json.dumps(chosen_plan, ensure_ascii=False, indent=2)}
```

[Rejected Plan Construction Task]
Create a PLAUSIBLE but INVALID alternative plan for the exact same task.
- Keep the same JSON schema and step style.
- Reuse or lightly edit the valid plan instead of inventing a totally unrelated one.
- Use only robot names: `chassis_car`, `dog_robot`, `delivery_drone`.
- Do not invent suffixed robot IDs such as `chassis_car_B`.
- The invalid plan must still be a NON-EMPTY plan with at least one step.
- The final plan must intentionally contain ALL of these error families:
{chr(10).join(error_details)}
- Keep the output as a JSON array of steps.
- Do not explain the errors explicitly in natural language.
- Do not output markdown outside the required tags.

[Output Requirements]
1. Put your brief reasoning inside <think>...</think>.
2. Put the final JSON plan inside <answer>...</answer>.
3. The JSON must be parseable.
"""


def select_target_errors(current_counts: Counter) -> List[str]:
    rand_val = random.random()
    if rand_val < 0.6:
        num_errors = 1
    elif rand_val < 0.9:
        num_errors = 2
    else:
        num_errors = 3

    error_types = list(ERROR_DEFINITIONS.keys())
    selected = []
    for _ in range(num_errors):
        if not error_types:
            break
        weights = [1.0 / (current_counts[name] + 1) for name in error_types]
        chosen = random.choices(error_types, weights=weights, k=1)[0]
        selected.append(chosen)
        error_types.remove(chosen)
    return selected


def preview_text(text: str, limit: int = 800) -> str:
    compact = re.sub(r"\s+", " ", text or "").strip()
    return compact[:limit]


def validate_rejected_plan(task_data: Dict, rejected_plan: List[Dict]) -> Dict:
    assert validator is not None
    return validator.validate_plan(
        rejected_plan,
        start_node=task_data["start"],
        end_node=task_data["end"],
        target_item=task_data["item"],
    )


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _fallback_rejected_utility(chosen_utility: float, rejected_feedback: Dict[str, Any]) -> float:
    penalty = _safe_float(rejected_feedback.get("penalty_score"), 0.0)
    trajectory_cost = _safe_float(rejected_feedback.get("trajectory_cost"), chosen_utility)
    # Hard negatives must stay utility-worse than chosen. Some invalid plans can still
    # report a low trajectory_cost, so enforce a lower bound above chosen_utility.
    margin_floor = max(1.0, penalty * 0.25)
    rejected_utility = max(trajectory_cost, chosen_utility + margin_floor)
    return round(rejected_utility, 6)


def _base_pair_payload(
    task_data: Dict[str, Any],
    *,
    pair_type: str,
    chosen_result: Any,
    rejected_plan: List[Dict[str, Any]],
    rejected_feedback: Dict[str, Any],
    rejected_thought: str,
    rejected_response_text: str,
    rejected_source: str,
    target_errors: List[str],
    chosen_utility: float,
    rejected_utility: float,
    generation_metadata_extra: Optional[Dict[str, Any]] = None,
    mutation_notes: Optional[List[str]] = None,
) -> Dict[str, Any]:
    utility_gap = round(rejected_utility - chosen_utility, 6)
    chosen_valid = bool(chosen_result.feedback.get("is_valid"))
    rejected_valid = bool(rejected_feedback.get("is_valid"))
    generation_metadata = {
        "phase": "phase1_generate_openrouter",
        "generator_model": GEMINI_MODEL,
        "chosen_source": "oracle_planner",
        "chosen_reasoning_source": "oracle_reasoning_trace",
        "rejected_attempt_budget": MAX_REJECTED_ATTEMPTS,
        "rejected_fallback_enabled": USE_LLM_REJECTED_FALLBACK,
        "pair_type": pair_type,
    }
    if generation_metadata_extra:
        generation_metadata.update(generation_metadata_extra)

    return {
        "dataset_schema_version": DATASET_SCHEMA_VERSION,
        "pair_type": pair_type,
        "task_id": task_data.get("task_id"),
        "pair_id": build_preference_pair_id(task_data["prompt"], chosen_result.plan, rejected_plan),
        "prompt_version": task_data.get("prompt_version", "graph_prompt_v3"),
        "prompt": task_data["prompt"],
        "instruction": task_data["instruction"],
        "start_node": task_data["start"],
        "end_node": task_data["end"],
        "target_item": task_data["item"],
        "item_profile": task_data["item_profile"],
        "task_metadata": task_data.get("task_metadata", {}),
        "local_graph": task_data.get("local_graph"),
        "reference_paths": task_data.get("reference_paths", []),
        "generation_metadata": generation_metadata,
        "target_errors": target_errors,
        "detected_errors": rejected_feedback.get("error_type", []),
        "chosen_valid": chosen_valid,
        "rejected_valid": rejected_valid,
        "chosen_utility": round(chosen_utility, 6),
        "rejected_utility": round(rejected_utility, 6),
        "utility_gap": utility_gap,
        "chosen": {
            "thought": chosen_result.thought,
            "reasoning_trace": chosen_result.reasoning_trace,
            "plan": chosen_result.plan,
            "response_text": render_plan_response(chosen_result.thought, chosen_result.plan),
            "feedback": chosen_result.feedback,
        },
        "rejected": {
            "thought": rejected_thought,
            "plan": rejected_plan,
            "response_text": rejected_response_text,
            "feedback": rejected_feedback,
            "mutation_notes": mutation_notes or [],
            "rejected_source": rejected_source,
        },
    }


def sample_unique_task(task_gen: TaskGenerator, seen_task_signatures: set[str]) -> Optional[Dict]:
    for _ in range(MAX_TASK_DEDUP_ATTEMPTS):
        task_data = task_gen.generate_task(excluded_task_signatures=seen_task_signatures)
        if not task_data:
            continue

        task_signature = task_data.get("task_metadata", {}).get("task_signature")
        if not task_signature:
            return task_data
        if task_signature in seen_task_signatures:
            stats["duplicate_task_skips"] += 1
            continue
        return task_data
    return None


def build_rejected_candidate(
    task_data: Dict,
    chosen_plan: List[Dict],
    target_errors: List[str],
) -> Tuple[Optional[Dict], Optional[Dict]]:
    assert negative_mutator is not None

    symbolic_result = negative_mutator.build_negative(
        chosen_plan=chosen_plan,
        start_node=task_data["start"],
        end_node=task_data["end"],
        target_item=task_data["item"],
        target_errors=target_errors,
    )
    if symbolic_result:
        stats["symbolic_rejected_successes"] += 1
        log("  [OK] Rejected plan built by symbolic mutator.")
        return (
            {
                "thought": symbolic_result["thought"],
                "plan": symbolic_result["plan"],
                "response_text": render_plan_response(symbolic_result["thought"], symbolic_result["plan"]),
                "mutation_notes": symbolic_result.get("mutation_notes", []),
                "rejected_source": symbolic_result.get("rejected_source", "symbolic_mutator"),
            },
            symbolic_result["feedback"],
        )

    if not USE_LLM_REJECTED_FALLBACK:
        return None, None

    backup_pair = None
    backup_feedback = None

    for attempt in range(1, MAX_REJECTED_ATTEMPTS + 1):
        log(f"  Generating rejected plan (attempt {attempt}/{MAX_REJECTED_ATTEMPTS})...")
        rejected_response = call_model(
            generate_error_prompt(task_data["prompt"], chosen_plan, target_errors),
            temperature=0.6,
            max_tokens=3200,
        )
        if not rejected_response:
            continue

        rejected_think, rejected_plan = extract_think_and_plan(rejected_response)
        if rejected_plan is None:
            log(f"  [FAIL] Could not parse rejected plan. Snippet: {preview_text(rejected_response)}", "WARNING")
            continue
        if not rejected_plan:
            log("  [FAIL] Rejected plan parsed as an empty list.", "WARNING")
            continue

        feedback = validate_rejected_plan(task_data, rejected_plan)
        if feedback.get("is_valid", True):
            stats["rejected_validation_failures"] += 1
            log("  [FAIL] Rejected candidate is still valid under rule validation.", "WARNING")
            log(f"  Response snippet: {preview_text(rejected_response)}", "WARNING")
            continue

        pair = {
            "thought": rejected_think,
            "plan": rejected_plan,
            "response_text": render_plan_response(rejected_think, rejected_plan),
            "rejected_source": "llm_editor",
        }

        detected_errors = set(feedback.get("error_type", []))
        target_set = set(target_errors)
        if target_set.issubset(detected_errors):
            stats["llm_rejected_successes"] += 1
            return pair, feedback

        stats["rejected_target_misses"] += 1
        if backup_pair is None:
            backup_pair = pair
            backup_feedback = feedback
        log(
            f"  [WARN] Rejected plan invalid but target errors mismatched. "
            f"expected={sorted(target_set)} detected={sorted(detected_errors)}",
            "WARNING",
        )

    if backup_pair is not None and backup_feedback is not None:
        if ALLOW_HARD_TARGET_MISMATCH:
            log(
                "  [WARN] Accepting hard-invalid fallback with target mismatch "
                "(PVR_PHASE1_ALLOW_HARD_TARGET_MISMATCH=1).",
                "WARNING",
            )
            return backup_pair, backup_feedback
        stats["hard_target_mismatch_drops"] += 1
        log("  [FAIL] Rejecting hard-invalid fallback due to target mismatch.", "WARNING")

    return None, None


def _build_hard_invalid_pair(
    task_data: Dict[str, Any],
    *,
    target_errors: List[str],
    chosen_result: Any,
) -> Optional[Dict[str, Any]]:
    rejected_pair, rejected_feedback = build_rejected_candidate(task_data, chosen_result.plan, target_errors)
    if not rejected_pair or not rejected_feedback:
        log("  [FAIL] Could not build a usable hard-invalid rejected plan.", "WARNING")
        return None

    chosen_utility = _safe_float(chosen_result.utility_score, _safe_float(chosen_result.feedback.get("trajectory_cost"), 0.0))
    rejected_utility = _fallback_rejected_utility(chosen_utility, rejected_feedback)
    payload = _base_pair_payload(
        task_data,
        pair_type="hard_invalid",
        chosen_result=chosen_result,
        rejected_plan=rejected_pair["plan"],
        rejected_feedback=rejected_feedback,
        rejected_thought=rejected_pair["thought"],
        rejected_response_text=rejected_pair["response_text"],
        rejected_source=rejected_pair.get("rejected_source", "llm_editor"),
        target_errors=target_errors,
        chosen_utility=chosen_utility,
        rejected_utility=rejected_utility,
        mutation_notes=rejected_pair.get("mutation_notes", []),
        generation_metadata_extra={
            "soft_candidate_count": 0,
            "topk_candidates": TOPK_CANDIDATES,
            "topk_paths_per_robot": TOPK_PATHS_PER_ROBOT,
        },
    )
    return payload


def _build_soft_suboptimal_pair(
    task_data: Dict[str, Any],
    *,
    candidates: List[Any],
    chosen_result: Any,
    oracle_planner: OraclePlanner,
) -> Optional[Dict[str, Any]]:
    chosen_utility = _safe_float(chosen_result.utility_score, _safe_float(chosen_result.feedback.get("trajectory_cost"), 0.0))
    fallback_rank = 0
    for idx, rejected_result in enumerate(candidates[1:], start=2):
        fallback_rank = idx
        if rejected_result.plan == chosen_result.plan:
            continue

        rejected_feedback = rejected_result.feedback
        if not rejected_feedback.get("is_valid", False):
            continue
        rejected_utility = _safe_float(
            rejected_result.utility_score,
            _safe_float(rejected_feedback.get("trajectory_cost"), chosen_utility),
        )
        utility_gap = rejected_utility - chosen_utility
        if utility_gap < MIN_SOFT_UTILITY_GAP:
            continue

        soft_thought = (
            "[Soft Negative]\n"
            "- This plan remains executable but is suboptimal under oracle utility.\n"
            f"- Chosen utility={round(chosen_utility, 6)}; rejected utility={round(rejected_utility, 6)}.\n"
            "- Keep this as a preference-learning pair for route ranking instead of validity filtering."
        )
        payload = _base_pair_payload(
            task_data,
            pair_type="soft_suboptimal",
            chosen_result=chosen_result,
            rejected_plan=rejected_result.plan,
            rejected_feedback=rejected_feedback,
            rejected_thought=soft_thought,
            rejected_response_text=render_plan_response(soft_thought, rejected_result.plan),
            rejected_source="oracle_topk_suboptimal",
            target_errors=[],
            chosen_utility=chosen_utility,
            rejected_utility=rejected_utility,
            generation_metadata_extra={
                "soft_candidate_rank": idx,
                "soft_candidate_count": len(candidates),
                "topk_candidates": TOPK_CANDIDATES,
                "topk_paths_per_robot": TOPK_PATHS_PER_ROBOT,
                "min_soft_utility_gap": MIN_SOFT_UTILITY_GAP,
            },
        )
        return payload

    # Fallback: synthesize a valid-but-suboptimal plan by injecting a local detour
    # on the chosen robot path (no extra LLM call).
    chosen_path = list(getattr(chosen_result, "path", []) or [])
    if len(chosen_path) >= 2:
        filtered_graph = oracle_planner._build_robot_filtered_graph(chosen_result.robot)
        for path_idx in range(len(chosen_path) - 1):
            node = chosen_path[path_idx]
            next_node = chosen_path[path_idx + 1]
            prev_node = chosen_path[path_idx - 1] if path_idx > 0 else None
            if node not in filtered_graph:
                continue

            neighbor_candidates = sorted(filtered_graph.neighbors(node))
            for neighbor in neighbor_candidates:
                if neighbor in {next_node, prev_node}:
                    continue
                # Add small reversible loops: node -> neighbor -> node (repeated).
                for loops in (1, 2, 3):
                    detour_segment: List[str] = []
                    for _ in range(loops):
                        detour_segment.extend([neighbor, node])
                    detour_path = chosen_path[: path_idx + 1] + detour_segment + chosen_path[path_idx + 1 :]
                    if detour_path == chosen_path:
                        continue

                    detour_plan = oracle_planner._path_to_plan(
                        chosen_result.robot,
                        detour_path,
                        task_data["item"],
                        task_data["start"],
                        task_data["end"],
                    )
                    rejected_feedback = validate_rejected_plan(task_data, detour_plan)
                    if not rejected_feedback.get("is_valid", False):
                        continue

                    rejected_utility = _safe_float(
                        rejected_feedback.get("trajectory_cost"),
                        chosen_utility + float(loops),
                    )
                    utility_gap = rejected_utility - chosen_utility
                    if utility_gap < MIN_SOFT_UTILITY_GAP:
                        continue

                    soft_thought = (
                        "[Soft Negative]\n"
                        "- This plan remains executable but is suboptimal under oracle utility.\n"
                        f"- Chosen utility={round(chosen_utility, 6)}; rejected utility={round(rejected_utility, 6)}.\n"
                        "- Keep this as a preference-learning pair for route ranking instead of validity filtering."
                    )
                    payload = _base_pair_payload(
                        task_data,
                        pair_type="soft_suboptimal",
                        chosen_result=chosen_result,
                        rejected_plan=detour_plan,
                        rejected_feedback=rejected_feedback,
                        rejected_thought=soft_thought,
                        rejected_response_text=render_plan_response(soft_thought, detour_plan),
                        rejected_source="oracle_detour_suboptimal",
                        target_errors=[],
                        chosen_utility=chosen_utility,
                        rejected_utility=rejected_utility,
                        generation_metadata_extra={
                            "soft_candidate_rank": fallback_rank,
                            "soft_candidate_count": len(candidates),
                            "topk_candidates": TOPK_CANDIDATES,
                            "topk_paths_per_robot": TOPK_PATHS_PER_ROBOT,
                            "min_soft_utility_gap": MIN_SOFT_UTILITY_GAP,
                            "soft_detour_loops": loops,
                        },
                    )
                    return payload
    return None


def generate_dpo_pairs(task_data: Dict, target_errors: List[str], oracle_planner: OraclePlanner) -> List[Dict[str, Any]]:
    candidates = oracle_planner.plan_task_topk(
        start_node=task_data["start"],
        end_node=task_data["end"],
        target_item=task_data["item"],
        k=max(2, TOPK_CANDIDATES),
        max_paths_per_robot=max(1, TOPK_PATHS_PER_ROBOT),
        preferred_robot=task_data.get("preferred_robot_hint"),
        force_preferred_if_feasible=task_data.get("force_preferred_if_feasible", False),
    )
    if not candidates:
        stats["chosen_planner_failures"] += 1
        log("  [FAIL] Oracle planner could not produce any valid candidate.", "WARNING")
        return []

    chosen_result = candidates[0]
    log(
        f"  [OK] Oracle chosen plan generated with {chosen_result.robot} "
        f"({len(chosen_result.plan)} steps, cost={chosen_result.feedback.get('trajectory_cost')}, "
        f"utility={round(_safe_float(chosen_result.utility_score), 6)})."
    )

    pairs: List[Dict[str, Any]] = []

    hard_pair = _build_hard_invalid_pair(
        task_data,
        target_errors=target_errors,
        chosen_result=chosen_result,
    )
    if hard_pair:
        pairs.append(hard_pair)

    if ENABLE_SOFT_SUBOPTIMAL:
        soft_pair = _build_soft_suboptimal_pair(
            task_data,
            candidates=candidates,
            chosen_result=chosen_result,
            oracle_planner=oracle_planner,
        )
    else:
        soft_pair = None
    if soft_pair:
        pairs.append(soft_pair)
    else:
        if ENABLE_SOFT_SUBOPTIMAL:
            log("  [WARN] No usable soft-suboptimal pair found for this task.", "WARNING")

    return pairs


def main() -> None:
    global log_file, validator, negative_mutator

    script_dir = Path(__file__).resolve().parent
    os.chdir(script_dir)
    random.seed(RANDOM_SEED)
    log_file = open(LOG_FILE, "w", encoding="utf-8")

    log("=" * 60)
    log("Phase 1: Generate graph-grounded preference pairs")
    log("=" * 60)
    log(f"Target samples: {NUM_SAMPLES}")
    log(f"Model: {GEMINI_MODEL}")
    log(f"Random seed: {RANDOM_SEED}")
    log(f"Rejected retries: {MAX_REJECTED_ATTEMPTS}")
    log(f"Task dedup retries: {MAX_TASK_DEDUP_ATTEMPTS}")
    log(f"Max global attempts: {MAX_GLOBAL_ATTEMPTS}")
    log(f"Include resource_conflict targets: {INCLUDE_RESOURCE_CONFLICT}")
    log(f"LLM rejected fallback enabled: {USE_LLM_REJECTED_FALLBACK}")
    log(f"Allow hard target mismatch fallback: {ALLOW_HARD_TARGET_MISMATCH}")
    log(f"Top-k candidates per task: {TOPK_CANDIDATES}")
    log(f"Top-k paths per robot: {TOPK_PATHS_PER_ROBOT}")
    log(f"Min soft utility gap: {MIN_SOFT_UTILITY_GAP}")
    log(f"Enable soft-suboptimal pairs: {ENABLE_SOFT_SUBOPTIMAL}")
    log("=" * 60)

    engine = build_default_graph_engine(script_dir)
    validator = SymbolicValidator(engine)
    negative_mutator = NegativePlanMutator(engine=engine, validator=validator)
    oracle_planner = OraclePlanner(
        engine,
        validator=validator,
        topk_paths_per_robot=max(1, TOPK_PATHS_PER_ROBOT),
    )
    task_gen = TaskGenerator(engine)
    seen_task_signatures: set[str] = set()

    output_file = open(OUTPUT_FILE, "w", encoding="utf-8")

    with tqdm(total=NUM_SAMPLES, desc="Generating") as pbar:
        while stats["successful_generations"] < NUM_SAMPLES:
            if stats["total_attempts"] >= MAX_GLOBAL_ATTEMPTS:
                log(
                    f"[STOP] Reached max global attempts ({MAX_GLOBAL_ATTEMPTS}) "
                    "before hitting target sample count.",
                    "ERROR",
                )
                break
            stats["total_attempts"] += 1
            target_errors = select_target_errors(stats["error_distribution"])

            log(f"\n--- Sample {stats['successful_generations'] + 1}/{NUM_SAMPLES} ---")
            log(f"Target errors: {', '.join(target_errors)}")

            task_data = sample_unique_task(task_gen, seen_task_signatures)
            if not task_data:
                log("  [SKIP] Task generation failed or could not find a unique task signature.", "WARNING")
                continue

            dpo_pairs = generate_dpo_pairs(task_data, target_errors, oracle_planner)
            if dpo_pairs:
                task_signature = task_data.get("task_metadata", {}).get("task_signature")
                if task_signature:
                    seen_task_signatures.add(task_signature)
                for dpo_pair in dpo_pairs:
                    if stats["successful_generations"] >= NUM_SAMPLES:
                        break
                    output_file.write(json.dumps(dpo_pair, ensure_ascii=False) + "\n")
                    output_file.flush()
                    stats["successful_generations"] += 1
                    pair_type = str(dpo_pair.get("pair_type", "unknown"))
                    stats["pair_type_distribution"][pair_type] += 1
                    if pair_type == "hard_invalid":
                        stats["hard_invalid_successes"] += 1
                        for error_name in dpo_pair.get("target_errors", []):
                            stats["error_distribution"][error_name] += 1
                    elif pair_type == "soft_suboptimal":
                        stats["soft_suboptimal_successes"] += 1
                    pbar.update(1)
                    log(f"  [SUCCESS] Generated {pair_type} pair.", "SUCCESS")
                if COOLDOWN_SECONDS > 0:
                    log(f"  Cooling down for {COOLDOWN_SECONDS} seconds to reduce rate limits...")
                    time.sleep(COOLDOWN_SECONDS)

    output_file.close()

    log("\n" + "=" * 60)
    log("Generation complete")
    log("=" * 60)
    log(f"Total attempts: {stats['total_attempts']}")
    log(f"Successful generations: {stats['successful_generations']}")
    if stats["total_attempts"]:
        success_rate = stats["successful_generations"] / stats["total_attempts"] * 100
        log(f"Success rate: {success_rate:.1f}%")
    log(f"API failures: {stats['api_failures']}")
    log(f"JSON parse failures: {stats['json_parse_failures']}")
    log(f"Chosen planner failures: {stats['chosen_planner_failures']}")
    log(f"Rejected validation failures: {stats['rejected_validation_failures']}")
    log(f"Rejected target misses: {stats['rejected_target_misses']}")
    log(f"Symbolic rejected successes: {stats['symbolic_rejected_successes']}")
    log(f"LLM rejected successes: {stats['llm_rejected_successes']}")
    log(f"Duplicate task skips: {stats['duplicate_task_skips']}")
    log(f"Hard-target mismatch drops: {stats['hard_target_mismatch_drops']}")
    log(f"Hard-invalid pairs: {stats['hard_invalid_successes']}")
    log(f"Soft-suboptimal pairs: {stats['soft_suboptimal_successes']}")
    log("Pair-type distribution:")
    for pair_name, count in stats["pair_type_distribution"].items():
        log(f"  {pair_name}: {count}")
    log("Error distribution:")
    for error_name, count in stats["error_distribution"].items():
        log(f"  {error_name}: {count}")
    log(f"Output saved to: {OUTPUT_FILE}")
    log("=" * 60)

    log_file.close()


if __name__ == "__main__":
    main()
