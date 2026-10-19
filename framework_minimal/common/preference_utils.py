from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from typing import Any, Dict, List

from planning_schema import ACTION_SPECS, AGENT_PROFILES


MAJOR_ERROR_TYPES = {
    "capability_mismatch",
    "morphology_constraint",
    "graph_infeasible",
    "coordination_failure",
    "terminal_failure",
}
MINOR_ERROR_TYPES = {"resource_conflict"}

MARGIN_BUCKETS = [
    {"name": "none", "min_score": 0.0, "max_score": 0.0, "margin": 0.0},
    {"name": "light", "min_score": 0.0, "max_score": 5.0, "margin": 0.15},
    {"name": "moderate", "min_score": 5.0, "max_score": 10.0, "margin": 0.35},
    {"name": "severe", "min_score": 10.0, "max_score": 15.0, "margin": 0.7},
    {"name": "critical", "min_score": 15.0, "max_score": float("inf"), "margin": 1.1},
]

SOFT_UTILITY_BUCKETS = [
    {"name": "none", "min_gap": 0.0, "max_gap": 0.0, "margin": 0.0},
    {"name": "light", "min_gap": 0.0, "max_gap": 1.0, "margin": 0.15},
    {"name": "moderate", "min_gap": 1.0, "max_gap": 3.0, "margin": 0.35},
    {"name": "severe", "min_gap": 3.0, "max_gap": 6.0, "margin": 0.7},
    {"name": "critical", "min_gap": 6.0, "max_gap": float("inf"), "margin": 1.1},
]


def canonical_json_dumps(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True)


def render_plan_response(thought: str, plan: List[Dict[str, Any]]) -> str:
    thought_block = (thought or "").strip()
    plan_block = json.dumps(plan, ensure_ascii=False, indent=2)
    return f"<think>\n{thought_block}\n</think>\n\n<answer>\n{plan_block}\n</answer>"


def _serialize_action_schema() -> Dict[str, Dict[str, Any]]:
    action_schema: Dict[str, Dict[str, Any]] = {}
    for name, spec in ACTION_SPECS.items():
        serializable_spec = {}
        for key, value in spec.items():
            if isinstance(value, set):
                serializable_spec[key] = sorted(value)
            elif isinstance(value, tuple):
                serializable_spec[key] = list(value)
            else:
                serializable_spec[key] = value
        action_schema[name] = serializable_spec
    return action_schema


def build_planning_prompt(
    instruction: str,
    scenario_json: Dict[str, Any],
    target_item: str,
    item_profile: Dict[str, Any],
    start_node: str,
    end_node: str,
    reference_paths: List[List[str]],
) -> str:
    scenario_json = scenario_json or {}
    reference_paths = reference_paths or []
    action_schema = _serialize_action_schema()

    return f"""You are a planner for heterogeneous embodied agents operating on a symbolic campus graph.

**Goal**
- Produce a valid executable transport plan.
- Prefer lower traversal cost, fewer unnecessary handovers, and fewer agents when possible.

**Robot Profiles**
1. `chassis_car`
   - Capabilities: {sorted(AGENT_PROFILES['chassis_car']['capabilities'])}
   - Max load: {AGENT_PROFILES['chassis_car']['max_load_kg']} kg
   - Can use elevators.
   - Cannot use stairs.

2. `dog_robot`
   - Capabilities: {sorted(AGENT_PROFILES['dog_robot']['capabilities'])}
   - Max load: {AGENT_PROFILES['dog_robot']['max_load_kg']} kg
   - Can use stairs.
   - CANNOT use elevators or operate doors/access panels.

3. `delivery_drone`
   - Capabilities: {sorted(AGENT_PROFILES['delivery_drone']['capabilities'])}
   - Max load: {AGENT_PROFILES['delivery_drone']['max_load_kg']} kg
   - Can only move on `AerialLink` edges between outdoor-compatible interface nodes.
   - CANNOT enter Indoor / Room / Elevator / Stairs nodes.

**World Semantics**
- Building macro nodes (e.g. `E1`, `Library`, `Activity_Center`) are outdoor-compatible interface nodes.
- `navigate` is only for a SINGLE adjacent edge whose `edge_type` is `FlatGround`, `Virtual_Entry`, or `AerialLink`.
- `call_elevator` is only for a SINGLE adjacent edge whose `edge_type` is `Elevator`.
- `climb_stairs` is only for a SINGLE adjacent edge whose `edge_type` is `Stairs`.
- Every movement step must use an adjacent `(source, target)` pair from `local_edges`.
- `handover` requires both robots to be at the same node.

**Task**
- Instruction: {instruction}
- Start node: `{start_node}`
- End node: `{end_node}`
- Target item: `{target_item}`
- Item profile: {json.dumps(item_profile, ensure_ascii=False)}
- The item starts EXACTLY at `{start_node}`.
- The delivery destination is EXACTLY `{end_node}`.
- `pick_up.args.node` must equal `{start_node}`.
- The final successful `put_down.args.node` must equal `{end_node}`.
- Do NOT replace the true start node with a nearby lobby, room, or building interface node.

**Local Graph**
{json.dumps(scenario_json, ensure_ascii=False, indent=2)}

**Reference Paths**
These are example shortest valid node sequences between start and end. They are hints, not mandatory full plans:
{json.dumps(reference_paths, ensure_ascii=False, indent=2)}

**Action Schema**
Use only these actions:
{json.dumps(action_schema, ensure_ascii=False, indent=2)}

**Output Requirements**
1. Put SHORT reasoning inside `<think>...</think>`.
2. Keep `<think>` to at most 6 bullet lines.
3. Do NOT mention any external validator, rule engine, or claim the plan has already been verified.
4. Put the final plan inside `<answer>...</answer>`.
5. The final plan must be a pure JSON array.
6. Every step must have:
   - `id`
   - `robot`
   - `action`
   - `args`
   - `depends_on`
7. Use only robot names: `chassis_car`, `dog_robot`, `delivery_drone`.
8. Use only action names: `navigate`, `call_elevator`, `climb_stairs`, `pick_up`, `put_down`, `handover`.
9. `navigate`, `call_elevator`, `climb_stairs` must include `args.source` and `args.target`.
10. `pick_up` and `put_down` must include `args.item` and `args.node`.
11. `handover` must include `args.item`, `args.node`, and `args.target` (the receiving robot).

**Hard Constraints**
- The FIRST step must be `pick_up` at `{start_node}`.
- The FINAL step must be `put_down` at `{end_node}`.
- No step may move across a non-adjacent edge.
- Elevator and stair movement must also be adjacent; never jump from `4F` directly to `1F` in one step.
- The drone must stay on macro outdoor-compatible nodes only.
- The dog must never use elevator edges.
- The chassis must never use stairs edges.
- An item can only be carried by robots whose payload capacity is high enough.
- Avoid planning around non-transportable fixtures.
- The final state is valid only when the item is delivered to the exact end node, not an adjacent node.

**Output Template**
<think>
- Choose robot: ...
- Route sketch: ...
- Key constraints: ...
</think>

<answer>
[
  {{
    "id": "t1",
    "robot": "chassis_car",
    "action": "pick_up",
    "args": {{"item": "{target_item}", "node": "{start_node}"}},
    "depends_on": []
  }}
]
</answer>
"""


def render_oracle_reasoning_trace(reasoning_trace: Dict[str, Any]) -> str:
    grounding = reasoning_trace["task_grounding"]
    selection = reasoning_trace["selection"]
    route = reasoning_trace["route_decomposition"]
    constraint = reasoning_trace["constraint_check"]

    candidate_parts = []
    for candidate in reasoning_trace["candidate_scan"]:
        robot = candidate["robot"]
        status = candidate["status"]
        if status == "feasible":
            candidate_parts.append(
                f"{robot}=feasible(cost={candidate.get('trajectory_cost')}, steps={candidate.get('num_steps')})"
            )
        else:
            candidate_parts.append(f"{robot}=infeasible({candidate.get('reason', 'unknown')})")

    move_actions = [move["action"] for move in route["movement_steps"]]
    if move_actions:
        movement_summary = ", ".join(move_actions[:4])
        if len(move_actions) > 4:
            movement_summary += ", ..."
    else:
        movement_summary = "none"

    return (
        "[Task]\n"
        f"- Deliver '{grounding['target_item']}' from {grounding['start_node']} to {grounding['end_node']}.\n"
        f"- Item profile: {grounding['item_profile']}.\n"
        "[Choice]\n"
        f"- Selected robot: {selection['selected_robot']}.\n"
        f"- Candidate summary: {'; '.join(candidate_parts)}.\n"
        f"- Chosen path nodes: {selection['selected_path_nodes']}.\n"
        "[Plan]\n"
        f"- First step: pick_up at {grounding['start_node']}; final step: put_down at {grounding['end_node']}.\n"
        f"- Movement sequence: {movement_summary}; total plan steps={selection['selected_plan_steps']}; trajectory cost={selection['selected_trajectory_cost']}.\n"
        "[Checks]\n"
        f"- Payload fits selected robot: {constraint['payload_ok']}; exact pickup={constraint['exact_pickup_node_ok']}; exact put_down={constraint['exact_putdown_node_ok']}."
    )


def _hash_payload(payload: Dict[str, Any], prefix: str) -> str:
    digest = hashlib.sha1(canonical_json_dumps(payload).encode("utf-8")).hexdigest()[:16]
    return f"{prefix}_{digest}"


def build_task_id(prompt: str, start_node: str, end_node: str, target_item: str) -> str:
    return _hash_payload(
        {
            "prompt": prompt,
            "start_node": start_node,
            "end_node": end_node,
            "target_item": target_item,
        },
        prefix="task",
    )


def build_preference_pair_id(
    prompt: str,
    chosen_plan: List[Dict[str, Any]],
    rejected_plan: List[Dict[str, Any]],
) -> str:
    return _hash_payload(
        {
            "prompt": prompt,
            "chosen_plan": chosen_plan,
            "rejected_plan": rejected_plan,
        },
        prefix="pair",
    )


def extract_building_id(node_id: str) -> str:
    if not isinstance(node_id, str) or not node_id:
        return "UNKNOWN"

    macro_match = re.match(r"^([WE]\d)(?:_|$)", node_id)
    if macro_match:
        return macro_match.group(1)

    if "_" not in node_id:
        return node_id

    return node_id.split("_", 1)[0]


def derive_task_metadata(start_node: str, end_node: str, target_item: str) -> Dict[str, Any]:
    start_building = extract_building_id(start_node)
    end_building = extract_building_id(end_node)
    return {
        "start_building": start_building,
        "end_building": end_building,
        "building_pair": f"{start_building}->{end_building}",
        "target_item": target_item,
        "task_signature": f"{start_node}|{end_node}|{target_item}",
    }


def _pick_margin_bucket(severity_score: float) -> Dict[str, Any]:
    if severity_score <= 0:
        return MARGIN_BUCKETS[0]

    for bucket in MARGIN_BUCKETS[1:]:
        if bucket["min_score"] <= severity_score < bucket["max_score"]:
            return bucket
    return MARGIN_BUCKETS[-1]


def _pick_soft_utility_bucket(utility_gap: float) -> Dict[str, Any]:
    if utility_gap <= 0:
        return SOFT_UTILITY_BUCKETS[0]

    for bucket in SOFT_UTILITY_BUCKETS[1:]:
        if bucket["min_gap"] <= utility_gap < bucket["max_gap"]:
            return bucket
    return SOFT_UTILITY_BUCKETS[-1]


def compute_preference_strength(
    chosen_feedback: Dict[str, Any],
    rejected_feedback: Dict[str, Any],
    *,
    pair_type: str = "hard_invalid",
    chosen_utility: float | None = None,
    rejected_utility: float | None = None,
) -> Dict[str, Any]:
    chosen_penalty = float(chosen_feedback.get("penalty_score", 0.0) or 0.0)
    rejected_penalty = float(rejected_feedback.get("penalty_score", 0.0) or 0.0)

    breakdown = rejected_feedback.get("violation_breakdown", {}) or {}
    if not isinstance(breakdown, dict):
        breakdown = {}

    error_counter = Counter(breakdown)
    major_count = sum(count for name, count in error_counter.items() if name in MAJOR_ERROR_TYPES)
    minor_count = sum(count for name, count in error_counter.items() if name in MINOR_ERROR_TYPES)
    total_count = sum(error_counter.values())

    chosen_cost = float(chosen_feedback.get("trajectory_cost", 0.0) or 0.0)
    rejected_cost = float(rejected_feedback.get("trajectory_cost", 0.0) or 0.0)

    norm_pair_type = pair_type if isinstance(pair_type, str) and pair_type else "hard_invalid"
    chosen_utility_value = float(chosen_utility if chosen_utility is not None else chosen_cost)
    rejected_utility_value = float(rejected_utility if rejected_utility is not None else rejected_cost)
    utility_gap = max(0.0, rejected_utility_value - chosen_utility_value)

    if norm_pair_type == "soft_suboptimal":
        severity_score = utility_gap
        bucket = _pick_soft_utility_bucket(utility_gap)
        severity_source = "utility_gap"
    else:
        severity_score = max(0.0, rejected_penalty - chosen_penalty)
        bucket = _pick_margin_bucket(severity_score)
        severity_source = "penalty_gap"

    return {
        "pair_type": norm_pair_type,
        "chosen_penalty_score": chosen_penalty,
        "rejected_penalty_score": rejected_penalty,
        "severity_score": severity_score,
        "severity_source": severity_source,
        "severity_level": bucket["name"],
        "severity_margin": bucket["margin"],
        "major_violation_count": int(major_count),
        "minor_violation_count": int(minor_count),
        "total_violation_count": int(total_count),
        "chosen_trajectory_cost": chosen_cost,
        "rejected_trajectory_cost": rejected_cost,
        "trajectory_cost_gap": round(rejected_cost - chosen_cost, 3),
        "chosen_utility": round(chosen_utility_value, 6),
        "rejected_utility": round(rejected_utility_value, 6),
        "utility_gap": round(utility_gap, 6),
        "error_types": sorted(error_counter.keys()),
        "violation_breakdown": dict(error_counter),
    }
