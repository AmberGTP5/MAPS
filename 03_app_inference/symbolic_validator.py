"""
Graph-grounded symbolic validator for multi-robot transport plans.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional

from graph_engine import EmbodiedGraphEngine, build_default_graph_engine
from planning_schema import (
    ACTION_SPECS,
    AGENT_PROFILES,
    DOOR_CONSTRAINTS,
    OBJECT_CATALOG,
    get_item_profile,
)


ACTION_ALIASES = {
    "Maps": "navigate",
    "enter_elevator": "call_elevator",
    "exit_elevator": "navigate",
}

SEVERITY_WEIGHTS = {
    "capability_mismatch": 5,
    "morphology_constraint": 5,
    "graph_infeasible": 5,
    "coordination_failure": 5,
    "resource_conflict": 2,
    "terminal_failure": 5,
}


class SymbolicValidator:
    def __init__(self, engine: Optional[EmbodiedGraphEngine] = None):
        self.engine = engine or build_default_graph_engine(Path(__file__).resolve().parent)
        self.agent_profiles = AGENT_PROFILES
        self.object_catalog = OBJECT_CATALOG

    def _build_default_engine(self) -> EmbodiedGraphEngine:
        base_dir = Path(__file__).resolve().parent
        return EmbodiedGraphEngine(
            macro_node_file=str(base_dir / "宏观建筑物节点信息.json"),
            micro_node_files=[
                str(base_dir / "W1-W4节点信息.json"),
                str(base_dir / "E1-E4节点信息.json"),
            ],
            edge_file=str(base_dir / "所有建筑物边信息.json"),
        )

    def validate_plan(
        self,
        plan: List[Dict],
        start_node: str = None,
        end_node: str = None,
        target_item: str = None,
    ) -> Dict:
        all_errors: List[Dict] = []
        state = {
            "agent_carrying": defaultdict(list),
            "agent_locations": {},
            "item_locations": {},
            "trajectory_cost": 0.0,
        }

        if target_item and start_node:
            state["item_locations"][target_item] = start_node

        for index, raw_step in enumerate(plan):
            step = self._normalize_step(raw_step, index)
            step_errors = self._validate_step(step, state)
            all_errors.extend(step_errors)

        all_errors.extend(self.check_resource_conflict(plan))

        if end_node and target_item:
            terminal_error = self.check_terminal_state(state, end_node, target_item)
            if terminal_error:
                all_errors.append(terminal_error)

        if not all_errors:
            return {
                "is_valid": True,
                "message": "Plan passed graph-grounded symbolic validation",
                "error_type": [],
                "error_details": "",
                "failed_steps": [],
                "penalty_score": 0,
                "trajectory_cost": round(state["trajectory_cost"], 3),
                "violation_breakdown": {},
                "validation_source": "rule_engine",
            }

        breakdown = Counter(err["error_type"] for err in all_errors)
        penalty_score = sum(SEVERITY_WEIGHTS.get(err["error_type"], 5) for err in all_errors)

        return {
            "is_valid": False,
            "error_type": sorted(breakdown.keys()),
            "error_details": " | ".join(err["error_details"] for err in all_errors),
            "failed_steps": [err["failed_step"] for err in all_errors],
            "penalty_score": penalty_score,
            "trajectory_cost": round(state["trajectory_cost"], 3),
            "violation_breakdown": dict(breakdown),
            "validation_source": "rule_engine",
        }

    def _normalize_step(self, step: Dict, index: int) -> Dict:
        robot = step.get("robot") or step.get("agent")
        action = ACTION_ALIASES.get(step.get("action"), step.get("action"))
        args = step.get("args") or step.get("parameters") or {}
        return {
            "id": step.get("id", f"t{index + 1}"),
            "step_num": step.get("step", index + 1),
            "robot": robot,
            "action": action,
            "args": args,
            "depends_on": step.get("depends_on", []),
        }

    def _error(self, error_type: str, detail: str, step_num: int) -> Dict:
        return {
            "error_type": error_type,
            "error_details": detail,
            "failed_step": step_num,
        }

    def _validate_step(self, step: Dict, state: Dict) -> List[Dict]:
        errors: List[Dict] = []
        robot = step["robot"]
        action = step["action"]
        args = step["args"]
        step_num = step["step_num"]

        if robot not in self.agent_profiles:
            return [self._error("capability_mismatch", f"Unknown robot '{robot}'.", step_num)]

        if action not in ACTION_SPECS:
            return [self._error("capability_mismatch", f"Unknown action '{action}'.", step_num)]

        required_args = ACTION_SPECS[action].get("required_args", ())
        for arg_name in required_args:
            if arg_name not in args:
                errors.append(
                    self._error(
                        "graph_infeasible",
                        f"Action '{action}' is missing required arg '{arg_name}'.",
                        step_num,
                    )
                )
        if errors:
            return errors

        if action in {"navigate", "call_elevator", "climb_stairs"}:
            errors.extend(self._validate_movement(step, state))
        elif action == "pick_up":
            errors.extend(self._validate_pick_up(step, state))
        elif action == "put_down":
            errors.extend(self._validate_put_down(step, state))
        elif action == "handover":
            errors.extend(self._validate_handover(step, state))

        return errors

    def _validate_movement(self, step: Dict, state: Dict) -> List[Dict]:
        errors: List[Dict] = []
        robot = step["robot"]
        action = step["action"]
        args = step["args"]
        step_num = step["step_num"]
        source = args["source"]
        target = args["target"]

        if source not in self.engine.G or target not in self.engine.G:
            return [
                self._error(
                    "graph_infeasible",
                    f"Movement uses undefined node(s): {source} -> {target}.",
                    step_num,
                )
            ]

        current_location = state["agent_locations"].get(robot)
        if current_location and current_location != source:
            errors.append(
                self._error(
                    "coordination_failure",
                    f"{robot} claims source {source} but is currently at {current_location}.",
                    step_num,
                )
            )

        edge_data = self.engine.get_edge_data(source, target)
        if not edge_data:
            return [
                self._error(
                    "graph_infeasible",
                    f"No graph edge exists between {source} and {target}.",
                    step_num,
                )
            ]

        allowed_edge_types = ACTION_SPECS[action]["edge_types"]
        edge_type = edge_data.get("edge_type")
        if edge_type not in allowed_edge_types:
            errors.append(
                self._error(
                    "graph_infeasible",
                    f"Action '{action}' cannot traverse edge type '{edge_type}' ({source} -> {target}).",
                    step_num,
                )
            )

        traversable_by = set(edge_data.get("traversable_by", []))
        if robot not in traversable_by:
            errors.append(
                self._error(
                    "morphology_constraint",
                    f"{robot} is not allowed on edge {source} -> {target} ({edge_type}).",
                    step_num,
                )
            )

        target_type = self.engine.nodes_data.get(target, {}).get("type", "Unknown")
        if target_type not in self.agent_profiles[robot]["allowed_node_types"]:
            errors.append(
                self._error(
                    "morphology_constraint",
                    f"{robot} cannot occupy node type '{target_type}' at {target}.",
                    step_num,
                )
            )

        constraints = set(edge_data.get("constraints", []))
        if robot != "chassis_car" and constraints & DOOR_CONSTRAINTS:
            errors.append(
                self._error(
                    "capability_mismatch",
                    f"{robot} cannot satisfy door/access constraints on {source} -> {target}.",
                    step_num,
                )
            )

        if action == "call_elevator" and robot != "chassis_car":
            errors.append(
                self._error(
                    "morphology_constraint",
                    f"{robot} cannot use elevator edges.",
                    step_num,
                )
            )

        if action == "climb_stairs" and robot != "dog_robot":
            errors.append(
                self._error(
                    "morphology_constraint",
                    f"{robot} cannot use stair edges.",
                    step_num,
                )
            )

        if robot == "delivery_drone":
            source_type = self.engine.nodes_data.get(source, {}).get("type", "Unknown")
            if source_type not in {"Building", "Outdoor"} or target_type not in {"Building", "Outdoor"}:
                errors.append(
                    self._error(
                        "morphology_constraint",
                        f"delivery_drone must stay on outdoor-compatible interface nodes ({source} -> {target}).",
                        step_num,
                    )
                )

        if not errors:
            state["agent_locations"][robot] = target
            state["trajectory_cost"] += float(edge_data.get("weight", 0.0))

        return errors

    def _validate_pick_up(self, step: Dict, state: Dict) -> List[Dict]:
        robot = step["robot"]
        args = step["args"]
        step_num = step["step_num"]
        node = args["node"]
        item = args["item"]
        errors: List[Dict] = []

        if node not in self.engine.G:
            return [self._error("graph_infeasible", f"Unknown pick_up node '{node}'.", step_num)]

        current_location = state["agent_locations"].get(robot)
        if current_location and current_location != node:
            errors.append(
                self._error(
                    "coordination_failure",
                    f"{robot} attempts pick_up at {node} while located at {current_location}.",
                    step_num,
                )
            )

        profile = get_item_profile(item)
        if not profile.get("transportable", False):
            errors.append(
                self._error(
                    "morphology_constraint",
                    f"Item '{item}' is marked non-transportable.",
                    step_num,
                )
            )

        if profile.get("weight_kg", 0.0) > self.agent_profiles[robot]["max_load_kg"]:
            errors.append(
                self._error(
                    "morphology_constraint",
                    f"{robot} exceeds payload limit when picking up '{item}'.",
                    step_num,
                )
            )

        target_type = self.engine.nodes_data.get(node, {}).get("type", "Unknown")
        if robot == "delivery_drone" and target_type not in {"Building", "Outdoor"}:
            errors.append(
                self._error(
                    "morphology_constraint",
                    f"delivery_drone cannot pick up items inside node type '{target_type}'.",
                    step_num,
                )
            )

        known_item_location = state["item_locations"].get(item)
        if known_item_location and known_item_location != node:
            errors.append(
                self._error(
                    "coordination_failure",
                    f"Item '{item}' is tracked at {known_item_location}, not at {node}.",
                    step_num,
                )
            )

        if not errors:
            state["agent_locations"][robot] = node
            state["agent_carrying"][robot].append(item)
            state["item_locations"][item] = f"carried_by_{robot}"

        return errors

    def _validate_put_down(self, step: Dict, state: Dict) -> List[Dict]:
        robot = step["robot"]
        args = step["args"]
        step_num = step["step_num"]
        node = args["node"]
        item = args["item"]
        errors: List[Dict] = []

        current_location = state["agent_locations"].get(robot)
        if current_location and current_location != node:
            errors.append(
                self._error(
                    "coordination_failure",
                    f"{robot} attempts put_down at {node} while located at {current_location}.",
                    step_num,
                )
            )

        if item not in state["agent_carrying"][robot]:
            errors.append(
                self._error(
                    "coordination_failure",
                    f"{robot} tries to put down '{item}' without carrying it.",
                    step_num,
                )
            )

        if not errors:
            state["agent_locations"][robot] = node
            state["agent_carrying"][robot].remove(item)
            state["item_locations"][item] = node

        return errors

    def _validate_handover(self, step: Dict, state: Dict) -> List[Dict]:
        robot = step["robot"]
        args = step["args"]
        step_num = step["step_num"]
        node = args["node"]
        item = args["item"]
        receiver = args["target"]
        errors: List[Dict] = []

        if receiver not in self.agent_profiles:
            return [
                self._error(
                    "capability_mismatch",
                    f"Unknown handover receiver '{receiver}'.",
                    step_num,
                )
            ]

        giver_location = state["agent_locations"].get(robot)
        receiver_location = state["agent_locations"].get(receiver)
        if giver_location != node:
            errors.append(
                self._error(
                    "coordination_failure",
                    f"{robot} must be at {node} for handover but is at {giver_location}.",
                    step_num,
                )
            )
        if receiver_location != node:
            errors.append(
                self._error(
                    "coordination_failure",
                    f"{receiver} must be at {node} for handover but is at {receiver_location}.",
                    step_num,
                )
            )

        if item not in state["agent_carrying"][robot]:
            errors.append(
                self._error(
                    "coordination_failure",
                    f"{robot} tries to hand over '{item}' without carrying it.",
                    step_num,
                )
            )

        item_profile = get_item_profile(item)
        if item_profile.get("weight_kg", 0.0) > self.agent_profiles[receiver]["max_load_kg"]:
            errors.append(
                self._error(
                    "morphology_constraint",
                    f"{receiver} exceeds payload limit when receiving '{item}'.",
                    step_num,
                )
            )

        if not errors:
            state["agent_carrying"][robot].remove(item)
            state["agent_carrying"][receiver].append(item)
            state["item_locations"][item] = f"carried_by_{receiver}"

        return errors

    def check_resource_conflict(self, plan: List[Dict]) -> List[Dict]:
        errors = []
        elevator_calls = []
        for index, raw_step in enumerate(plan):
            step = self._normalize_step(raw_step, index)
            if step["action"] == "call_elevator":
                elevator_calls.append(step)

        for i, step1 in enumerate(elevator_calls):
            for step2 in elevator_calls[i + 1 :]:
                if step1["robot"] == step2["robot"]:
                    continue
                if step1.get("depends_on") or step2.get("depends_on"):
                    continue
                if step1["args"].get("source") == step2["args"].get("source"):
                    errors.append(
                        self._error(
                            "resource_conflict",
                            f"{step1['robot']} and {step2['robot']} compete for elevator at "
                            f"{step1['args'].get('source')} without dependency ordering.",
                            step1["step_num"],
                        )
                    )
        return errors

    def check_terminal_state(self, state: Dict, end_node: str, target_item: str) -> Optional[Dict]:
        item_loc = state["item_locations"].get(target_item)
        if item_loc == end_node:
            return None

        if isinstance(item_loc, str) and item_loc.startswith("carried_by_"):
            carrier = item_loc.replace("carried_by_", "", 1)
            if state["agent_locations"].get(carrier) == end_node:
                return None

        return self._error(
            "terminal_failure",
            f"Task ended but '{target_item}' is at {item_loc}; expected delivery to {end_node}.",
            999,
        )


if __name__ == "__main__":
    validator = SymbolicValidator()
    test_plan = [
        {
            "id": "t1",
            "robot": "delivery_drone",
            "action": "navigate",
            "args": {"source": "E1_1F_Lobby", "target": "E1"},
            "depends_on": [],
        },
        {
            "id": "t2",
            "robot": "dog_robot",
            "action": "call_elevator",
            "args": {"source": "W1_Elevator_1F", "target": "W1_Elevator_2F"},
            "depends_on": [],
        },
    ]
    result = validator.validate_plan(
        test_plan,
        start_node="E1_1F_Lobby",
        end_node="W1_2F_Office",
        target_item="document",
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))
