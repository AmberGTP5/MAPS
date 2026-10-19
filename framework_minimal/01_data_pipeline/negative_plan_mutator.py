from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple

from graph_engine import EmbodiedGraphEngine
from planning_schema import DOOR_CONSTRAINTS
from symbolic_validator import SymbolicValidator


class NegativePlanMutator:
    def __init__(
        self,
        engine: EmbodiedGraphEngine,
        validator: SymbolicValidator,
    ):
        self.engine = engine
        self.validator = validator

    def build_negative(
        self,
        chosen_plan: List[Dict[str, Any]],
        start_node: str,
        end_node: str,
        target_item: str,
        target_errors: List[str],
    ) -> Optional[Dict[str, Any]]:
        if not chosen_plan:
            return None

        mutation_steps: List[Tuple[int, Dict[str, Any], str]] = []
        mutation_notes: List[str] = []
        working_plan = deepcopy(chosen_plan)

        for error_name in target_errors:
            mutation = self._build_error_injection(
                error_name=error_name,
                base_plan=working_plan,
                start_node=start_node,
                end_node=end_node,
                target_item=target_item,
            )
            if mutation is None:
                return None

            insert_after_index, injected_step, note = mutation
            mutation_steps.append((insert_after_index, injected_step, note))
            mutation_notes.append(note)

        rejected_plan = self._apply_mutations(working_plan, mutation_steps)
        feedback = self.validator.validate_plan(
            rejected_plan,
            start_node=start_node,
            end_node=end_node,
            target_item=target_item,
        )
        detected_errors = set(feedback.get("error_type", []))
        target_set = set(target_errors)
        if feedback.get("is_valid", True) or not target_set.issubset(detected_errors):
            return None

        reasoning = self._render_reasoning(
            target_item=target_item,
            target_errors=target_errors,
            mutation_notes=mutation_notes,
            detected_errors=sorted(detected_errors),
        )

        return {
            "thought": reasoning,
            "plan": rejected_plan,
            "feedback": feedback,
            "mutation_notes": mutation_notes,
            "rejected_source": "symbolic_mutator",
        }

    def _build_error_injection(
        self,
        error_name: str,
        base_plan: List[Dict[str, Any]],
        start_node: str,
        end_node: str,
        target_item: str,
    ) -> Optional[Tuple[int, Dict[str, Any], str]]:
        if error_name == "coordination_failure":
            return self._coordination_injection(base_plan, end_node, target_item)
        if error_name == "morphology_constraint":
            return self._morphology_injection(base_plan)
        if error_name == "capability_mismatch":
            return self._capability_injection(base_plan)
        return None

    def _coordination_injection(
        self,
        base_plan: List[Dict[str, Any]],
        end_node: str,
        target_item: str,
    ) -> Optional[Tuple[int, Dict[str, Any], str]]:
        for index, step in enumerate(base_plan):
            if step.get("action") != "pick_up":
                continue
            robot = step.get("robot")
            pickup_node = step.get("args", {}).get("node")
            if robot and pickup_node and pickup_node != end_node:
                note = (
                    f"Insert an early `{robot}` `put_down` at {end_node} immediately after pickup, "
                    "before the robot has moved there, to trigger a location-consistency failure."
                )
                return (
                    index,
                    {
                        "robot": robot,
                        "action": "put_down",
                        "args": {"item": target_item, "node": end_node},
                    },
                    note,
                )
        return None

    def _morphology_injection(
        self,
        base_plan: List[Dict[str, Any]],
    ) -> Optional[Tuple[int, Dict[str, Any], str]]:
        for index, step in enumerate(base_plan):
            if step.get("action") != "call_elevator":
                continue
            args = step.get("args", {})
            note = (
                "Insert a `dog_robot` elevator traversal on a valid elevator edge so the plan violates "
                "robot morphology constraints without changing the original chosen trajectory."
            )
            return (
                index,
                {
                    "robot": "dog_robot",
                    "action": "call_elevator",
                    "args": {"source": args.get("source"), "target": args.get("target")},
                },
                note,
            )

        for index, step in enumerate(base_plan):
            if step.get("action") != "navigate":
                continue
            args = step.get("args", {})
            source = args.get("source")
            target = args.get("target")
            if not source or not target:
                continue
            target_type = self.engine.nodes_data.get(target, {}).get("type", "Unknown")
            source_type = self.engine.nodes_data.get(source, {}).get("type", "Unknown")
            if source_type in {"Building", "Outdoor"} and target_type in {"Indoor", "Room"}:
                note = (
                    "Insert a `delivery_drone` navigation step from an outdoor-compatible node into an indoor "
                    "node to trigger a morphology violation."
                )
                return (
                    index,
                    {
                        "robot": "delivery_drone",
                        "action": "navigate",
                        "args": {"source": source, "target": target},
                    },
                    note,
                )

        for index, step in enumerate(base_plan):
            args = step.get("args", {})
            candidate_nodes = []
            if "source" in args:
                candidate_nodes.append(args["source"])
            if "target" in args:
                candidate_nodes.append(args["target"])
            if "node" in args:
                candidate_nodes.append(args["node"])

            for node_id in candidate_nodes:
                if self.engine.nodes_data.get(node_id, {}).get("type") not in {"Building", "Outdoor"}:
                    continue
                for neighbor in self.engine.G.neighbors(node_id):
                    neighbor_type = self.engine.nodes_data.get(neighbor, {}).get("type", "Unknown")
                    if neighbor_type not in {"Indoor", "Room", "Elevator", "Stairs"}:
                        continue
                    note = (
                        "Insert a `delivery_drone` step from an outdoor-compatible interface node into a nearby "
                        "indoor node so the validator flags a morphology violation."
                    )
                    return (
                        index,
                        {
                            "robot": "delivery_drone",
                            "action": "navigate",
                            "args": {"source": node_id, "target": neighbor},
                        },
                        note,
                    )
        return None

    def _capability_injection(
        self,
        base_plan: List[Dict[str, Any]],
    ) -> Optional[Tuple[int, Dict[str, Any], str]]:
        for index, step in enumerate(base_plan):
            if step.get("action") != "navigate":
                continue
            args = step.get("args", {})
            source = args.get("source")
            target = args.get("target")
            if not source or not target:
                continue
            edge_data = self.engine.get_edge_data(source, target) or {}
            constraints = set(edge_data.get("constraints", []))
            if constraints & DOOR_CONSTRAINTS:
                note = (
                    "Insert a `dog_robot` navigation step across a door/access-controlled edge so the validator "
                    "flags an unsatisfied door-operation capability."
                )
                return (
                    index,
                    {
                        "robot": "dog_robot",
                        "action": "navigate",
                        "args": {"source": source, "target": target},
                    },
                    note,
                )

        for index, step in enumerate(base_plan):
            if step.get("action") != "navigate":
                continue
            args = step.get("args", {})
            note = (
                "Insert an unsupported `scan_access_card` action so the validator flags an action-level "
                "capability mismatch."
            )
            return (
                index,
                {
                    "robot": step.get("robot"),
                    "action": "scan_access_card",
                    "args": {"source": args.get("source"), "target": args.get("target")},
                },
                note,
            )
        return None

    def _apply_mutations(
        self,
        base_plan: List[Dict[str, Any]],
        mutation_steps: List[Tuple[int, Dict[str, Any], str]],
    ) -> List[Dict[str, Any]]:
        plan = deepcopy(base_plan)
        for insert_after_index, injected_step, _ in sorted(mutation_steps, key=lambda item: item[0], reverse=True):
            step_payload = {
                "id": "tmp",
                "robot": injected_step["robot"],
                "action": injected_step["action"],
                "args": dict(injected_step.get("args", {})),
                "depends_on": [],
            }
            plan.insert(insert_after_index + 1, step_payload)

        for index, step in enumerate(plan):
            step["id"] = f"t{index + 1}"
            step["depends_on"] = [f"t{index}"] if index > 0 else []
        return plan

    def _render_reasoning(
        self,
        target_item: str,
        target_errors: List[str],
        mutation_notes: List[str],
        detected_errors: List[str],
    ) -> str:
        note_lines = [f"- {note}" for note in mutation_notes]
        return (
            "[Edit Objective]\n"
            f"- Start from a valid transport plan for '{target_item}' and inject the target error families "
            f"{target_errors} while keeping the plan parseable.\n\n"
            "[Injected Errors]\n"
            f"{chr(10).join(note_lines)}\n\n"
            "[Expected Validator Outcome]\n"
            f"- The edited plan should be invalid.\n"
            f"- Expected detected error families include: {detected_errors}."
        )
