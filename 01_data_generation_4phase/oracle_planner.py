from __future__ import annotations

import random
from collections import Counter
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import networkx as nx

from graph_engine import EmbodiedGraphEngine
from planning_schema import AGENT_PROFILES, DOOR_CONSTRAINTS, get_item_profile
from preference_utils import render_oracle_reasoning_trace
from symbolic_validator import SymbolicValidator


@dataclass
class OraclePlanResult:
    robot: str
    plan: List[Dict]
    thought: str
    feedback: Dict
    path: List[str]
    reasoning_trace: Dict[str, Any]
    trajectory_cost: float = float("inf")
    utility_score: float = float("inf")
    step_count: int = 0
    handover_count: int = 0
    risk_score: float = 0.0
    path_rank: int = 1


class OraclePlanner:
    def __init__(
        self,
        engine: EmbodiedGraphEngine,
        validator: Optional[SymbolicValidator] = None,
        enable_diverse_selection: bool = True,
        max_cost_ratio: float = 1.35,
        max_cost_gap: float = 35.0,
        utility_cost_weight: float = 1.0,
        utility_step_weight: float = 0.15,
        utility_handover_weight: float = 2.0,
        utility_risk_weight: float = 0.6,
        topk_paths_per_robot: int = 3,
    ):
        self.engine = engine
        self.validator = validator or SymbolicValidator(engine)
        self.enable_diverse_selection = enable_diverse_selection
        self.max_cost_ratio = max_cost_ratio
        self.max_cost_gap = max_cost_gap
        self.utility_cost_weight = utility_cost_weight
        self.utility_step_weight = utility_step_weight
        self.utility_handover_weight = utility_handover_weight
        self.utility_risk_weight = utility_risk_weight
        self.topk_paths_per_robot = max(1, int(topk_paths_per_robot))
        self.selected_robot_counts: Counter[str] = Counter()

    def plan_task(
        self,
        start_node: str,
        end_node: str,
        target_item: str,
        preferred_robot: Optional[str] = None,
        force_preferred_if_feasible: bool = False,
    ) -> Optional[OraclePlanResult]:
        item_profile = get_item_profile(target_item)
        candidate_results, candidate_summaries = self._collect_candidate_results(
            start_node=start_node,
            end_node=end_node,
            target_item=target_item,
            item_profile=item_profile,
            max_paths_per_robot=1,
        )

        if not candidate_results:
            return None

        candidate_results.sort(
            key=lambda result: (
                result.feedback.get("trajectory_cost", float("inf")),
                len(result.plan),
            )
        )

        usage_before_selection = dict(self.selected_robot_counts)
        best_result = self._select_candidate(
            candidate_results,
            preferred_robot=preferred_robot,
            force_preferred_if_feasible=force_preferred_if_feasible,
        )
        reasoning_trace = self._build_reasoning_trace(
            selected_robot=best_result.robot,
            selected_path=best_result.path,
            target_item=target_item,
            item_profile=item_profile,
            start_node=start_node,
            end_node=end_node,
            plan=best_result.plan,
            feedback=best_result.feedback,
            candidate_summaries=candidate_summaries,
            usage_before_selection=usage_before_selection,
            selected_utility_score=best_result.utility_score,
        )
        self.selected_robot_counts[best_result.robot] += 1
        best_result.reasoning_trace = reasoning_trace
        best_result.thought = self._render_reasoning_trace(reasoning_trace)
        return best_result

    def plan_task_topk(
        self,
        start_node: str,
        end_node: str,
        target_item: str,
        *,
        k: int = 3,
        max_paths_per_robot: Optional[int] = None,
        preferred_robot: Optional[str] = None,
        force_preferred_if_feasible: bool = False,
    ) -> List[OraclePlanResult]:
        if k <= 0:
            return []

        item_profile = get_item_profile(target_item)
        candidate_results, candidate_summaries = self._collect_candidate_results(
            start_node=start_node,
            end_node=end_node,
            target_item=target_item,
            item_profile=item_profile,
            max_paths_per_robot=max_paths_per_robot or self.topk_paths_per_robot,
        )
        if not candidate_results:
            return []

        candidate_results.sort(
            key=lambda result: (
                result.utility_score,
                result.trajectory_cost,
                result.step_count,
            )
        )

        if preferred_robot:
            preferred_rows = [row for row in candidate_results if row.robot == preferred_robot]
            if force_preferred_if_feasible and preferred_rows:
                candidate_results = preferred_rows + [row for row in candidate_results if row.robot != preferred_robot]

        topk = candidate_results[:k]
        usage_before_selection = dict(self.selected_robot_counts)
        for row in topk:
            reasoning_trace = self._build_reasoning_trace(
                selected_robot=row.robot,
                selected_path=row.path,
                target_item=target_item,
                item_profile=item_profile,
                start_node=start_node,
                end_node=end_node,
                plan=row.plan,
                feedback=row.feedback,
                candidate_summaries=candidate_summaries,
                usage_before_selection=usage_before_selection,
                selected_utility_score=row.utility_score,
            )
            row.reasoning_trace = reasoning_trace
            row.thought = self._render_reasoning_trace(reasoning_trace)
        return topk

    def summarize_robot_options(
        self,
        start_node: str,
        end_node: str,
        target_item: str,
    ) -> List[Dict[str, Any]]:
        item_profile = get_item_profile(target_item)
        summaries: List[Dict[str, Any]] = []

        for robot in ("chassis_car", "delivery_drone", "dog_robot"):
            feasibility_issue = self._check_robot_feasibility(
                robot=robot,
                start_node=start_node,
                end_node=end_node,
                item_profile=item_profile,
            )
            if feasibility_issue:
                summaries.append(
                    {
                        "robot": robot,
                        "status": "infeasible",
                        "reason": feasibility_issue,
                        "trajectory_cost": None,
                        "step_count": None,
                        "path_hops": None,
                        "edge_profile": {},
                        "path_nodes": [],
                    }
                )
                continue

            path = self._shortest_path_for_robot(
                robot=robot,
                start_node=start_node,
                end_node=end_node,
                item_weight=item_profile.get("weight_kg", 0.0),
            )
            if not path:
                summaries.append(
                    {
                        "robot": robot,
                        "status": "infeasible",
                        "reason": "no_path_under_robot_constraints",
                        "trajectory_cost": None,
                        "step_count": None,
                        "path_hops": None,
                        "edge_profile": {},
                        "path_nodes": [],
                    }
                )
                continue

            plan = self._path_to_plan(
                robot=robot,
                path=path,
                target_item=target_item,
                start_node=start_node,
                end_node=end_node,
            )
            feedback = self.validator.validate_plan(
                plan,
                start_node=start_node,
                end_node=end_node,
                target_item=target_item,
            )
            edge_profile = self._summarize_edge_profile(path)
            summaries.append(
                {
                    "robot": robot,
                    "status": "feasible" if feedback.get("is_valid", False) else "invalid_after_validation",
                    "reason": "valid_graph_grounded_route_found"
                    if feedback.get("is_valid", False)
                    else feedback.get("error_details", "validator_rejected_plan"),
                    "trajectory_cost": feedback.get("trajectory_cost"),
                    "step_count": len(plan),
                    "path_hops": max(len(path) - 1, 0),
                    "edge_profile": edge_profile,
                    "path_nodes": path,
                }
            )

        summaries.sort(
            key=lambda row: (
                0 if row["status"] == "feasible" else 1,
                float(row["trajectory_cost"]) if row["trajectory_cost"] is not None else float("inf"),
                float(row["step_count"]) if row["step_count"] is not None else float("inf"),
            )
        )
        return summaries

    def _collect_candidate_results(
        self,
        *,
        start_node: str,
        end_node: str,
        target_item: str,
        item_profile: Dict[str, Any],
        max_paths_per_robot: int,
    ) -> tuple[List[OraclePlanResult], List[Dict[str, Any]]]:
        candidate_results: List[OraclePlanResult] = []
        candidate_summaries: List[Dict[str, Any]] = []

        for robot in ("chassis_car", "delivery_drone", "dog_robot"):
            paths = self._k_shortest_paths_for_robot(
                robot=robot,
                start_node=start_node,
                end_node=end_node,
                item_weight=item_profile.get("weight_kg", 0.0),
                k=max(1, int(max_paths_per_robot)),
            )
            if not paths:
                candidate_summaries.append(
                    self._build_infeasible_candidate_summary(
                        robot=robot,
                        start_node=start_node,
                        end_node=end_node,
                        target_item=target_item,
                        item_profile=item_profile,
                    )
                )
                continue

            for path_rank, path in enumerate(paths, start=1):
                plan = self._path_to_plan(
                    robot=robot,
                    path=path,
                    target_item=target_item,
                    start_node=start_node,
                    end_node=end_node,
                )
                feedback = self.validator.validate_plan(
                    plan,
                    start_node=start_node,
                    end_node=end_node,
                    target_item=target_item,
                )
                if not feedback.get("is_valid", False):
                    candidate_summaries.append(
                        {
                            "robot": robot,
                            "status": "invalid_after_validation",
                            "reason": feedback.get("error_details", "validator_rejected_plan"),
                            "trajectory_cost": feedback.get("trajectory_cost"),
                            "num_steps": len(plan),
                            "path": path,
                            "path_rank": path_rank,
                        }
                    )
                    continue

                result = self._build_candidate_result(
                    robot=robot,
                    plan=plan,
                    feedback=feedback,
                    path=path,
                    path_rank=path_rank,
                )
                candidate_summaries.append(
                    self._build_feasible_candidate_summary(
                        robot=robot,
                        path=path,
                        feedback=feedback,
                        utility_score=result.utility_score,
                        risk_score=result.risk_score,
                        path_rank=path_rank,
                    )
                )
                candidate_results.append(result)

        return candidate_results, candidate_summaries

    def _build_candidate_result(
        self,
        *,
        robot: str,
        plan: List[Dict[str, Any]],
        feedback: Dict[str, Any],
        path: List[str],
        path_rank: int,
    ) -> OraclePlanResult:
        step_count = len(plan)
        handover_count = sum(1 for step in plan if step.get("action") == "handover")
        trajectory_cost = float(feedback.get("trajectory_cost", float("inf")) or float("inf"))
        risk_score = self._estimate_plan_risk(robot=robot, path=path)
        utility_score = self._compute_utility_score(
            trajectory_cost=trajectory_cost,
            step_count=step_count,
            handover_count=handover_count,
            risk_score=risk_score,
        )
        return OraclePlanResult(
            robot=robot,
            plan=plan,
            thought="",
            feedback=feedback,
            path=path,
            reasoning_trace={},
            trajectory_cost=trajectory_cost,
            utility_score=utility_score,
            step_count=step_count,
            handover_count=handover_count,
            risk_score=risk_score,
            path_rank=path_rank,
        )

    def _select_candidate(
        self,
        candidate_results: List[OraclePlanResult],
        preferred_robot: Optional[str] = None,
        force_preferred_if_feasible: bool = False,
    ) -> OraclePlanResult:
        if not self.enable_diverse_selection or len(candidate_results) == 1:
            return candidate_results[0]

        best_cost = float(candidate_results[0].feedback.get("trajectory_cost", float("inf")))
        cost_threshold = min(best_cost * self.max_cost_ratio, best_cost + self.max_cost_gap)
        eligible = [
            result
            for result in candidate_results
            if float(result.feedback.get("trajectory_cost", float("inf"))) <= cost_threshold
        ]
        if not eligible:
            return candidate_results[0]

        if preferred_robot:
            preferred_any = [result for result in candidate_results if result.robot == preferred_robot]
            if force_preferred_if_feasible and preferred_any:
                preferred_any.sort(
                    key=lambda result: (
                        result.feedback.get("trajectory_cost", float("inf")),
                        len(result.plan),
                    )
                )
                return preferred_any[0]

            preferred_threshold = min(best_cost * 1.8, best_cost + 80.0)
            preferred_candidates = [
                result
                for result in candidate_results
                if result.robot == preferred_robot
                and float(result.feedback.get("trajectory_cost", float("inf"))) <= preferred_threshold
            ]
            if preferred_candidates:
                preferred_candidates.sort(
                    key=lambda result: (
                        self.selected_robot_counts[result.robot],
                        result.feedback.get("trajectory_cost", float("inf")),
                        len(result.plan),
                    )
                )
                return preferred_candidates[0]

        eligible.sort(
            key=lambda result: (
                self.selected_robot_counts[result.robot],
                result.feedback.get("trajectory_cost", float("inf")),
                len(result.plan),
            )
        )
        best_key = (
            self.selected_robot_counts[eligible[0].robot],
            eligible[0].feedback.get("trajectory_cost", float("inf")),
            len(eligible[0].plan),
        )
        tied = [
            result
            for result in eligible
            if (
                self.selected_robot_counts[result.robot],
                result.feedback.get("trajectory_cost", float("inf")),
                len(result.plan),
            )
            == best_key
        ]
        return random.choice(tied)

    def _shortest_path_for_robot(
        self,
        robot: str,
        start_node: str,
        end_node: str,
        item_weight: float,
    ) -> Optional[List[str]]:
        if item_weight > AGENT_PROFILES[robot]["max_load_kg"]:
            return None
        if start_node not in self.engine.G or end_node not in self.engine.G:
            return None
        if not self._node_allowed(robot, start_node) or not self._node_allowed(robot, end_node):
            return None

        filtered_graph = self._build_robot_filtered_graph(robot)

        if start_node not in filtered_graph or end_node not in filtered_graph:
            return None

        try:
            return nx.shortest_path(filtered_graph, start_node, end_node, weight="weight")
        except nx.NetworkXNoPath:
            return None

    def _k_shortest_paths_for_robot(
        self,
        *,
        robot: str,
        start_node: str,
        end_node: str,
        item_weight: float,
        k: int,
    ) -> List[List[str]]:
        if k <= 0:
            return []
        if k == 1:
            one_path = self._shortest_path_for_robot(
                robot=robot,
                start_node=start_node,
                end_node=end_node,
                item_weight=item_weight,
            )
            return [one_path] if one_path else []

        if item_weight > AGENT_PROFILES[robot]["max_load_kg"]:
            return []
        if start_node not in self.engine.G or end_node not in self.engine.G:
            return []
        if not self._node_allowed(robot, start_node) or not self._node_allowed(robot, end_node):
            return []

        filtered_graph = self._build_robot_filtered_graph(robot)
        if start_node not in filtered_graph or end_node not in filtered_graph:
            return []

        paths: List[List[str]] = []
        try:
            generator = nx.shortest_simple_paths(filtered_graph, start_node, end_node, weight="weight")
            seen = set()
            for path in generator:
                key = tuple(path)
                if key in seen:
                    continue
                seen.add(key)
                paths.append(list(path))
                if len(paths) >= k:
                    break
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return []
        return paths

    def _build_robot_filtered_graph(self, robot: str) -> nx.Graph:
        filtered_graph = nx.Graph()
        for source, target, edge_data in self.engine.G.edges(data=True):
            if not self._edge_allowed(robot, source, target, edge_data):
                continue
            filtered_graph.add_edge(source, target, weight=float(edge_data.get("weight", 1.0)))
        return filtered_graph

    def _compute_utility_score(
        self,
        *,
        trajectory_cost: float,
        step_count: int,
        handover_count: int,
        risk_score: float,
    ) -> float:
        utility = (
            self.utility_cost_weight * float(trajectory_cost)
            + self.utility_step_weight * float(step_count)
            + self.utility_handover_weight * float(handover_count)
            + self.utility_risk_weight * float(risk_score)
        )
        return round(utility, 6)

    def _estimate_plan_risk(self, *, robot: str, path: List[str]) -> float:
        risk = 0.0
        for source, target in zip(path, path[1:]):
            edge_data = self.engine.get_edge_data(source, target) or {}
            edge_type = edge_data.get("edge_type", "")
            if edge_type == "Elevator":
                risk += 0.8
            elif edge_type == "Stairs":
                risk += 1.0

            constraints = edge_data.get("constraints", [])
            if isinstance(constraints, list):
                risk += 0.15 * len(constraints)

        # Keep a small prior penalty for aerial delivery so top-k can expose non-drone
        # alternatives when costs are close.
        if robot == "delivery_drone":
            risk += 0.25
        return round(risk, 6)

    def _node_allowed(self, robot: str, node_id: str) -> bool:
        node_type = self.engine.nodes_data.get(node_id, {}).get("type", "Unknown")
        return node_type in AGENT_PROFILES[robot]["allowed_node_types"]

    def _check_robot_feasibility(
        self,
        robot: str,
        start_node: str,
        end_node: str,
        item_profile: Dict[str, Any],
    ) -> Optional[str]:
        if item_profile.get("weight_kg", 0.0) > AGENT_PROFILES[robot]["max_load_kg"]:
            return f"payload_exceeded: {item_profile.get('weight_kg')}kg > {AGENT_PROFILES[robot]['max_load_kg']}kg"
        if start_node not in self.engine.G or end_node not in self.engine.G:
            return "undefined_start_or_end_node"
        if not self._node_allowed(robot, start_node):
            start_type = self.engine.nodes_data.get(start_node, {}).get("type", "Unknown")
            return f"start_node_type_not_allowed: {start_type}"
        if not self._node_allowed(robot, end_node):
            end_type = self.engine.nodes_data.get(end_node, {}).get("type", "Unknown")
            return f"end_node_type_not_allowed: {end_type}"
        return None

    def _edge_allowed(self, robot: str, source: str, target: str, edge_data: Dict) -> bool:
        traversable_by = set(edge_data.get("traversable_by", []))
        if robot not in traversable_by:
            return False

        if not self._node_allowed(robot, source) or not self._node_allowed(robot, target):
            return False

        constraints = set(edge_data.get("constraints", []))
        if robot != "chassis_car" and constraints & DOOR_CONSTRAINTS:
            return False

        return True

    def _build_infeasible_candidate_summary(
        self,
        robot: str,
        start_node: str,
        end_node: str,
        target_item: str,
        item_profile: Dict[str, Any],
    ) -> Dict[str, Any]:
        feasibility_issue = self._check_robot_feasibility(
            robot=robot,
            start_node=start_node,
            end_node=end_node,
            item_profile=item_profile,
        )
        if feasibility_issue:
            return {
                "robot": robot,
                "status": "infeasible",
                "reason": feasibility_issue,
                "path": [],
            }

        path = self._shortest_path_for_robot(
            robot=robot,
            start_node=start_node,
            end_node=end_node,
            item_weight=item_profile.get("weight_kg", 0.0),
        )
        if path:
            return {
                "robot": robot,
                "status": "unexpected_feasible",
                "reason": f"{robot} actually has a valid path for '{target_item}'.",
                "path": path,
            }

        return {
            "robot": robot,
            "status": "infeasible",
            "reason": "no_path_under_robot_constraints",
            "path": [],
        }

    def _build_feasible_candidate_summary(
        self,
        robot: str,
        path: List[str],
        feedback: Dict[str, Any],
        utility_score: Optional[float] = None,
        risk_score: Optional[float] = None,
        path_rank: int = 1,
    ) -> Dict[str, Any]:
        return {
            "robot": robot,
            "status": "feasible",
            "reason": "valid_graph_grounded_route_found",
            "trajectory_cost": feedback.get("trajectory_cost"),
            "num_steps": len(path) + 1,
            "path": path,
            "path_rank": path_rank,
            "utility_score": utility_score,
            "risk_score": risk_score,
        }

    def _summarize_edge_profile(self, path: List[str]) -> Dict[str, int]:
        edge_counter: Counter[str] = Counter()
        for source, target in zip(path, path[1:]):
            edge_data = self.engine.get_edge_data(source, target) or {}
            edge_type = str(edge_data.get("edge_type", "Unknown"))
            edge_counter[edge_type] += 1
        return dict(edge_counter)

    def _path_to_plan(
        self,
        robot: str,
        path: List[str],
        target_item: str,
        start_node: str,
        end_node: str,
    ) -> List[Dict]:
        plan: List[Dict] = []
        previous_step_ids: List[str] = []

        def append_step(action: str, args: Dict) -> None:
            step_id = f"t{len(plan) + 1}"
            plan.append(
                {
                    "id": step_id,
                    "robot": robot,
                    "action": action,
                    "args": args,
                    "depends_on": previous_step_ids[-1:] if previous_step_ids else [],
                }
            )
            previous_step_ids.append(step_id)

        append_step("pick_up", {"item": target_item, "node": start_node})

        for source, target in zip(path, path[1:]):
            edge_data = self.engine.get_edge_data(source, target) or {}
            edge_type = edge_data.get("edge_type", "FlatGround")
            if edge_type == "Elevator":
                action = "call_elevator"
            elif edge_type == "Stairs":
                action = "climb_stairs"
            else:
                action = "navigate"
            append_step(action, {"source": source, "target": target})

        append_step("put_down", {"item": target_item, "node": end_node})
        return plan

    def _build_reasoning_trace(
        self,
        selected_robot: str,
        selected_path: List[str],
        target_item: str,
        item_profile: Dict[str, Any],
        start_node: str,
        end_node: str,
        plan: List[Dict[str, Any]],
        feedback: Dict[str, Any],
        candidate_summaries: List[Dict[str, Any]],
        usage_before_selection: Dict[str, int],
        selected_utility_score: Optional[float] = None,
    ) -> Dict[str, Any]:
        start_profile = self.engine.nodes_data.get(start_node, {})
        end_profile = self.engine.nodes_data.get(end_node, {})
        edges = []
        movement_steps = []
        action_counter: Dict[str, int] = {}

        for step in plan:
            action_counter[step["action"]] = action_counter.get(step["action"], 0) + 1
            if step["action"] in {"navigate", "call_elevator", "climb_stairs"}:
                edge = self.engine.get_edge_data(step["args"]["source"], step["args"]["target"]) or {}
                movement_steps.append(
                    {
                        "step_id": step["id"],
                        "action": step["action"],
                        "source": step["args"]["source"],
                        "target": step["args"]["target"],
                        "edge_type": edge.get("edge_type", "Unknown"),
                        "weight": edge.get("weight"),
                        "constraints": edge.get("constraints", []),
                    }
                )
                edges.append(edge)

        exact_pickup_ok = bool(plan and plan[0]["action"] == "pick_up" and plan[0]["args"].get("node") == start_node)
        exact_putdown_ok = bool(plan and plan[-1]["action"] == "put_down" and plan[-1]["args"].get("node") == end_node)

        return {
            "task_grounding": {
                "target_item": target_item,
                "item_profile": item_profile,
                "start_node": start_node,
                "start_node_type": start_profile.get("type", "Unknown"),
                "end_node": end_node,
                "end_node_type": end_profile.get("type", "Unknown"),
            },
            "candidate_scan": candidate_summaries,
            "selection": {
                "selected_robot": selected_robot,
                "selection_rule": (
                    "choose among near-optimal valid candidates using robot-balancing first, "
                    "then trajectory cost, then fewest steps"
                ),
                "selected_trajectory_cost": feedback.get("trajectory_cost", 0.0),
                "selected_utility_score": selected_utility_score,
                "selected_plan_steps": len(plan),
                "selected_path_nodes": selected_path,
                "robot_usage_before_selection": usage_before_selection,
            },
            "route_decomposition": {
                "movement_steps": movement_steps,
                "action_counts": action_counter,
            },
            "constraint_check": {
                "payload_ok": item_profile.get("weight_kg", 0.0) <= AGENT_PROFILES[selected_robot]["max_load_kg"],
                "robot_payload_limit_kg": AGENT_PROFILES[selected_robot]["max_load_kg"],
                "exact_pickup_node_ok": exact_pickup_ok,
                "exact_putdown_node_ok": exact_putdown_ok,
                "all_moves_adjacent": all(edge for edge in edges) if movement_steps else True,
                "validator_is_valid": feedback.get("is_valid", False),
                "validator_source": feedback.get("validation_source", "unknown"),
                "trajectory_cost": feedback.get("trajectory_cost", 0.0),
            },
            "terminal_check": {
                "delivered_item": target_item,
                "expected_destination": end_node,
                "terminal_condition": "item is placed at the exact end node after the final put_down step",
            },
        }

    def _render_reasoning_trace(self, reasoning_trace: Dict[str, Any]) -> str:
        return render_oracle_reasoning_trace(reasoning_trace)
