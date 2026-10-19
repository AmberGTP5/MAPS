from __future__ import annotations

import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from app.llm import LLM
from app.logger import logger


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
if str(DATA_DIR) not in sys.path:
    sys.path.insert(0, str(DATA_DIR))

from eval_utils import parse_response_text, summarize_plan_shape, validate_plan_schema
from graph_engine import build_default_graph_engine
from oracle_planner import OraclePlanner
from symbolic_validator import SymbolicValidator


_GRAPH_ENGINE = None
_GRAPH_VALIDATOR = None


def _get_graph_components() -> tuple[Any, SymbolicValidator]:
    global _GRAPH_ENGINE, _GRAPH_VALIDATOR
    if _GRAPH_ENGINE is None:
        _GRAPH_ENGINE = build_default_graph_engine(DATA_DIR)
    if _GRAPH_VALIDATOR is None:
        _GRAPH_VALIDATOR = SymbolicValidator(_GRAPH_ENGINE)
    return _GRAPH_ENGINE, _GRAPH_VALIDATOR


def load_graph_tasks(task_file: str | Path) -> List[Dict[str, Any]]:
    task_path = Path(task_file).resolve()
    if task_path.suffix.lower() == ".jsonl":
        rows: List[Dict[str, Any]] = []
        with task_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                rows.append(json.loads(line))
        return rows

    payload = json.loads(task_path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"Graph task file must contain a JSON list or JSONL rows: {task_path}")
    return payload


def select_graph_task(
    task_file: str | Path,
    *,
    task_id: str | None = None,
    task_index: int | None = None,
) -> Dict[str, Any]:
    rows = load_graph_tasks(task_file)
    if task_id:
        for row in rows:
            if row.get("task_id") == task_id or row.get("id") == task_id:
                return row
        raise KeyError(f"Could not find task_id='{task_id}' in {task_file}")

    index = task_index if task_index is not None else 0
    if index < 0 or index >= len(rows):
        raise IndexError(f"task_index={index} is out of range for {task_file} ({len(rows)} rows)")
    return rows[index]


class GraphTaskCoordinator:
    def __init__(
        self,
        planning_mode: str = "retry",
        max_retries: int = 2,
        dry_run: bool = True,
        guidance_mode: str = "route_hint",
    ) -> None:
        if planning_mode not in {"greedy", "retry", "search"}:
            raise ValueError("planning_mode must be one of: greedy, retry, search")
        if not dry_run:
            raise ValueError("GraphTaskCoordinator currently supports dry-run symbolic evaluation only.")
        if guidance_mode not in {"none", "feasibility", "route_hint"}:
            raise ValueError("guidance_mode must be one of: none, feasibility, route_hint")

        self.planning_mode = planning_mode
        self.max_retries = max_retries if planning_mode == "retry" else 1
        self.dry_run = dry_run
        self.guidance_mode = guidance_mode

        self.engine, self.validator = _get_graph_components()
        self.search_engine = OraclePlanner(
            self.engine,
            validator=self.validator,
            enable_diverse_selection=True,
        )

        logger.info("Initializing planner LLM for graph mode.")
        self.planner_llm = LLM(config_key="planner")
        self.trace_history: Dict[str, Any] = self._new_trace({})

    async def run_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        self.trace_history = self._new_trace(task)
        base_prompt = str(task.get("prompt", "")).strip()
        if not base_prompt:
            raise ValueError("Graph task is missing a non-empty 'prompt'.")

        symbolic_prompt_context = self._build_symbolic_prompt_context(task)
        self.trace_history["meta"]["symbolic_prompt_context"] = symbolic_prompt_context

        for attempt in range(1, self.max_retries + 1):
            logger.info(f"--- Graph planning phase (attempt {attempt}) ---")
            failure_context = self.trace_history.get("attempts", [])[-1] if self.trace_history.get("attempts") else None
            prompt = self._build_prompt(base_prompt, symbolic_prompt_context, failure_context)
            reasoning, raw_response = await self._generate_plan(prompt)
            planner_eval = self._evaluate_response(raw_response, task)

            selected_eval = planner_eval
            selected_source = "planner"
            search_summary = None

            if self.planning_mode == "search":
                search_eval, search_summary = self._run_search(task, planner_eval)
                if search_eval.get("is_valid"):
                    if not planner_eval.get("is_valid"):
                        selected_eval = search_eval
                        selected_source = "search_repair"
                    else:
                        planner_cost = float(planner_eval.get("trajectory_cost") or 10**9)
                        search_cost = float(search_eval.get("trajectory_cost") or 10**9)
                        planner_steps = int(planner_eval.get("step_count") or 10**6)
                        search_steps = int(search_eval.get("step_count") or 10**6)
                        if (search_cost, search_steps) < (planner_cost, planner_steps):
                            selected_eval = search_eval
                            selected_source = "search_improved"

            attempt_trace = {
                "attempt_id": attempt,
                "planning_mode": self.planning_mode,
                "planner_reasoning": reasoning,
                "planner_raw_response": raw_response,
                "planner_eval": planner_eval,
                "selected_plan_source": selected_source,
                "selected_eval": selected_eval,
                "search_summary": search_summary,
                "outcome": "success_dry_run" if selected_eval.get("is_valid") else "failure_dry_run",
            }
            self.trace_history["attempts"].append(attempt_trace)

            if selected_eval.get("is_valid"):
                self.trace_history["meta"]["status"] = "success_dry_run"
                self._save_trace()
                return attempt_trace

            if self.planning_mode != "retry":
                break

        self.trace_history["meta"]["status"] = "failed_max_retries"
        self._save_trace()
        return self.trace_history["attempts"][-1]

    def _new_trace(self, task: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "meta": {
                "task_id": task.get("task_id") or task.get("id"),
                "task_profile": task.get("sampling_profile")
                or task.get("category")
                or task.get("task_metadata", {}).get("task_profile"),
                "start_node": task.get("start") or task.get("start_node"),
                "end_node": task.get("end") or task.get("end_node"),
                "target_item": task.get("item") or task.get("target_item"),
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "planning_mode": self.planning_mode,
                "guidance_mode": self.guidance_mode,
                "dry_run": self.dry_run,
                "status": "pending",
            },
            "attempts": [],
        }

    def _build_prompt(
        self,
        base_prompt: str,
        symbolic_prompt_context: str,
        prior_attempt: Optional[Dict[str, Any]],
    ) -> str:
        prompt_parts = [base_prompt]
        if symbolic_prompt_context:
            prompt_parts.append(symbolic_prompt_context)

        if prior_attempt and not prior_attempt.get("selected_eval", {}).get("is_valid"):
            selected_eval = prior_attempt.get("selected_eval", {})
            failure_detail = selected_eval.get("error_details") or selected_eval.get("parse_error") or "unknown"
            failed_steps = selected_eval.get("failed_steps", [])
            prompt_parts.append(
                "\n".join(
                    [
                        "### Replanning Context",
                        f"- Previous failure: {failure_detail}",
                        f"- Failed steps: {failed_steps}",
                        "- Fix the broken transitions instead of repeating the same route skeleton.",
                        "- Do not reuse an invalid handover, teleport, or wrong-floor transition.",
                    ]
                )
            )

        return "\n\n".join(part for part in prompt_parts if part)

    def _build_symbolic_prompt_context(self, task: Dict[str, Any]) -> str:
        if self.guidance_mode == "none":
            return ""

        ctx = self._task_context(task)
        start_node = ctx["start_node"]
        end_node = ctx["end_node"]
        target_item = ctx["target_item"]
        if not start_node or not end_node or not target_item:
            return ""

        option_rows = self.search_engine.summarize_robot_options(
            start_node=start_node,
            end_node=end_node,
            target_item=target_item,
        )
        feasible_rows = [row for row in option_rows if row.get("status") == "feasible"]

        lines = [
            "### Runtime Planning Rules",
            "- Prefer the lowest-cost feasible SINGLE-robot plan whenever one exists.",
            "- For the same robot, every step must follow its actual current node. Never branch or teleport.",
            "- If a route-node hint is provided, follow that node order exactly and do not skip intermediate nodes.",
            "- Before `call_elevator`, navigate into the elevator node; after elevator arrival, navigate out from the destination elevator node.",
            "- Do NOT emulate a handover with `put_down` followed by another robot's `pick_up`.",
            "- Use `handover` only if both robots already have explicit steps that place them at the same node.",
            "- Before finalizing, mentally simulate the current node and carried item after every step.",
            "",
            "### Symbolic Feasibility Hints",
        ]

        for row in option_rows:
            robot = row.get("robot", "unknown")
            status = row.get("status")
            if status == "feasible":
                cost = row.get("trajectory_cost")
                steps = row.get("step_count")
                hops = row.get("path_hops")
                edge_profile = row.get("edge_profile") or {}
                edge_profile_str = ", ".join(f"{k}:{v}" for k, v in sorted(edge_profile.items())) or "none"
                lines.append(
                    f"- {robot}: feasible exact single-robot route; estimated cost={cost}, steps={steps}, hops={hops}, edge_mix={edge_profile_str}."
                )
            else:
                lines.append(f"- {robot}: {status} ({row.get('reason', 'unknown_reason')}).")

        if feasible_rows:
            best_row = min(
                feasible_rows,
                key=lambda row: (
                    float(row.get("trajectory_cost") or 10**9),
                    int(row.get("step_count") or 10**6),
                ),
            )
            lines.append(
                f"- Best exact single-robot option: {best_row['robot']}."
            )
            best_path_nodes = best_row.get("path_nodes") or []
            if self.guidance_mode == "route_hint" and best_path_nodes:
                lines.append(
                    f"- Route-node hint for the best option: {best_path_nodes}."
                )
            lines.append(
                "- Unless the task is impossible for one robot, avoid multi-robot handovers."
            )
        else:
            lines.append(
                "- No exact single-robot route was found; if you use multiple robots, explicitly move every receiver to the handover node before `handover`."
            )

        return "\n".join(lines)

    async def _generate_plan(self, prompt: str) -> Tuple[str, str]:
        messages = [{"role": "user", "content": prompt}]
        full_content = ""

        try:
            if hasattr(self.planner_llm, "chat_stream"):
                async for chunk in self.planner_llm.chat_stream(messages=messages):
                    full_content += chunk
            else:
                response = await self.planner_llm.chat(messages=messages)
                full_content = response.content or ""
        except Exception as exc:
            logger.error(f"Graph planner stream error: {exc}")
            return "Planner error", ""

        parsed = parse_response_text(full_content)
        return parsed.get("think", "") or "No reasoning provided.", full_content

    def _task_context(self, task: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "start_node": task.get("start") or task.get("start_node"),
            "end_node": task.get("end") or task.get("end_node"),
            "target_item": task.get("item") or task.get("target_item"),
        }

    def _evaluate_response(self, response_text: str, task: Dict[str, Any]) -> Dict[str, Any]:
        parsed = parse_response_text(response_text)
        if not parsed.get("parse_ok"):
            return {
                "parse_ok": False,
                "parse_error": parsed.get("parse_error"),
                "schema_pass": False,
                "schema_errors": [],
                "is_valid": False,
                "error_details": parsed.get("parse_error"),
                "failed_steps": [],
                "penalty_score": 5.0,
                "trajectory_cost": None,
                "step_count": 0,
                "handover_count": 0,
                "single_robot": False,
                "selected_robot": None,
                "plan": None,
            }

        plan = parsed.get("plan")
        schema_result = validate_plan_schema(plan)
        if not schema_result.get("is_valid"):
            return {
                "parse_ok": True,
                "parse_error": "",
                "schema_pass": False,
                "schema_errors": schema_result.get("errors", []),
                "is_valid": False,
                "error_details": " | ".join(schema_result.get("errors", [])),
                "failed_steps": [],
                "penalty_score": 5.0,
                "trajectory_cost": None,
                "step_count": len(plan) if isinstance(plan, list) else 0,
                "handover_count": 0,
                "single_robot": False,
                "selected_robot": plan[0].get("robot") if isinstance(plan, list) and plan else None,
                "plan": plan,
            }

        ctx = self._task_context(task)
        feedback = self.validator.validate_plan(
            plan,
            start_node=ctx["start_node"],
            end_node=ctx["end_node"],
            target_item=ctx["target_item"],
        )
        shape = summarize_plan_shape(plan)
        return {
            "parse_ok": True,
            "parse_error": "",
            "schema_pass": True,
            "schema_errors": [],
            "is_valid": bool(feedback.get("is_valid")),
            "error_details": feedback.get("error_details", ""),
            "failed_steps": feedback.get("failed_steps", []),
            "penalty_score": feedback.get("penalty_score"),
            "trajectory_cost": feedback.get("trajectory_cost"),
            "violation_breakdown": feedback.get("violation_breakdown", {}),
            "step_count": shape["step_count"],
            "handover_count": shape["handover_count"],
            "single_robot": shape["single_robot"],
            "selected_robot": plan[0].get("robot") if plan else None,
            "plan": plan,
        }

    def _run_search(self, task: Dict[str, Any], planner_eval: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        ctx = self._task_context(task)
        planner_plan = planner_eval.get("plan") if isinstance(planner_eval.get("plan"), list) else []
        preferred_robot = None
        if planner_plan and isinstance(planner_plan[0], dict):
            robot = planner_plan[0].get("robot")
            if isinstance(robot, str) and robot:
                preferred_robot = robot

        search_result = self.search_engine.plan_task(
            start_node=ctx["start_node"],
            end_node=ctx["end_node"],
            target_item=ctx["target_item"],
            preferred_robot=preferred_robot,
            force_preferred_if_feasible=False,
        )
        if search_result is None:
            failed = {
                "parse_ok": True,
                "parse_error": "",
                "schema_pass": True,
                "schema_errors": [],
                "is_valid": False,
                "error_details": "search_failed_to_find_valid_candidate",
                "failed_steps": [],
                "penalty_score": 5.0,
                "trajectory_cost": None,
                "step_count": 0,
                "handover_count": 0,
                "single_robot": False,
                "selected_robot": None,
                "plan": None,
            }
            return failed, {"preferred_robot": preferred_robot, "search_available": False}

        shape = summarize_plan_shape(search_result.plan)
        eval_result = {
            "parse_ok": True,
            "parse_error": "",
            "schema_pass": True,
            "schema_errors": [],
            "is_valid": bool(search_result.feedback.get("is_valid")),
            "error_details": search_result.feedback.get("error_details", ""),
            "failed_steps": search_result.feedback.get("failed_steps", []),
            "penalty_score": search_result.feedback.get("penalty_score"),
            "trajectory_cost": search_result.feedback.get("trajectory_cost"),
            "violation_breakdown": search_result.feedback.get("violation_breakdown", {}),
            "step_count": shape["step_count"],
            "handover_count": shape["handover_count"],
            "single_robot": shape["single_robot"],
            "selected_robot": search_result.robot,
            "plan": search_result.plan,
        }
        summary = {
            "preferred_robot": preferred_robot,
            "search_available": True,
            "selected_robot": search_result.robot,
            "path_length": len(search_result.path),
            "trajectory_cost": search_result.feedback.get("trajectory_cost"),
            "plan_step_count": len(search_result.plan),
            "reasoning_trace": search_result.reasoning_trace,
        }
        return eval_result, summary

    def _save_trace(self) -> None:
        save_dir = PROJECT_ROOT / "traces"
        save_dir.mkdir(exist_ok=True)
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        task_id = self.trace_history["meta"].get("task_id") or "graph"
        safe_task_id = "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in str(task_id))
        filename = save_dir / f"graph_run_{safe_task_id}_{timestamp_str}.json"
        with filename.open("w", encoding="utf-8") as handle:
            json.dump(self.trace_history, handle, indent=2, ensure_ascii=False)
        logger.info(f"Graph trace saved to {filename}")
