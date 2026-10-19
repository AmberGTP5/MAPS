from __future__ import annotations

import json
import os
import re
from datetime import datetime
from typing import Any, Dict, List, Tuple

from app.llm import LLM
from app.logger import logger
from app.prompt.instructions import EXECUTOR_PROMPT, PLANNER_PROMPT
from app.robot_registry import LOCATION_DATABASE
from app.search.mcts import PlanSearchEngine
from app.search.verifier import (
    ELEVATOR_BY_FLOOR,
    HANDOVER_ZONES,
    SOURCES,
    PlanSearchVerifier,
    SearchConstraints,
)
from app.tool.robot_tools import ArmManage, ChassisNavigate, DogNavigate, ElevatorControl
from app.tool.terminate import Terminate
from app.tool.tool_collection import ToolCollection


class RobotCoordinator:
    def __init__(
        self,
        planning_mode: str = "retry",
        max_retries: int = 2,
        search_iterations: int = 96,
        search_rollout_depth: int = 18,
        dry_run: bool = False,
    ) -> None:
        if planning_mode not in {"greedy", "retry", "search"}:
            raise ValueError("planning_mode must be one of: greedy, retry, search")

        self.planning_mode = planning_mode
        self.max_retries = max_retries if planning_mode == "retry" else 1
        self.dry_run = dry_run

        self.tools = ToolCollection(
            ChassisNavigate(),
            DogNavigate(),
            ArmManage(),
            ElevatorControl(),
            Terminate(),
        )

        logger.info("Initializing planner LLM.")
        self.planner_llm = LLM(config_key="planner")

        self.executor_llm = None
        if not self.dry_run:
            logger.info("Initializing executor LLM.")
            self.executor_llm = LLM(config_key="executor")

        self.search_verifier = PlanSearchVerifier()
        self.search_engine = PlanSearchEngine(
            verifier=self.search_verifier,
            max_iterations=search_iterations,
            rollout_depth=search_rollout_depth,
        )

        self.trace_history: Dict[str, Any] = self._new_trace("")

    async def run(self, user_request: str) -> None:
        logger.info(f"Robot Coordinator started. Mode={self.planning_mode}, task={user_request}")
        self.trace_history = self._new_trace(user_request)
        task_constraints = self._infer_search_constraints(user_request)
        self.search_verifier.set_constraints(task_constraints)
        self.trace_history["meta"]["task_constraints"] = {
            "allowed_sources": list(task_constraints.allowed_sources),
            "allowed_handover_zones": list(task_constraints.allowed_handover_zones),
        }

        current_state = {
            "last_location": "Robot_Station_5F",
            "failed_reason": None,
        }

        for attempt in range(1, self.max_retries + 1):
            logger.info(f"--- Planning phase (attempt {attempt}) ---")
            prompt = self._build_dynamic_prompt(user_request, current_state)
            reasoning, planner_plan, raw_response = await self._generate_plan(prompt)

            if not planner_plan:
                logger.error("Planner failed to generate any usable steps.")
                self._record_terminal_failure("planner_failed", raw_response)
                self._save_trace()
                return

            planner_valid, planner_error, planner_final_state = self.search_verifier.plan_is_valid(planner_plan)
            plan_steps = planner_plan
            search_summary = None
            selected_plan_source = "planner"

            if self.planning_mode == "search":
                search_outcome = self.search_engine.search(prior_plan=planner_plan)
                search_valid, search_error, search_final_state = self.search_verifier.plan_is_valid(search_outcome.plan)
                search_summary = {
                    **search_outcome.summary,
                    "selected_plan_valid": search_valid,
                    "selected_plan_error": search_error,
                    "selected_plan_score": round(search_outcome.score, 3),
                }

                if search_valid:
                    plan_steps = search_outcome.plan
                    selected_plan_source = "search"
                elif planner_valid:
                    search_summary["fallback_to_planner"] = True
                    plan_steps = planner_plan
                    selected_plan_source = "planner_fallback"

            logger.info(
                f"Selected plan source={selected_plan_source} "
                f"planner_valid={planner_valid} "
                f"search_enabled={self.planning_mode == 'search'}"
            )
            self._print_plan(plan_steps, selected_plan_source)

            current_attempt_trace = {
                "attempt_id": attempt,
                "planning_mode": self.planning_mode,
                "dry_run": self.dry_run,
                "task_constraints": self.trace_history["meta"]["task_constraints"],
                "selected_plan_source": selected_plan_source,
                "planner_reasoning": reasoning,
                "planner_raw_response": raw_response,
                "planner_plan": planner_plan,
                "planner_plan_valid": planner_valid,
                "planner_plan_error": planner_error,
                "planner_plan_score": round(self.search_verifier.score_state(planner_final_state), 3),
                "executed_plan": plan_steps,
                "search_summary": search_summary,
                "execution_log": [],
                "outcome": "unknown",
            }

            if self.dry_run:
                selected_valid, selected_error, selected_final_state = self.search_verifier.plan_is_valid(plan_steps)
                current_attempt_trace["selected_plan_valid"] = selected_valid
                current_attempt_trace["selected_plan_error"] = selected_error
                current_attempt_trace["selected_plan_score"] = round(
                    self.search_verifier.score_state(selected_final_state), 3
                )
                current_attempt_trace["execution_log"] = []

                if selected_valid:
                    current_attempt_trace["outcome"] = "success_dry_run"
                    self.trace_history["attempts"].append(current_attempt_trace)
                    self.trace_history["meta"]["status"] = "success_dry_run"
                    logger.info("Dry-run symbolic validation succeeded.")
                    self._save_trace()
                    return

                current_attempt_trace["outcome"] = "failure_dry_run"
                current_attempt_trace["error"] = selected_error
                self.trace_history["attempts"].append(current_attempt_trace)
                current_state["failed_reason"] = selected_error or "Symbolic validation failed."
                logger.warning(f"Dry-run validation failed on attempt {attempt}: {current_state['failed_reason']}")
                continue

            logger.info(f"--- Execution phase (attempt {attempt}) ---")
            success, execution_log, failure_reason = await self._execute_plan(plan_steps, current_state)
            current_attempt_trace["execution_log"] = execution_log

            if success:
                current_attempt_trace["outcome"] = "success"
                self.trace_history["attempts"].append(current_attempt_trace)
                self.trace_history["meta"]["status"] = "success"
                logger.info("Mission completed successfully.")
                self._save_trace()
                return

            current_attempt_trace["outcome"] = "failure"
            current_attempt_trace["error"] = failure_reason
            self.trace_history["attempts"].append(current_attempt_trace)
            current_state["failed_reason"] = failure_reason
            logger.warning(f"Attempt {attempt} failed: {failure_reason}")

        self.trace_history["meta"]["status"] = "failed_max_retries"
        logger.critical("Max retries reached. Mission failed.")
        self._save_trace()

    def _new_trace(self, task: str) -> Dict[str, Any]:
        return {
            "meta": {
                "task": task,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "status": "pending",
                "planning_mode": self.planning_mode,
                "dry_run": self.dry_run,
            },
            "attempts": [],
        }

    def _record_terminal_failure(self, status: str, raw_response: str) -> None:
        self.trace_history["meta"]["status"] = status
        self.trace_history["attempts"].append(
            {
                "attempt_id": 1,
                "planning_mode": self.planning_mode,
                "planner_raw_response": raw_response,
                "outcome": status,
            }
        )

    def _build_dynamic_prompt(self, user_request: str, current_state: Dict[str, Any]) -> str:
        if not current_state["failed_reason"]:
            return f"{PLANNER_PROMPT}\n\nUser Request: {user_request}"

        return (
            f"{PLANNER_PROMPT}\n\n"
            "### Replanning context\n"
            f"- Current location: {current_state['last_location']}\n"
            f"- Failure reason: {current_state['failed_reason']}\n"
            f"- Original task: {user_request}\n"
            "Generate a new plan that avoids the failure while still satisfying all constraints."
        )

    def _infer_search_constraints(self, user_request: str) -> SearchConstraints:
        request = user_request.lower()

        allowed_sources = SOURCES
        if "starbucks" in request:
            allowed_sources = ("Starbucks_Outdoor",)
        elif "vending machine" in request or "vending" in request:
            allowed_sources = ("Vending_Machine_2F",)

        allowed_handover_zones = HANDOVER_ZONES
        if "info hub" in request or "info_hub" in request:
            allowed_handover_zones = ("Info_Hub_4F",)
        elif re.search(r"(handoff|handover|meet).*(elevator lobby|lobby)|4f elevator lobby|elevator_lobby_4f", request):
            allowed_handover_zones = ("Elevator_Lobby_4F",)

        return SearchConstraints(
            allowed_sources=tuple(allowed_sources),
            allowed_handover_zones=tuple(allowed_handover_zones),
        )

    async def _generate_plan(self, prompt: str) -> Tuple[str, List[dict], str]:
        messages = [{"role": "user", "content": prompt}]
        print("\nPlanner is thinking...")
        print("-" * 50)

        full_content = ""
        try:
            if hasattr(self.planner_llm, "chat_stream"):
                async for chunk in self.planner_llm.chat_stream(messages=messages):
                    print(chunk, end="", flush=True)
                    full_content += chunk
            else:
                response = await self.planner_llm.chat(messages=messages)
                full_content = response.content or ""
                print(full_content)
        except Exception as exc:
            logger.error(f"Planner stream error: {exc}")
            print(f"\nPlanner stream error: {exc}")
            return "Planner error", [], ""

        print("\n" + "-" * 50)
        reasoning, steps = self._extract_plan(full_content)
        return reasoning, steps, full_content

    def _extract_plan(self, content: str) -> Tuple[str, List[dict]]:
        think_match = re.search(r"<think>(.*?)</think>", content, re.DOTALL | re.IGNORECASE)
        reasoning = think_match.group(1).strip() if think_match else "No reasoning provided."

        candidates: List[str] = []

        answer_match = re.search(r"<answer>(.*?)</answer>", content, re.DOTALL | re.IGNORECASE)
        if answer_match:
            candidates.append(answer_match.group(1).strip())

        for block in re.findall(r"```json(.*?)```", content, re.DOTALL | re.IGNORECASE):
            candidates.append(block.strip())

        array_match = re.search(r"(\[\s*\{.*\}\s*\])", content, re.DOTALL)
        if array_match:
            candidates.append(array_match.group(1).strip())

        seen = set()
        for candidate in candidates:
            if candidate in seen:
                continue
            seen.add(candidate)
            try:
                payload = json.loads(candidate)
            except json.JSONDecodeError:
                continue
            normalized = self._normalize_plan(payload)
            if normalized:
                return reasoning, normalized

        logger.error(f"Failed to parse planner output into JSON steps: {content}")
        return reasoning, []

    def _normalize_plan(self, payload: Any) -> List[dict]:
        if not isinstance(payload, list):
            return []

        plan_steps: List[dict] = []
        for item in payload:
            if not isinstance(item, dict):
                return []

            robot = str(item.get("robot", "")).strip()
            action = str(item.get("action", "")).strip()
            target = str(item.get("target", "")).strip()
            target_floor = item.get("target_floor")
            details = str(item.get("details", "")).strip()

            if action in {"take_elevator", "call_elevator"}:
                action = "use_elevator"
            elif action in {
                "grasp_item",
                "pick_up",
                "pickup",
                "acquire",
                "acquire_item",
                "obtain",
                "get_item",
                "fetch_item",
                "buy",
                "purchase",
            }:
                action = "grasp"
            elif action in {"transfer", "transfer_item", "handover"}:
                action = "transfer_to_dog"
                target = "coffee"
            elif action in {"deliver", "deliver_item", "put_down"}:
                action = "deliver"
            elif action in {"receive", "receive_item"}:
                action = "receive"

            if action == "use_elevator":
                if target.startswith("Elevator_Lobby_") and "_" in target:
                    target = target.rsplit("_", 1)[-1]
                elif target_floor not in (None, ""):
                    target = f"{str(target_floor).strip()}F".replace("FF", "F")

            if action in {"grasp", "transfer_to_dog"} and robot == "Chassis":
                robot = "Arm"

            if action == "grasp" and not target:
                target = "coffee"

            if action in {"deliver", "receive"} and not target:
                target = "coffee"

            if not robot or not action or not target:
                return []

            plan_steps.append(
                {
                    "robot": robot,
                    "action": action,
                    "target": target,
                    "details": details,
                }
            )

        return plan_steps

    def _print_plan(self, plan_steps: List[dict], source_label: str) -> None:
        print(f"Plan source: {source_label}")
        print(f"Plan contains {len(plan_steps)} steps.")
        for index, step in enumerate(plan_steps, start=1):
            print(f"{index}. {step}")
            logger.info(
                f"[Plan] Step {index}: [{step.get('robot', 'System')}] "
                f"{step.get('action', 'unknown')} -> {step.get('target', 'unknown')}"
            )
        print("-" * 50)

    async def _execute_plan(
        self, plan_steps: List[dict], current_state: Dict[str, Any]
    ) -> Tuple[bool, List[dict], str]:
        execution_log: List[dict] = []

        for index, step in enumerate(plan_steps, start=1):
            step_desc = (
                f"Step {index}/{len(plan_steps)}: "
                f"[{step.get('robot', 'System')}] {step.get('action', 'unknown')} -> {step.get('target', 'unknown')}"
            )
            print(f"\nProcessing {step_desc}")
            logger.info(f"[Exec] {step_desc}")

            success, actions = await self._execute_step(step)
            execution_log.append(
                {
                    "step_index": index,
                    "instruction": step,
                    "actions": actions,
                    "success": success,
                }
            )

            self._update_runtime_location(current_state, step)

            if not success:
                error_msg = self._extract_failure_reason(actions) or "Unknown execution failure."
                return False, execution_log, error_msg

            terminate_status = self._extract_terminate_status(actions)
            if terminate_status == "success":
                return True, execution_log, ""
            if terminate_status == "failure":
                return False, execution_log, "Terminate tool reported failure."

        return True, execution_log, ""

    async def _execute_step(self, step: dict) -> Tuple[bool, List[dict]]:
        if self.executor_llm is None:
            return False, [{"tool": "system", "status": "failed", "output": "executor_llm_not_initialized"}]

        instruction_str = json.dumps(step, ensure_ascii=False)
        current_prompt = EXECUTOR_PROMPT.format(instruction=instruction_str)
        messages = [
            {"role": "system", "content": current_prompt},
            {"role": "user", "content": "Generate Tool Call."},
        ]

        response = await self.executor_llm.chat(messages=messages, tools=self.tools.to_params())
        if not response.tool_calls:
            logger.warning("Executor returned no tool call.")
            return False, []

        executed_actions: List[dict] = []
        step_success = True

        for tool_call in response.tool_calls:
            tool_name = tool_call.function.name
            tool_args = self._safe_json_loads(tool_call.function.arguments)

            print(f"Calling Tool: {tool_name} | Args: {tool_args}")
            logger.info(f"Tool Call: {tool_name} {tool_args}")
            result = await self.tools.execute(name=tool_name, tool_input=tool_args)

            output_content = result.output if getattr(result, "error", None) is None else result.error
            is_error = bool(getattr(result, "error", None))
            if is_error:
                step_success = False
                print(f"Execution Failed: {output_content}")
            else:
                print(f"Execution Success: {output_content}")

            executed_actions.append(
                {
                    "tool": tool_name,
                    "args": tool_args,
                    "status": "failed" if is_error else "success",
                    "output": output_content,
                }
            )

            if "HARDWARE ALERT" in str(output_content):
                step_success = False

        return step_success, executed_actions

    def _safe_json_loads(self, payload: Any) -> Dict[str, Any]:
        if isinstance(payload, dict):
            return payload
        if not isinstance(payload, str):
            return {}
        try:
            return json.loads(payload)
        except json.JSONDecodeError:
            return {}

    def _extract_failure_reason(self, actions: List[dict]) -> str:
        for action in actions:
            output = str(action.get("output", ""))
            if output:
                return output
        return ""

    def _extract_terminate_status(self, actions: List[dict]) -> str:
        for action in actions:
            if action.get("tool") != "terminate":
                continue
            args = action.get("args", {})
            return str(args.get("status", "")).lower()
        return ""

    def _update_runtime_location(self, current_state: Dict[str, Any], step: dict) -> None:
        target = str(step.get("target", ""))
        if step.get("robot") == "Chassis" and step.get("action") == "use_elevator":
            current_state["last_location"] = ELEVATOR_BY_FLOOR.get(target, current_state["last_location"])
            return
        if target in LOCATION_DATABASE:
            current_state["last_location"] = target

    def _save_trace(self) -> None:
        save_dir = "traces"
        os.makedirs(save_dir, exist_ok=True)
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(save_dir, f"run_{timestamp_str}.json")

        with open(filename, "w", encoding="utf-8") as handle:
            json.dump(self.trace_history, handle, indent=2, ensure_ascii=False)

        print(f"\nTrace saved to: {filename}")
        logger.info(f"Trace saved to {filename}")
