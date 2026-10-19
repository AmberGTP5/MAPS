import json
import asyncio
import os
import re
from datetime import datetime
from loguru import logger  # 引入 loguru 记录日志文件
from app.tool.tool_collection import ToolCollection
from app.tool.robot_tools import ChassisNavigate, DogNavigate, ArmManage, ElevatorControl
from app.tool.terminate import Terminate
from app.prompt.instructions import PLANNER_PROMPT, EXECUTOR_PROMPT
from app.llm import LLM

class RobotCoordinator:
    def __init__(self):
        # 初始化工具集
        self.tools = ToolCollection(
            ChassisNavigate(),
            DogNavigate(),
            ArmManage(),
            ElevatorControl(),
            Terminate()
        )
        
        logger.info("Initializing Planner LLM (Smart)...")
        self.planner_llm = LLM(config_key="planner")
        
        logger.info("Initializing Executor LLM (Fast)...")
        self.executor_llm = LLM(config_key="executor")

        # 初始化 Trace 历史 (用于生成 trace JSON)
        self.trace_history = {
            "meta": {
                "task": "",
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "status": "pending"
            },
            "attempts": []
        }

    async def run(self, user_request):
        # 1. 设置任务元数据
        logger.info(f" Robot Coordinator started. Task: {user_request}")
        self.trace_history["meta"]["task"] = user_request

        # 初始化状态追踪
        current_state = {
            "last_location": "Robot_Station_5F", # 假设初始位置
            "failed_reason": None
        }
        
        max_retries = 3
        attempt = 0
        
        while attempt < max_retries:
            attempt += 1
            print(f"\n Planning Phase (Attempt {attempt})...")
            logger.info(f"--- Planning Phase (Attempt {attempt}) ---")
            
            # --- 1. 动态构建 Prompt ---
            prompt = self._build_dynamic_prompt(user_request, current_state)
            
            # --- 2. 生成计划 ---
            reasoning, plan_steps = await self._generate_plan(prompt)
            
            if not plan_steps:
                print(" Planner failed to generate steps.")
                logger.error("Planner failed to generate steps.")
                break

            # 打印和记录思考过程
            print(f"\n **Planner Reasoning**:")
            if "<think>" not in reasoning:
                print(f"<think>\n{reasoning}\n</think>")
            else:
                print(f"{reasoning}")
            
            print("-" * 50)
            print(f" Plan Generated with {len(plan_steps)} steps.")
            for i, step in enumerate(plan_steps):
                print(f"{i+1}. {step}")
            print("-" * 50)

            # 记录当前 Attempt 的 Trace 数据
            current_attempt_trace = {
                "attempt_id": attempt,
                "reasoning": reasoning,
                "plan": plan_steps,
                "execution_log": [],
                "outcome": "unknown"
            }

            # --- 3. 执行阶段 ---
            print(f"\n Execution Phase (Attempt {attempt})...")
            logger.info(f"--- Execution Phase (Attempt {attempt}) ---")
            
            plan_success = True 
            
            for i, step in enumerate(plan_steps):
                step_desc = f"Step {i+1}/{len(plan_steps)}: {step.get('action', 'unknown')} -> {step.get('target', 'unknown')}"
                print(f"\n Processing {step_desc}")
                logger.info(f"[Exec] {step_desc}")
                
                # 执行步骤
                is_step_success, actions = await self._execute_step(step)
                
                #  更新 Trace 执行日志
                current_attempt_trace["execution_log"].append({
                    "step_index": i + 1,
                    "instruction": step,
                    "actions": actions,
                    "success": is_step_success
                })

                # 更新最后已知位置 (用于重规划上下文)
                if 'target' in step: 
                    current_state['last_location'] = step['target']
                
                #  故障检查逻辑
                if not is_step_success:
                    print(f" ALERT: Step {i+1} Failed! Stopping execution.")
                    
                    # 提取错误信息
                    error_msg = "Unknown Error"
                    if actions and 'output' in actions[0]:
                        error_msg = actions[0]['output']
                    
                    logger.error(f"Execution failed at step {i+1}: {error_msg}")
                    current_state['failed_reason'] = error_msg
                    
                    plan_success = False
                    current_attempt_trace["outcome"] = "failure"
                    current_attempt_trace["error"] = error_msg
                    break # 跳出 for 循环，准备重规划
                
                # 检查是否包含硬件警告 (即便 tool 返回 success，内容里可能有 HARDWARE ALERT)
                for action in actions:
                    output_str = str(action.get('output', ''))
                    if "HARDWARE ALERT" in output_str:
                        print(f" HARDWARE FAILURE DETECTED: {output_str}")
                        logger.error(f"Hardware Alert: {output_str}")
                        current_state['failed_reason'] = output_str
                        plan_success = False
                        is_step_success = False # 强制标记为失败
                        current_attempt_trace["outcome"] = "failure"
                        current_attempt_trace["error"] = output_str
                        break

                if not plan_success:
                    break

                # 检查是否调用了 terminate 工具
                for action in actions:
                    if action['tool'] == 'terminate':
                        print(" Task Terminated by Robot.")
                        logger.info("Task terminated by tool.")
                        current_attempt_trace["outcome"] = "terminated"
                        self.trace_history["attempts"].append(current_attempt_trace)
                        self.trace_history["meta"]["status"] = "success"
                        self._save_trace()
                        return

            # 保存当前 Attempt 的 Trace
            self.trace_history["attempts"].append(current_attempt_trace)

            # --- 4. 结果判定 ---
            if plan_success:
                print("\n Task Completed Successfully!")
                logger.info("Mission Completed Successfully.")
                self.trace_history["meta"]["status"] = "success"
                self._save_trace() #  成功保存
                return
            else:
                print(f"Preparing to Replan (Context: {current_state['failed_reason']})...")
                logger.warning(f"Preparing to replan. Reason: {current_state['failed_reason']}")
        
        # 超过最大重试次数
        print("\n Critical Failure: Max retries reached. Mission Aborted.")
        logger.critical("Max retries reached. Mission Failed.")
        self.trace_history["meta"]["status"] = "failed_max_retries"
        # 失败也保存
        self._save_trace() 

    def _build_dynamic_prompt(self, user_request, current_state):
        """构建带有上下文的 Prompt"""
        base_prompt = PLANNER_PROMPT
        
        if not current_state['failed_reason']:
            return base_prompt + f"\n\nUser Request: {user_request}"
            
        print(" Injecting Failure Context into Planner...")
        replan_instruction = f"""
        
        ###  URGENT: PREVIOUS PLAN FAILED 
        The previous execution halted due to an error.
        - **Current Location**: {current_state['last_location']}
        - **Failure Reason**: "{current_state['failed_reason']}"
        
        **YOUR NEW TASK**:
        Generate a NEW plan starting from the **Current Location** to fulfill the original request: "{user_request}".
        You MUST propose an alternative solution to bypass the failure in your <think> block.
        """
        return base_prompt + replan_instruction

    async def _generate_plan(self, prompt_or_request):
        # 兼容逻辑
        if "You are the **Lead Planner**" in prompt_or_request:
            messages = [{"role": "user", "content": prompt_or_request}]
        else:
            messages = [
                {"role": "system", "content": PLANNER_PROMPT},
                {"role": "user", "content": f"User Request: {prompt_or_request}"}
            ]
        
        response = await self.planner_llm.chat(messages=messages)
        content = response.content

        reasoning = "No reasoning provided."
        json_content = "[]"
        
        # 提取 <think>
        think_match = re.search(r"<think>(.*?)</think>", content, re.DOTALL)
        if think_match:
            reasoning = think_match.group(1).strip()
        else:
            if "```json" in content:
                reasoning = content.split("```json")[0].strip()

        # 提取 JSON
        json_match = re.search(r"```json(.*?)```", content, re.DOTALL)
        if json_match:
            json_content = json_match.group(1).strip()
        else:
            if '[' in content:
                start_index = content.find('[')
                json_content = content[start_index:]
            else:
                json_content = content

        try:
            steps = json.loads(json_content)
            return reasoning, steps
        except json.JSONDecodeError:
            print(f" JSON Decode Error. Raw Output: {content}")
            logger.error(f"JSON Decode Error: {content}")
            return reasoning, []

    async def _execute_step(self, step_text):
        # 这里的 instruction=json.dumps(step_text) 确保 step_text 是字符串
        instruction_str = json.dumps(step_text) if isinstance(step_text, dict) else str(step_text)
        current_prompt = EXECUTOR_PROMPT.format(instruction=instruction_str)
        
        messages = [
            {"role": "system", "content": current_prompt},
            {"role": "user", "content": "Generate Tool Call."}
        ]

        # 调用 LLM，传入 tools 定义
        response = await self.executor_llm.chat(messages=messages, tools=self.tools.to_params())
        
        step_success = True
        executed_actions = []

        if not response.tool_calls:
            print(f" Warning: Model did not call any tool.")
            return False, []

        for tool_call in response.tool_calls:
            tool_name = tool_call.function.name
            tool_args_str = tool_call.function.arguments
            
            try:
                tool_args = json.loads(tool_args_str) if isinstance(tool_args_str, str) else tool_args_str
            except:
                tool_args = {}
            
            print(f" Calling Tool: {tool_name} | Args: {tool_args}")
            logger.info(f"Tool Call: {tool_name} {tool_args}")
            
            # 执行工具
            result = await self.tools.execute(name=tool_name, tool_input=tool_args)

            # 检查结果
            output_content = ""
            is_error = False
            
            if hasattr(result, 'error') and result.error:
                print(f" Execution Failed: {result.error}")
                output_content = result.error
                is_error = True
                step_success = False
            else:
                output_content = result.output if hasattr(result, 'output') else str(result)
                print(f" Execution Success: {output_content}")

            executed_actions.append({
                "tool": tool_name,
                "args": tool_args,
                "status": "failed" if is_error else "success",
                "output": output_content
            })
            
        return step_success, executed_actions
    

    def _save_trace(self):
        """保存运行轨迹为 JSON"""
        save_dir = "traces"
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
            
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{save_dir}/run_{timestamp_str}.json"
        
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(self.trace_history, f, indent=2, ensure_ascii=False)
            print(f"\n Trace saved to: {filename}")
            logger.info(f"Trace saved to {filename}")
        except Exception as e:
            print(f" Failed to save trace: {e}")
            logger.error(f"Failed to save trace: {e}")