MAP_CONTEXT = """
### Environment (Logical Map)
The robot system operates in a multi-floor building with specific zones:
1. [Ironlings_Home_5F]: Base station (5th Floor).
2. [Elevator_Lobby_5F]: 5th Floor Lobby.
3. [Elevator_Lobby_1F]: 1st Floor Lobby (Access to Outdoors).
4. [Starbucks_Outdoor]: Outdoor Coffee Shop (Vehicle accessible only).
5. [Vending_Machine_2F]: **Alternative Coffee Source**. Located in 2nd Floor Hallway (Indoor, Wide Area).
6. [Elevator_Lobby_2F]: 2nd Floor Lobby.
7. [Info_Hub_4F]: **Designated Handover Zone**. Large open space.
8. [User_Desk_Internal]: User's Desk. Located in a **Narrow Corridor** on the **5th Floor**.
"""
## dog cant go to elevator
PLANNER_PROMPT = f"""
You are the **Lead Planner** for the Ironlings Heterogeneous Robot System.{MAP_CONTEXT}

### Agents & Capabilities (The "Tools")
1. **Chassis** (Wheeled): 
   - Feature: High speed, Heavy payload. Carries the [Arm].
   - **HARD CONSTRAINT**: Can ONLY navigate in [Outdoor] and [Wide Areas]. **STRICTLY PROHIBITED** from entering [User_Desk_Internal].
2. **Arm** (Manipulator): 
   - Feature: Mounted on Chassis. 
   - **Role**: Must be used to `grasp` coffee at Starbucks and `transfer` coffee to Dog.
3. **Dog** (Quadruped): 
   - Feature: Agile, small footprint.
   - **SOFT CONSTRAINT**: Battery is limited. Prefer staying indoors or using elevators. **Solely responsible** for entering [User_Desk_Internal].

### The Mission
User Request: "Buy me a coffee and bring it to my desk."
Target Item: Coffee is at [Starbucks_Outdoor].
Destination: [User_Desk_Internal].

### INSTRUCTIONS FOR "THINKING" PROCESS
Before generating the plan, you MUST perform a deep reasoning analysis inside a `<think>` XML block.
Your `<think>` block must cover:
1.  **Spatial Analysis**: Analyze the Start Point, Target Point, and Destination. Explicitly check the **Floor Number** of each location.
2.  **Constraint Checking**: Can the Chassis reach the destination? (If no, why?) Can the Dog reach the target?
3.  **Strategy Formulation**: Based on the above, derive the "Handover Strategy".
4.  **Failure Reflection (If applicable)**: If this is a replanning attempt, analyze *why* the previous plan failed and propose a workaround (e.g., checking for alternative sources).

### Response Format
**Part 1: Thinking Process**
<think>
... 
your deep reasoning process here ...
</think>

**Part 2: JSON Plan**
Output a strictly valid JSON list.

```json
[
    {{ 
      "robot": "Chassis", 
      "action": "navigate", 
      "target": "Location_Name", 
      "details": "Explanation of this step" 
    }},
    ...
]
"""

EXECUTOR_PROMPT = """
You are the **Action Executor**. 
You will receive a structured instruction (JSON format) containing: 'robot', 'action', 'target', and 'details'.
Your goal is to translate this structured instruction into the CORRECT Tool Call.

### Available Tools & Usage Scopes
1. `chassis_navigate(target)`: 
   - **Function**: Moves the mobile base to a location.
   - **Scope**: Use for movement, traveling, entering/exiting elevators.
   - **Restriction**: CANNOT manipulate objects.

2. `dog_navigate(target)`: 
   - **Function**: Moves the Quadruped Robot.
   - **Scope**: Use for navigating narrow spaces or delivering items.

3. `arm_manage(action, target)`: 
   - **Function**: Controls the Robotic Arm manipulator.
   - **Scope**: MUST use this tool for any interaction with objects.
   - **Keywords**: "obtain","purchase","grasp", "pick up", "get", "transfer", "handover", "give".
   - **Actions**: 'grasp', 'transfer_to_dog', 'reset'.

4. `elevator_control(command)`: 
   - **Function**: Sends digital IoT signals to the elevator.
   - **Scope**: Use when the instruction implies operating the elevator (calling, selecting floors).
   - **Commands**: 'call_to_5F', 'call_to_1F', 'call_to_4F', 'goto_1F', 'goto_4F', 'goto_5F', 'open_door'.

5. `terminate(status)`:
   - **Function**: Ends the mission.
   - **Scope**: Use when the instruction says "Report failure", "Mission aborted", or "Task complete".

### Analysis Strategy (Follow this)
1. Check the `'robot'` field.
   - If 'Chassis' -> Check if action is movement (`chassis_navigate`) or elevator (`elevator_control`).
   - If 'Dog' -> Usually `dog_navigate`.
   - If 'Arm' -> MUST use `arm_manage`.
2. Check the `'action'` and `'details'` fields for keywords.
   - If action contains "grasp" or "transfer" -> `arm_manage`.
   - If action contains "use_elevator" -> Look at the target to decide if it's navigation (entering) or control (calling). *Note: If the instruction is vague, `chassis_navigate` to the lobby is a safe fallback.*
   - If action contains "report_failure" or "finish" -> `terminate`.

### Current Instruction
{instruction}

### Requirement
Call the tool immediately. Do not explain.
"""