import time
import asyncio
from app.tool.base import BaseTool
from app.robot_registry import LOCATION_DATABASE, ELEVATOR_CONFIG

# ==============================================================================
# Mock ROS Client (模拟 ROS 2 接口)
# ==============================================================================
class MockROSClient:
    """
    模拟 ROS 2 Action Client。
    在真实部署时，这里会替换为 rclpy 的 ActionClient 代码。
    """
    async def send_navigation_goal(self, x, y, yaw, bt_name="NavigateToPose"):
        print(f"   [ROS] Sending Goal to Nav2: x={x:.2f}, y={y:.2f}, yaw={yaw:.2f}")
        print(f"   [ROS] Loading Behavior Tree: '{bt_name}.xml'...")
        
        # 模拟硬件执行耗时
        await asyncio.sleep(1.0) 
        
        # 模拟执行成功
        print(f"   [ROS] Behavior Tree '{bt_name}' Execution Succeeded.")
        return True

    async def run_shell_script(self, script_path):
        if not script_path:
            return True
            
        print(f"   [ROS] Running Hardware Script: {script_path}")
        # 模拟脚本执行耗时
        await asyncio.sleep(0.5)
        print(f"   [ROS] Script Execution Finished.")
        return True

    async def execute_elevator_sequence(self, floor_config):
        """执行复杂的跨楼层逻辑 (对应 XML 中的 TakeLiftUp/Down)"""
        bt_name = floor_config["behavior_tree"]
        map_script = floor_config["map_script"]
        icp_script = floor_config["icp_script"]
        
        print(f"   [ROS] Initiating Elevator Sequence: {bt_name}")
        await self.run_shell_script(map_script) # 切图
        await self.run_shell_script(icp_script) # 重定位
        print(f"   [ROS] Elevator Sequence Complete. Robot localized on new floor.")
        return True

# 初始化全局 ROS 客户端实例
ros_client = MockROSClient()


# ==============================================================================
# Robot Tools (工具定义)
# ==============================================================================

class ChassisNavigate(BaseTool):
    name: str = "chassis_navigate"
    description: str = "Moves the mobile base to a location."
    parameters: dict = {
        "type": "object",
        "properties": {
            "target": {"type": "string", "description": "Target location name from map"}
        },
        "required": ["target"]
    }

    async def execute(self, target: str):
        # 1. 模拟故障注入 (用于演示重规划)
        if target == "Starbucks_Outdoor":
             print(f"   [SENSOR] HARDWARE ALERT: Lidar detected blockage on path to {target}!")
             return self.fail_response(f"Navigation Failed: Path to {target} is blocked by construction work.")

        # 2. 查表：获取物理坐标和底层行为树
        location_data = LOCATION_DATABASE.get(target)

        # 3. 处理跨楼层导航 (如果目标是电梯大厅，检查是否隐含了楼层切换逻辑)
        # 注意：这里简化处理，假设移动到电梯口就是为了换层。
        # 实际逻辑中，Planner 通常会显式调用 elevator_control，但有时会直接 navigate。
        
        if not location_data:
            # 容错：如果是未知地点，尝试作为通用点处理（假设坐标为 0,0）
            print(f"   [WARN] Location '{target}' not in registry. Using generic navigation.")
            location_data = {
                "coords": {"x": 0.0, "y": 0.0, "yaw": 0.0},
                "behavior_tree": "NavigateToPose",
                "arrival_script": None
            }

        coords = location_data["coords"]
        bt_name = location_data.get("behavior_tree", "NavigateToPose")
        arrival_script = location_data.get("arrival_script")

        # 4. 发送 ROS 指令
        success = await ros_client.send_navigation_goal(
            x=coords["x"], y=coords["y"], yaw=coords["yaw"], bt_name=bt_name
        )

        if success:
            # 5. 到达后触发后续脚本 (如 detect.sh 或 drop.sh)
            if arrival_script:
                await ros_client.run_shell_script(arrival_script)
            return self.success_response(f"Chassis arrived at [{target}]. Executed BT: {bt_name}")
        else:
            return self.fail_response("ROS Navigation Action failed.")


class DogNavigate(BaseTool):
    name: str = "dog_navigate"
    description: str = "Navigate the quadruped robot (Suitable for narrow/complex terrain)."
    parameters: dict = {
        "type": "object",
        "properties": {
            "target": {"type": "string", "description": "Location name"}
        },
        "required": ["target"]
    }

    async def execute(self, target: str):
        # 狗的逻辑类似，也去查表
        location_data = LOCATION_DATABASE.get(target)
        
        if location_data:
            coords = location_data["coords"]
            # 狗可能用不同的 controller
            await ros_client.send_navigation_goal(coords["x"], coords["y"], coords["yaw"], bt_name="DogQuadrupedController")
        else:
            # 盲跑模式
            print(f"   [ROS] Sending topological graph goal: {target}")

        return self.success_response(f"Dog traversed narrow path and arrived at [{target}].")


class ArmManage(BaseTool):
    name: str = "arm_manage"
    description: str = "Control the robotic arm for manipulation and handover."
    parameters: dict = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string", 
                "enum": ["grasp", "transfer_to_dog", "reset"],
                "description": "Action type"
            },
            "target": {"type": "string", "description": "Target object"}
        },
        "required": ["action", "target"]
    }

    async def execute(self, action: str, target: str):
        # 将语义动作映射为 Shell 脚本或 ROS Service
        script = None
        
        if action == "grasp" or "obtain" in action:
            script = "/home/agv/armws/coffee_ws/detect.sh" # 复用 XML 中的 detect 脚本
        elif action == "transfer_to_dog":
            script = "/home/agv/armws/coffee_ws/drop.sh"   # 复用 XML 中的 drop 脚本
            
        if script:
            await ros_client.run_shell_script(script)
            return self.success_response(f"Arm execution of [{action}] completed via script.")
            
        return self.success_response(f"Arm performed [{action}] on [{target}] with adaptive force control.")


class ElevatorControl(BaseTool):
    name: str = "elevator_control"
    description: str = "Interact with the IoT Elevator System (Digital Interface)."
    parameters: dict = {
        "type": "object",
        "properties": {
            "command": {"type": "string", "description": "Command (e.g., call_to_5F, goto_1F)"}
        },
        "required": ["command"]
    }

    async def execute(self, command: str):
        # 解析指令，例如 "goto_4F" -> "4F"
        floor = None
        if "4F" in command: floor = "4F"
        elif "1F" in command: floor = "1F"
        elif "5F" in command: floor = "5F"
        elif "2F" in command: floor = "2F"

        if floor and floor in ELEVATOR_CONFIG:
            # 执行 XML 中定义的切图和重定位逻辑
            config = ELEVATOR_CONFIG[floor]
            await ros_client.execute_elevator_sequence(config)
            return self.success_response(f"Elevator arrived at {floor}. Map switched and Robot relocalized.")
        
        # 简单的 IoT 信号模拟
        return self.success_response(f"IoT Elevator received command: [{command}].")