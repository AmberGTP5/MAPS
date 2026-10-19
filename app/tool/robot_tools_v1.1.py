import time
import asyncio
from app.tool.base import BaseTool
from app.robot_registry import LOCATION_DATABASE, ELEVATOR_CONFIG

# ==============================================================================
# Mock ROS Client (模拟 ROS 2 接口)
# ==============================================================================
class MockROSClient:
    async def send_navigation_goal(self, coords, bt_name="NavigateToPose"):
        # 极简模式：直接发送四元数信号
        # 不再判断 x, y，假设传入的 coords 里就是我们要发的信号
        
        # 提取四元数信息用于打印 (兼容性处理：如果只有 yaw 就显示 yaw，如果有 ori 就显示 ori)
        signal_info = ""
        if 'ori_w' in coords:
            signal_info = f"Quaternion(x={coords.get('ori_x',0):.4f}, y={coords.get('ori_y',0):.4f}, z={coords.get('ori_z',0):.4f}, w={coords.get('ori_w',0):.4f})"
        else:
            # 兼容旧数据的显示
            x = coords.get('x', 0.0)
            y = coords.get('y', 0.0)
            yaw = coords.get('yaw', 0.0)
            signal_info = f"Pose(x={x:.2f}, y={y:.2f}, yaw={yaw:.2f})"

        print(f"   [ROS] Sending Navigation Signal: {signal_info}")
        print(f"   [ROS] Loading Behavior Tree: '{bt_name}.xml'...")
        
        await asyncio.sleep(1.0) 
        
        print(f"   [ROS] ARRIVAL SIGNAL RECEIVED: Target Reached via '{bt_name}'.")
        return True

    async def run_shell_script(self, script_path):
        if not script_path:
            return True
        print(f"   [ROS] Running Hardware Script: {script_path}")
        await asyncio.sleep(0.5)
        print(f"   [ROS] Script Execution Finished.")
        return True

    async def execute_elevator_sequence(self, floor_config):
        bt_name = floor_config["behavior_tree"]
        map_script = floor_config["map_script"]
        icp_script = floor_config["icp_script"]
        print(f"   [ROS] Initiating Elevator Sequence: {bt_name}")
        await self.run_shell_script(map_script) 
        await self.run_shell_script(icp_script) 
        print(f"   [ROS] Elevator Sequence Complete. Robot localized on new floor.")
        return True

ros_client = MockROSClient()


# ==============================================================================
# 🛠️ Robot Tools (工具定义)
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
        # 1. 模拟故障注入
        if target == "Starbucks_Outdoor":
             print(f"   [SENSOR] HARDWARE ALERT: Lidar detected blockage on path to {target}!")
             return self.fail_response(f"Navigation Failed: Path to {target} is blocked by construction work.")

        # 2. 查表
        location_data = LOCATION_DATABASE.get(target)
        if not location_data:
            print(f"   [WARN] Location '{target}' not in registry. Using generic signal.")
            location_data = {
                "coords": {"ori_w": 1.0, "ori_x": 0.0, "ori_y": 0.0, "ori_z": 0.0}, # 默认四元数
                "behavior_tree": "NavigateToPose",
                "arrival_script": None
            }

        coords = location_data["coords"]
        bt_name = location_data.get("behavior_tree", "NavigateToPose")
        arrival_script = location_data.get("arrival_script")

        # 3. 发送信号
        success = await ros_client.send_navigation_goal(coords, bt_name=bt_name)

        if success:
            if arrival_script:
                await ros_client.run_shell_script(arrival_script)
            return self.success_response(f"Signal Received: Chassis arrived at [{target}].")
        else:
            return self.fail_response("ROS Navigation Action failed.")


class DogNavigate(BaseTool):
    name: str = "dog_navigate"
    description: str = "Navigate the quadruped robot."
    parameters: dict = {
        "type": "object",
        "properties": {
            "target": {"type": "string", "description": "Location name"}
        },
        "required": ["target"]
    }

    async def execute(self, target: str):
        location_data = LOCATION_DATABASE.get(target)
        if location_data:
            coords = location_data["coords"]
            await ros_client.send_navigation_goal(coords, bt_name="DogQuadrupedController")
        else:
            print(f"   [ROS] Sending topological graph goal: {target}")

        return self.success_response(f"Signal Received: Dog arrived at [{target}].")


class ArmManage(BaseTool):
    name: str = "arm_manage"
    description: str = "Control the robotic arm."
    parameters: dict = {
        "type": "object",
        "properties": {
            "action": {"type": "string", "enum": ["grasp", "transfer_to_dog", "reset"], "description": "Action type"},
            "target": {"type": "string", "description": "Target object"}
        },
        "required": ["action", "target"]
    }

    async def execute(self, action: str, target: str):
        script = None
        if action == "grasp" or "obtain" in action:
            script = "/home/agv/armws/coffee_ws/detect.sh"
        elif action == "transfer_to_dog":
            script = "/home/agv/armws/coffee_ws/drop.sh"
            
        if script:
            await ros_client.run_shell_script(script)
            return self.success_response(f"Arm execution of [{action}] completed via script.")
        return self.success_response(f"Arm performed [{action}] on [{target}].")


class ElevatorControl(BaseTool):
    name: str = "elevator_control"
    description: str = "Interact with the IoT Elevator System."
    parameters: dict = {
        "type": "object",
        "properties": {
            "command": {"type": "string", "description": "Command (e.g., goto_4F)"}
        },
        "required": ["command"]
    }

    async def execute(self, command: str):
        floor = None
        if "4F" in command: floor = "4F"
        elif "1F" in command: floor = "1F"
        elif "5F" in command: floor = "5F"
        elif "2F" in command: floor = "2F"

        if floor and floor in ELEVATOR_CONFIG:
            config = ELEVATOR_CONFIG[floor]
            await ros_client.execute_elevator_sequence(config)
            return self.success_response(f"Elevator arrived at {floor}. Map switched and Robot relocalized.")
        
        return self.success_response(f"IoT Elevator received command: [{command}].")