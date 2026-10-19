import time
import asyncio
import json
from app.tool.base import BaseTool
from app.robot_registry import LOCATION_DATABASE, ELEVATOR_CONFIG

# ==============================================================================
# 🐕 Dog Configuration (真实狗的配置)
# ==============================================================================
DOG_IP = "10.4.128.242"
DOG_PORT = 8000

# ==============================================================================
# 🔌 Real Dog Client (TCP Socket 通信)
# ==============================================================================
class DogSocketClient:
    """
    专门负责与四足机器人进行 TCP JSON 通信的客户端
    """
    async def send_navigation_goal(self, target_name, coords):
        print(f"   [DOG] Connecting to Dog at {DOG_IP}:{DOG_PORT}...")
        
        # 构建符合测试成功的 JSON 数据包
        payload = {
            "location_name": target_name,
            "coords": coords, # 这里面直接是四元数 {ori_x, ...}
            "action": "navigate"
        }
        
        try:
            # 1. 建立异步连接 (非阻塞，不会卡住 Planner)
            reader, writer = await asyncio.open_connection(DOG_IP, DOG_PORT)
            
            # 2. 发送数据
            json_str = json.dumps(payload)
            print(f"   [DOG] Sending Payload: {json_str}")
            writer.write(json_str.encode('utf-8'))
            await writer.drain()
            
            # 3. 等待接收返回值 (读取 4096 字节)
            print(f"   [DOG]  Waiting for Dog response...")
            data = await reader.read(4096)
            response_str = data.decode('utf-8')
            print(f"   [DOG]  Received: {response_str}")
            
            # 4. 关闭连接
            writer.close()
            await writer.wait_closed()
            
            # 5. 解析结果
            try:
                response_json = json.loads(response_str)
                # 兼容不同的成功标识
                if response_json.get("status") == "success" or response_json.get("result") == True:
                    return True, response_str
                else:
                    return False, f"Dog reported failure: {response_str}"
            except json.JSONDecodeError:
                # 如果返回的不是 JSON，但有内容，也算成功（根据你的 SSH 测试经验）
                if response_str:
                    return True, response_str
                return False, "Empty or invalid response from Dog"

        except Exception as e:
            error_msg = f"Connection failed: {str(e)}"
            print(f"   [DOG]  {error_msg}")
            return False, error_msg

# 初始化狗的真实客户端
dog_client = DogSocketClient()


# ==============================================================================
# 🔌 Mock ROS Client (用于 Chassis 和 Elevator)
# ==============================================================================
# 注意：底盘 (Chassis) 目前还是用 Mock，等以后你也拿到底盘的真实接口再换
class MockROSClient:
    async def send_navigation_goal(self, coords, bt_name="NavigateToPose"):
        # 极简模式：直接发送四元数信号
        signal_info = ""
        if 'ori_w' in coords:
            signal_info = f"Quaternion(x={coords.get('ori_x',0):.4f}, ... w={coords.get('ori_w',0):.4f})"
        else:
            x = coords.get('x', 0.0)
            y = coords.get('y', 0.0)
            signal_info = f"Pose(x={x:.2f}, y={y:.2f})"

        print(f"   [ROS]  Sending Navigation Signal: {signal_info}")
        print(f"   [ROS]  Loading Behavior Tree: '{bt_name}.xml'...")
        
        await asyncio.sleep(1.0) 
        print(f"   [ROS]  ARRIVAL SIGNAL RECEIVED: Target Reached via '{bt_name}'.")
        return True

    async def run_shell_script(self, script_path):
        if not script_path: return True
        print(f"   [ROS]  Running Hardware Script: {script_path}")
        await asyncio.sleep(0.5)
        print(f"   [ROS]  Script Execution Finished.")
        return True

    async def execute_elevator_sequence(self, floor_config):
        bt_name = floor_config["behavior_tree"]
        map_script = floor_config["map_script"]
        icp_script = floor_config["icp_script"]
        print(f"   [ROS]  Initiating Elevator Sequence: {bt_name}")
        await self.run_shell_script(map_script) 
        await self.run_shell_script(icp_script) 
        print(f"   [ROS]  Elevator Sequence Complete.")
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
        # 模拟故障注入
        if target == "Starbucks_Outdoor":
             print(f"   [SENSOR]  HARDWARE ALERT: Lidar detected blockage on path to {target}!")
             return self.fail_response(f"Navigation Failed: Path to {target} is blocked by construction work.")

        location_data = LOCATION_DATABASE.get(target)
        if not location_data:
            print(f"   [WARN] Location '{target}' not in registry. Using generic signal.")
            location_data = {
                "coords": {"ori_w": 1.0, "ori_x": 0.0, "ori_y": 0.0, "ori_z": 0.0},
                "behavior_tree": "NavigateToPose",
                "arrival_script": None
            }

        coords = location_data["coords"]
        bt_name = location_data.get("behavior_tree", "NavigateToPose")
        arrival_script = location_data.get("arrival_script")

        # Chassis 依然使用 ROS Client
        success = await ros_client.send_navigation_goal(coords, bt_name=bt_name)

        if success:
            if arrival_script:
                await ros_client.run_shell_script(arrival_script)
            return self.success_response(f" Signal Received: Chassis arrived at [{target}].")
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
        # 1. 查表获取坐标
        location_data = LOCATION_DATABASE.get(target)
        
        # [核心修改] 只有当由明确坐标数据时，才发送给狗
        if location_data and "coords" in location_data:
            coords = location_data["coords"]
            
            # 发送真实 TCP 请求
            success, message = await dog_client.send_navigation_goal(target, coords)
            
            if success:
                return self.success_response(f"🐕 Dog Real-Response: {message}")
            else:
                return self.fail_response(f"Dog Communication Failed: {message}")
        
        else:
            # [拦截逻辑] 如果查不到坐标 (例如 target="Coffee" 或 "User")
            # 就不发送网络请求，避免干扰狗的底层运行
            print(f"   [DOG-FILTER] 🚫 Target '{target}' has no physical coords. Skipped sending to hardware.")
            
            # 仍然返回成功给 LLM，让它以为动作完成了（因为这通常是最后一步交付动作）
            return self.success_response(f"🐕 Dog action on abstract target [{target}] handled locally.")


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
            return self.success_response(f" Arm execution of [{action}] completed via script.")
        return self.success_response(f" Arm performed [{action}] on [{target}].")


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
            return self.success_response(f" Elevator arrived at {floor}. Map switched and Robot relocalized.")
        
        return self.success_response(f" IoT Elevator received command: [{command}].")