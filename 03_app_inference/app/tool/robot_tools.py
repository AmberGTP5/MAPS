import time
import asyncio
import json
from app.tool.base import BaseTool
from app.robot_registry import LOCATION_DATABASE

# ==============================================================================
# Hardware Configuration (硬件配置)
# ==============================================================================

DOG_IP = "10.4.128.242"
DOG_PORT = 8000

CHASSIS_IP = "10.4.162.106"  
CHASSIS_PORT = 8000


# ==============================================================================
# Real Hardware Clients (TCP Socket 通信)
# ==============================================================================

class RobotSocketClient:
    def __init__(self, ip, port, robot_name="ROBOT"):
        self.ip = ip
        self.port = port
        self.name = robot_name

    async def send_task(self, location_name, coords, action="navigate"):
        print(f"   [{self.name}]  Connecting to {self.ip}:{self.port}...")
        
        payload = {
            "location_name": location_name,
            "coords": coords,
            "action": action
        }
        
        try:
            reader, writer = await asyncio.open_connection(self.ip, self.port)
            
            json_str = json.dumps(payload)
            print(f"   [{self.name}]  Sending Payload (Waiting for completion)...")
            writer.write(json_str.encode('utf-8'))
            await writer.drain()
            
            # 等待响应，设置超时时间 (例如 600秒)
            data = await asyncio.wait_for(reader.read(4096), timeout=600)
            
            response_str = data.decode('utf-8')
            print(f"   [{self.name}]  FINALLY Received: {response_str}")
            
            writer.close()
            await writer.wait_closed()
            
            if "success" in response_str.lower():
                return True, response_str
            return False, response_str

        except asyncio.TimeoutError:
            return False, "TIMEOUT: Robot took too long (>600s) to finish task."
        except Exception as e:
            return False, f"Connection Error: {str(e)}"

# 初始化两个真实客户端
dog_client = RobotSocketClient(DOG_IP, DOG_PORT, robot_name="DOG")
chassis_client = RobotSocketClient(CHASSIS_IP, CHASSIS_PORT, robot_name="CHASSIS")


# [删除] MockROSClient 类和 ros_client 实例已在此处移除


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
        # 1. 模拟故障注入 (星巴克堵车)
        # if target == "Starbucks_Outdoor":
        #      print(f"   [SENSOR]  HARDWARE ALERT: Lidar detected blockage on path to {target}!")
        #      return self.fail_response(f"Navigation Failed: Path to {target} is blocked by construction work.")

        # 2. 查表
        location_data = LOCATION_DATABASE.get(target)
        
        # 调用真实底座 Client
        if location_data and "coords" in location_data:
            coords = location_data["coords"]
            
            # 发送 TCP 请求 (发送 x, y, yaw)
            success, message = await chassis_client.send_task(target, coords)
            
            if success:
                # [修改] 删除本地 arrival_script 执行逻辑
                # 假设底座到达后，会根据 target 自动处理所需的后续动作（如识别等），或者完全由 Agent 后续指令控制
                return self.success_response(f" Chassis Real-Response: {message}")
            else:
                return self.fail_response(f"Chassis Communication Failed: {message}")

            # if success:
            #     # [修改] 删除本地 arrival_script 执行逻辑
            #     # 假设底座到达后，会根据 target 自动处理所需的后续动作（如识别等），或者完全由 Agent 后续指令控制
            #     return self.fail_response(f"Chassis Communication Failed: {message}")
        else:
             # 如果是未知地点
            print(f"   [CHASSIS-FILTER]  Target '{target}' has no coords. Skipped.")
            return self.success_response(f" Chassis action on abstract target [{target}] handled locally.")


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
        
        # 调用真实狗 Client
        if location_data and "coords" in location_data:
            coords = location_data["coords"]
            
            # 发送 TCP 请求
            success, message = await dog_client.send_task(target, coords)
            
            if success:
                return self.success_response(f" Dog Real-Response: {message}")
            else:
                return self.fail_response(f"Dog Communication Failed: {message}")
        else:
            print(f"   [DOG-FILTER]  Target '{target}' has no coords. Skipped.")
            return self.success_response(f" Dog action on abstract target [{target}] handled locally.")

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
        # 纯语义指令，不包含任何路径
        payload_data = {
            "module": "arm_system",    # 告诉底座这是机械臂的任务
            "operation": action,       # 具体操作: "grasp" / "transfer_to_dog"
            "target_object": target    # 操作对象: "Coffee"
        }
        
        # 发送给底座 (Chassis)
        success, message = await chassis_client.send_task(
            location_name="Arm_Action", 
            coords=payload_data, # 把语义指令塞在这里
            action="execute_script"
        )
            
        if success:
            return self.success_response(f" Arm Successfully Executed: {action}. Hardware msg: {message}")
        else:
            return self.fail_response(f" Arm Action Failed. Reason: {message}")


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

        if floor:
            # 纯语义指令
            payload_data = {
                "module": "elevator_system", # 告诉底座这是电梯任务
                "operation": "switch_floor", # 操作类型
                "target_floor": floor        # 目标楼层
            }
            
            success, message = await chassis_client.send_task(
                location_name=f"Elevator_{floor}",
                coords=payload_data,
                action="system_control"
            )
            
            if success:
                return self.success_response(f" Elevator Success: Arrived at {floor}. Map switched. Msg: {message}")
            else:
                return self.fail_response(f" Elevator Failed to switch floor. Hardware Error: {message}")
        
        return self.fail_response(f" Invalid Elevator Command: [{command}]. Please specify 1F, 2F, 4F, or 5F.")