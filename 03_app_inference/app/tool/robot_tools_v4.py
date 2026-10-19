import time
import asyncio
import json
from app.tool.base import BaseTool
from app.robot_registry import LOCATION_DATABASE, ELEVATOR_CONFIG

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

# class RobotSocketClient:
#     """通用机器人类，处理 TCP 连接和数据发送"""
#     def __init__(self, ip, port, robot_name="ROBOT"):
#         self.ip = ip
#         self.port = port
#         self.name = robot_name

#     async def send_task(self, location_name, coords, action="navigate"):
#         print(f"   [{self.name}]  Connecting to {self.ip}:{self.port}...")
        
#         # 构建 JSON 数据包
#         payload = {
#             "location_name": location_name,
#             "coords": coords, # 狗是四元数，底座是 x,y,yaw
#             "action": action
#         }
        
#         try:
#             # 1. 建立异步连接
#             reader, writer = await asyncio.open_connection(self.ip, self.port)
            
#             # 2. 发送数据
#             json_str = json.dumps(payload)
#             print(f"   [{self.name}]  Sending Payload: {json_str}")
#             writer.write(json_str.encode('utf-8'))
#             await writer.drain()
            
#             # 3. 等待回执
#             print(f"   [{self.name}]  Waiting for response...")
#             data = await reader.read(4096)
#             response_str = data.decode('utf-8')
#             print(f"   [{self.name}]  Received: {response_str}")
            
#             # 4. 关闭
#             writer.close()
#             await writer.wait_closed()
            
#             # 5. 简单解析
#             if "success" in response_str.lower() or "received" in response_str.lower():
#                 return True, response_str
#             return True, response_str # 只要连通了通常就算成功，根据你的Server逻辑

#         except Exception as e:
#             error_msg = f"Connection failed: {str(e)}"
#             print(f"   [{self.name}]  {error_msg}")
#             return False, error_msg

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
            # [关键修改] 使用 open_connection 并手动控制 reader
            # 这里的建立连接通常很快
            reader, writer = await asyncio.open_connection(self.ip, self.port)
            
            json_str = json.dumps(payload)
            print(f"   [{self.name}]  Sending Payload (Waiting for completion)...")
            writer.write(json_str.encode('utf-8'))
            await writer.drain()
            
            # [关键修改] 读取响应时，因为 Server 是阻塞的
            # 这一步可能会卡住很久（例如 2分钟），这是符合预期的！
            # asyncio.wait_for 可以设置超时，我们这里设为 None (无限等待) 或者 600秒
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


# ==============================================================================
# Mock ROS Client (仅用于 电梯 和 机械臂 脚本执行)
# ==============================================================================
# 注：如果机械臂和电梯也有 Socket 接口，也可以换掉这个 Mock
class MockROSClient:
    async def run_shell_script(self, script_path):
        if not script_path: return True
        print(f"   [HARDWARE]  Running Local Script: {script_path}")
        await asyncio.sleep(0.5)
        print(f"   [HARDWARE]  Script Execution Finished.")
        return True

    async def execute_elevator_sequence(self, floor_config):
        bt_name = floor_config["behavior_tree"]
        map_script = floor_config["map_script"]
        icp_script = floor_config["icp_script"]
        print(f"   [ELEVATOR]  Initiating Sequence: {bt_name}")
        await self.run_shell_script(map_script) 
        await self.run_shell_script(icp_script) 
        print(f"   [ELEVATOR]  Sequence Complete.")
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
        # 1. 模拟故障注入 (星巴克堵车)
        if target == "Starbucks_Outdoor":
             print(f"   [SENSOR]  HARDWARE ALERT: Lidar detected blockage on path to {target}!")
             return self.fail_response(f"Navigation Failed: Path to {target} is blocked by construction work.")

        # 2. 查表
        location_data = LOCATION_DATABASE.get(target)
        
        # [修改点] 调用真实底座 Client
        if location_data and "coords" in location_data:
            coords = location_data["coords"]
            
            # 发送 TCP 请求 (发送 x, y, yaw)
            success, message = await chassis_client.send_task(target, coords)
            
            if success:
                # 如果有到达脚本 (比如到了还要执行 detect.sh)，这里依然在本地跑脚本
                # (或者你可以改到底座 Server 里跑，看你的架构设计)
                arrival_script = location_data.get("arrival_script")
                if arrival_script:
                    await ros_client.run_shell_script(arrival_script)
                    
                return self.success_response(f" Chassis Real-Response: {message}")
            else:
                return self.fail_response(f"Chassis Communication Failed: {message}")
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
            
            # 发送 TCP 请求 (发送 ori_x, ori_y...)
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
        # [修改点] 纯语义指令，不包含任何路径
        # 现在：只发送 "operation": "grasp"
        
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
            # 只有成功了，才告诉 Planner "我做完了"
            return self.success_response(f" Arm Successfully Executed: {action}. Hardware msg: {message}")
        
        else:
            # 如果 success 是 False (比如底座断连、脚本报错)
            # 返回 fail_response，Planner 会收到这个错误，并尝试自我修正或停止
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
            # [修改点] 纯语义指令，不包含任何脚本路径
            # 现在：只发送 "target_floor": "4F"
            
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
            
            # 严格判断 success
            if success:
                # 只有底座确认了，我们才敢告诉 Planner 地图已经切好了
                return self.success_response(f" Elevator Success: Arrived at {floor}. Map switched. Msg: {message}")
            else:
                # 否则报错
                return self.fail_response(f" Elevator Failed to switch floor. Hardware Error: {message}")
        
        # 如果命令里没解析出楼层
        return self.fail_response(f" Invalid Elevator Command: [{command}]. Please specify 1F, 2F, 4F, or 5F.")