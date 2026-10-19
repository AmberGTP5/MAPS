"""
机器人位置与行为注册表
数据来源: Demo3.xml & 现场测量
作用：将 LLM 的语义位置映射为 ROS 2 的物理坐标和底层原子行为树
"""

# ==============================================================================
# 位置数据库 (Location Database)
# ==============================================================================
LOCATION_DATABASE = {
    # ---------------- [1F Areas] ----------------
    "Starbucks_Outdoor": {
        "coords": {"x": 96.73, "y": -31.81, "yaw": 1.10},   
        "behavior_tree": "GoToStarbucks",                   
        
    },
    "Elevator_Lobby_1F": {
        "coords": {"x": -9.88, "y": 3.81, "yaw": 1.49},     
        "behavior_tree": "GoToLift_1F",                     
        
    },

    # ---------------- [2F Areas] ----------------
    "Vending_Machine_2F": {
        # [重规划新增点位]
        "coords": {"x": 10.5, "y": 20.0, "yaw": 0.0},       
        "behavior_tree": "NavigateToPose",                  
        
    },
    "Elevator_Lobby_2F": {
         # [重规划新增点位]
        "coords": {"x": -9.88, "y": 3.81, "yaw": 1.49},     # 假设坐标
        "behavior_tree": "NavigateToPose",
        
    },

    # ---------------- [4F Areas] ----------------
    "Info_Hub_4F": {
        "coords": {"x": -6.13, "y": 58.09, "yaw": 1.73},    
        "behavior_tree": "GoToInfoHub",                     
         
    },
    "Elevator_Lobby_4F": {
        # XML line 146 (TakeLiftUp_4F 出电梯点)
        "coords": {"x": -2.93, "y": 0.16, "yaw": 3.04},     
        "behavior_tree": "NavigateToPose",  
        
    },
    

    # ---------------- [5F Areas] ----------------
    "Robot_Station_5F": {
        "coords": {"x": -1.9599, "y": 1.896, "yaw": -2.30}, 
        "behavior_tree": "ReturnToStart_5F",                
        
    },
    "Elevator_Lobby_5F": {
        "coords": {
            "ori_x": -0.0036090810311144896,
            "ori_y": 0.01049063675181623,
            "ori_z": 0.7511391720958466,
            "ori_w": 0.6600506535242754,
            "x": -2.63,
            "y": -4.86,
            "yaw": 2.40
        },  
        "behavior_tree": "GoToLift_5F",                     
        
    },

    # ---------------- [User Areas] ----------------
    "User_Desk_Internal": {
        "coords": {"x": 15.0, "y": 25.0, "yaw": 1.57},
        "behavior_tree": "DogNavigate",                     
        
    }
}