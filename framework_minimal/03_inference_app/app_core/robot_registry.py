LOCATION_DATABASE = {
    # ==========================================
    # 🟢 1F Areas (Pickup Zone)
    # ==========================================
    "Starbucks_Outdoor": {
        "coords": {"x": 96.73, "y": -31.81, "yaw": 1.10},   
        "behavior_tree": "GoToStarbucks",                    
        "floor": "1F"
    },
    "Elevator_Lobby_1F": {
        "coords": {"x": -9.88, "y": 3.81, "yaw": 1.49},      
        "behavior_tree": "GoToLift_1F",                      
        "floor": "1F"
    },

    # ==========================================
    # 🟡 2F Areas (Alternative Pickup)
    # ==========================================
    "Vending_Machine_2F": {
        "coords": {"x": 10.5, "y": 20.0, "yaw": 0.0},        
        "behavior_tree": "NavigateToPose",                  
        "floor": "2F"
    },
    "Elevator_Lobby_2F": {
        "coords": {"x": -9.88, "y": 3.81, "yaw": 1.49},     
        "behavior_tree": "NavigateToPose",
        "floor": "2F"
    },

    # ==========================================
    # 🔵 4F Areas (Dog Base & Handover & Delivery)
    # ==========================================
    "Dog_Station_4F": {
        # [新增] 狗的初始位置 (4楼)
        "coords": {"x": 0.0, "y": 0.0, "yaw": 0.0},          
        "behavior_tree": "DogNavigate",
        "floor": "4F"
    },
    "Info_Hub_4F": {
        # [交接点] 底盘把咖啡给狗的地方
        "coords": {"x": -6.13, "y": 58.09, "yaw": 1.73},     
        "behavior_tree": "GoToInfoHub",                      
        "floor": "4F"
    },
    "Elevator_Lobby_4F": {
        # 底盘坐电梯到达 4F 的位置
        "coords": {"x": -2.93, "y": 0.16, "yaw": 3.04},      
        "behavior_tree": "NavigateToPose",  
        "floor": "4F"
    },
    "User_Desk_Internal": {
        # [终点] 用户的工位 (必须在 4F，因为狗不能坐电梯)
        "coords": {"x": 15.0, "y": 25.0, "yaw": 1.57},
        "behavior_tree": "DogNavigate",                      
        "floor": "4F"
    },

    # ==========================================
    # 🟣 5F Areas (Chassis Base)
    # ==========================================
    "Robot_Station_5F": {
        # [底盘起点] 只有底盘在这里
        "coords": {"x": -1.9599, "y": 1.896, "yaw": -2.30}, 
        "behavior_tree": "ReturnToStart_5F",                 
        "floor": "5F"
    },
    "Elevator_Lobby_5F": {
        "coords": {"x": -19.28, "y": -49.68, "yaw": 2.51},  
        "behavior_tree": "GoToLift_5F",                      
        "floor": "5F"
    }
}