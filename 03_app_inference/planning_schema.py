from __future__ import annotations

from typing import Any, Dict


AGENT_PROFILES: Dict[str, Dict[str, Any]] = {
    "chassis_car": {
        "label": "Chassis Robot with arm",
        "capabilities": {
            "moving",
            "grasping",
            "door_opening",
            "access_control",
            "elevator_use",
        },
        "max_load_kg": 10.0,
        "allowed_node_types": {"Building", "Outdoor", "Indoor", "Room", "Elevator"},
        "notes": "Ground robot for flat indoor/outdoor travel, doors, and elevators.",
    },
    "dog_robot": {
        "label": "Quadruped robot",
        "capabilities": {"moving", "stair_climbing", "narrow_space"},
        "max_load_kg": 2.0,
        "allowed_node_types": {"Building", "Outdoor", "Indoor", "Room", "Stairs"},
        "notes": "Ground robot for stairs and narrow indoor areas. Cannot use elevators or doors.",
    },
    "delivery_drone": {
        "label": "Outdoor delivery drone",
        "capabilities": {"moving", "flying", "outdoor_only"},
        "max_load_kg": 5.0,
        "allowed_node_types": {"Building", "Outdoor"},
        "notes": "Outdoor-only aerial robot that moves between building interface nodes.",
    },
}


ACTION_SPECS: Dict[str, Dict[str, Any]] = {
    "navigate": {
        "kind": "movement",
        "required_args": ("source", "target"),
        "edge_types": {"FlatGround", "Virtual_Entry", "AerialLink"},
        "description": "Traverse a single adjacent flat-ground, building-entry, or aerial edge.",
    },
    "call_elevator": {
        "kind": "movement",
        "required_args": ("source", "target"),
        "edge_types": {"Elevator"},
        "description": "Traverse a single adjacent elevator edge.",
    },
    "climb_stairs": {
        "kind": "movement",
        "required_args": ("source", "target"),
        "edge_types": {"Stairs"},
        "description": "Traverse a single adjacent stair edge.",
    },
    "pick_up": {
        "kind": "manipulation",
        "required_args": ("item", "node"),
        "description": "Pick up a transportable item at the current node.",
    },
    "put_down": {
        "kind": "manipulation",
        "required_args": ("item", "node"),
        "description": "Put down a carried item at the current node.",
    },
    "handover": {
        "kind": "coordination",
        "required_args": ("item", "node", "target"),
        "description": "Transfer an item from one robot to another at the same node.",
    },
}


DOOR_CONSTRAINTS = {
    "requires_arm_to_open_door",
    "requires_arm_to_swipe_access_card",
}

ELEVATOR_CONSTRAINTS = {
    "requires_mechanical_arm_to_press_floor_button",
    "chassis_must_stay_stationary_during_operation",
}

STAIR_CONSTRAINTS = {
    "requires_stair_climbing_capability",
    "chassis_car_blocked",
}

AERIAL_EDGE_TYPES = {"AerialLink"}
GROUND_EDGE_TYPES = {"FlatGround", "Virtual_Entry"}


OBJECT_CATALOG: Dict[str, Dict[str, Any]] = {
    "access_button": {"weight_kg": 3.0, "transportable": False, "fragile": False},
    "access_card": {"weight_kg": 0.05, "transportable": True, "fragile": False},
    "access_control_panel": {"weight_kg": 6.0, "transportable": False, "fragile": True},
    "access_panel": {"weight_kg": 4.0, "transportable": False, "fragile": True},
    "basketball": {"weight_kg": 0.7, "transportable": True, "fragile": False},
    "bed": {"weight_kg": 25.0, "transportable": False, "fragile": False},
    "bleachers": {"weight_kg": 100.0, "transportable": False, "fragile": False},
    "book": {"weight_kg": 0.6, "transportable": True, "fragile": False},
    "classroom_door": {"weight_kg": 40.0, "transportable": False, "fragile": False},
    "coffee_cup": {"weight_kg": 0.45, "transportable": True, "fragile": True},
    "computer": {"weight_kg": 4.5, "transportable": True, "fragile": True},
    "conference_table": {"weight_kg": 30.0, "transportable": False, "fragile": False},
    "desk": {"weight_kg": 18.0, "transportable": False, "fragile": False},
    "desk_lamp": {"weight_kg": 1.2, "transportable": True, "fragile": True},
    "direction_sign": {"weight_kg": 12.0, "transportable": False, "fragile": False},
    "document": {"weight_kg": 0.2, "transportable": True, "fragile": False},
    "dorm_door": {"weight_kg": 35.0, "transportable": False, "fragile": False},
    "drink": {"weight_kg": 0.6, "transportable": True, "fragile": True},
    "elevator_call_panel": {"weight_kg": 5.0, "transportable": False, "fragile": True},
    "event_poster": {"weight_kg": 0.2, "transportable": True, "fragile": True},
    "file_cabinet": {"weight_kg": 35.0, "transportable": False, "fragile": False},
    "floor_button_panel": {"weight_kg": 5.0, "transportable": False, "fragile": True},
    "football": {"weight_kg": 0.5, "transportable": True, "fragile": False},
    "gym_door": {"weight_kg": 45.0, "transportable": False, "fragile": False},
    "hand_sanitizer": {"weight_kg": 0.4, "transportable": True, "fragile": True},
    "info_panel": {"weight_kg": 4.0, "transportable": False, "fragile": True},
    "intercom_panel": {"weight_kg": 4.0, "transportable": False, "fragile": True},
    "key_card": {"weight_kg": 0.02, "transportable": True, "fragile": False},
    "kungfu_order_panel": {"weight_kg": 6.0, "transportable": False, "fragile": True},
    "lab_coat": {"weight_kg": 0.6, "transportable": True, "fragile": False},
    "lab_door": {"weight_kg": 45.0, "transportable": False, "fragile": False},
    "lab_equipment": {"weight_kg": 3.0, "transportable": True, "fragile": True},
    "library_card": {"weight_kg": 0.02, "transportable": True, "fragile": False},
    "map_leaflet": {"weight_kg": 0.05, "transportable": True, "fragile": False},
    "meal_box": {"weight_kg": 0.8, "transportable": True, "fragile": True},
    "microphone": {"weight_kg": 0.4, "transportable": True, "fragile": True},
    "office_door": {"weight_kg": 40.0, "transportable": False, "fragile": False},
    "payment_terminal": {"weight_kg": 5.0, "transportable": False, "fragile": True},
    "projector": {"weight_kg": 3.0, "transportable": True, "fragile": True},
    "reception_desk": {"weight_kg": 30.0, "transportable": False, "fragile": False},
    "rest_area": {"weight_kg": 20.0, "transportable": False, "fragile": False},
    "running_track": {"weight_kg": 999.0, "transportable": False, "fragile": False},
    "safety_gloves": {"weight_kg": 0.2, "transportable": True, "fragile": False},
    "scoreboard": {"weight_kg": 20.0, "transportable": False, "fragile": True},
    "seat": {"weight_kg": 8.0, "transportable": False, "fragile": False},
    "security_booth": {"weight_kg": 999.0, "transportable": False, "fragile": False},
    "self_checkout_panel": {"weight_kg": 6.0, "transportable": False, "fragile": True},
    "speaker": {"weight_kg": 4.0, "transportable": True, "fragile": True},
    "stage": {"weight_kg": 999.0, "transportable": False, "fragile": False},
    "starbucks_order_panel": {"weight_kg": 6.0, "transportable": False, "fragile": True},
    "test_tube": {"weight_kg": 0.1, "transportable": True, "fragile": True},
    "timer_panel": {"weight_kg": 4.0, "transportable": False, "fragile": True},
    "trash_can": {"weight_kg": 1.5, "transportable": True, "fragile": False},
    "whiteboard": {"weight_kg": 12.0, "transportable": False, "fragile": False},
}


def get_item_profile(item_name: str) -> Dict[str, Any]:
    return OBJECT_CATALOG.get(
        item_name,
        {"weight_kg": 1.0, "transportable": False, "fragile": False},
    )


def is_transportable_item(item_name: str) -> bool:
    return bool(get_item_profile(item_name).get("transportable", False))

