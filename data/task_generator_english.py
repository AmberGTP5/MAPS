"""
English task generator for graph-grounded preference data.
"""

from __future__ import annotations

import json
import os
import random
from typing import Dict, List, Optional

from graph_engine import EmbodiedGraphEngine
from planning_schema import ACTION_SPECS, AGENT_PROFILES, get_item_profile, is_transportable_item
from preference_utils import build_planning_prompt, build_task_id, derive_task_metadata, extract_building_id


class EnglishTaskGenerator:
    def __init__(self, engine: EmbodiedGraphEngine):
        self.engine = engine
        self.task_profiles = self._build_task_profiles()

    def _profile_weight(self, profile_name: str, default: float) -> float:
        env_name = f"PVR_TASK_PROFILE_{profile_name.upper()}_WEIGHT".replace("-", "_")
        raw = os.getenv(env_name)
        if raw is None:
            return default
        try:
            value = float(raw)
        except ValueError:
            return default
        return max(0.0, value)

    def _build_task_profiles(self) -> List[Dict]:
        self.task_profiles = [
            {
                "name": "room_general",
                "weight": self._profile_weight("room_general", 0.20),
                "start_types": {"Room", "Indoor"},
                "end_types": {"Room", "Indoor"},
                "max_item_weight": 10.0,
                "preferred_robot": "chassis_car",
                "force_preferred_if_feasible": False,
            },
            {
                "name": "interface_light",
                "weight": self._profile_weight("interface_light", 0.14),
                "start_types": {"Room", "Indoor"},
                "end_types": {"Room", "Indoor"},
                "max_item_weight": 5.0,
                "preferred_robot": "chassis_car",
                "force_preferred_if_feasible": False,
            },
            {
                "name": "interface_ultralight",
                "weight": self._profile_weight("interface_ultralight", 0.10),
                "start_types": {"Room", "Indoor"},
                "end_types": {"Room", "Indoor"},
                "same_building_as_start": True,
                "same_floor_as_start": True,
                "max_item_weight": 2.0,
                "preferred_robot": "dog_robot",
                "force_preferred_if_feasible": False,
            },
            {
                "name": "dog_indoor_last_meter",
                "weight": self._profile_weight("dog_indoor_last_meter", 0.07),
                "start_types": {"Room", "Indoor"},
                "start_name_contains": "_1F_",
                "end_types": {"Indoor"},
                "end_name_contains": "_1F_Lobby",
                "same_building_as_start": True,
                "same_floor_as_start": True,
                "max_item_weight": 2.0,
                "preferred_robot": "dog_robot",
                "force_preferred_if_feasible": True,
            },
            {
                "name": "dog_upper_floor_same_building",
                "weight": self._profile_weight("dog_upper_floor_same_building", 0.05),
                "start_types": {"Room", "Indoor"},
                "end_types": {"Indoor"},
                "end_name_contains": "_Lobby",
                "end_floor_min": 2,
                "same_building_as_start": True,
                "same_floor_as_start": True,
                "max_item_weight": 2.0,
                "preferred_robot": "dog_robot",
                "force_preferred_if_feasible": True,
            },
            {
                "name": "mixed_light",
                "weight": self._profile_weight("mixed_light", 0.10),
                "start_types": {"Room", "Indoor"},
                "end_types": {"Room", "Indoor"},
                "max_item_weight": 2.0,
                "preferred_robot": "chassis_car",
                "force_preferred_if_feasible": False,
            },
            {
                "name": "same_building_cross_floor",
                "weight": self._profile_weight("same_building_cross_floor", 0.18),
                "start_types": {"Room", "Indoor"},
                "end_types": {"Room", "Indoor"},
                "same_building_as_start": True,
                "different_floor_from_start": True,
                "max_item_weight": 10.0,
                "preferred_robot": "chassis_car",
                "force_preferred_if_feasible": False,
            },
            {
                "name": "cross_building_same_floor",
                "weight": self._profile_weight("cross_building_same_floor", 0.16),
                "start_types": {"Room", "Indoor"},
                "end_types": {"Room", "Indoor"},
                "different_building_from_start": True,
                "same_floor_as_start": True,
                "max_item_weight": 10.0,
                "preferred_robot": "chassis_car",
                "force_preferred_if_feasible": False,
            },
            {
                "name": "cross_building_cross_floor",
                "weight": self._profile_weight("cross_building_cross_floor", 0.16),
                "start_types": {"Room", "Indoor"},
                "end_types": {"Room", "Indoor"},
                "different_building_from_start": True,
                "different_floor_from_start": True,
                "max_item_weight": 10.0,
                "preferred_robot": "chassis_car",
                "force_preferred_if_feasible": False,
            },
        ]
        return self.task_profiles

    def generate_task(self, excluded_task_signatures: Optional[set[str]] = None) -> Optional[Dict]:
        excluded_task_signatures = excluded_task_signatures or set()
        for _ in range(40):
            profile = self._sample_profile()
            sampled = self._sample_nodes_for_profile(profile)
            if not sampled:
                continue

            start_node, target_item, item_profile, end_node = sampled
            result = self.engine.get_local_scenario(start_node, end_node, k=3)
            if not result[0]:
                continue

            scenario_json, paths = result
            task_metadata = derive_task_metadata(start_node, end_node, target_item)
            task_metadata["task_profile"] = profile["name"]
            if task_metadata["task_signature"] in excluded_task_signatures:
                continue
            instruction = (
                f"Please transport a {target_item} from {start_node} to {end_node} "
                f"while respecting all robot and environment constraints."
            )
            system_prompt = self._build_llm_prompt(
                instruction,
                scenario_json,
                target_item,
                item_profile,
                start_node,
                end_node,
                paths,
            )
            task_id = build_task_id(system_prompt, start_node, end_node, target_item)

            return {
                "task_id": task_id,
                "instruction": instruction,
                "start": start_node,
                "end": end_node,
                "item": target_item,
                "item_profile": item_profile,
                "task_metadata": task_metadata,
                "local_graph": scenario_json,
                "reference_paths": paths,
                "shortest_path_length": len(paths[0]),
                "found_k_paths": len(paths),
                "prompt": system_prompt,
                "prompt_version": "graph_prompt_v3",
                "sampling_profile": profile["name"],
                "preferred_robot_hint": profile.get("preferred_robot"),
                "force_preferred_if_feasible": profile.get("force_preferred_if_feasible", False),
            }
        return None

    def _sample_profile(self) -> Dict:
        weights = [profile["weight"] for profile in self.task_profiles]
        return random.choices(self.task_profiles, weights=weights, k=1)[0]

    def _sample_nodes_for_profile(self, profile: Dict) -> Optional[tuple[str, str, Dict, str]]:
        valid_start_nodes = []
        potential_end_nodes = []
        start_name_contains = profile.get("start_name_contains")
        end_name_contains = profile.get("end_name_contains")

        for node_id, node_data in self.engine.nodes_data.items():
            node_type = node_data.get("type")
            if node_type not in {"Room", "Indoor", "Building", "Outdoor"}:
                continue

            if node_type in profile["end_types"] and self._matches_end_filters(node_id, profile):
                potential_end_nodes.append(node_id)

            if node_type not in profile["start_types"]:
                continue
            if start_name_contains and start_name_contains not in node_id:
                continue

            transportable_items = []
            for item in node_data.get("spawns", []):
                if not is_transportable_item(item):
                    continue
                item_profile = get_item_profile(item)
                if item_profile.get("weight_kg", 0.0) <= profile["max_item_weight"]:
                    transportable_items.append((item, item_profile))
            if transportable_items:
                valid_start_nodes.append((node_id, transportable_items))

        if len(valid_start_nodes) < 1 or len(potential_end_nodes) < 1:
            return None

        candidate_starts = list(valid_start_nodes)
        random.shuffle(candidate_starts)
        for start_node, transportable_items in candidate_starts:
            end_candidates = [
                node_id
                for node_id in potential_end_nodes
                if node_id != start_node and self._matches_start_end_relation(start_node, node_id, profile)
            ]
            if not end_candidates:
                continue

            end_node = random.choice(end_candidates)
            target_item, item_profile = random.choice(transportable_items)
            return start_node, target_item, item_profile, end_node
        return None

    def _matches_end_filters(self, node_id: str, profile: Dict) -> bool:
        end_name_contains = profile.get("end_name_contains")
        if end_name_contains and end_name_contains not in node_id:
            return False

        end_floor_min = profile.get("end_floor_min")
        if end_floor_min is not None:
            floor = self._extract_floor_number(node_id)
            if floor is None or floor < int(end_floor_min):
                return False

        end_floor_max = profile.get("end_floor_max")
        if end_floor_max is not None:
            floor = self._extract_floor_number(node_id)
            if floor is None or floor > int(end_floor_max):
                return False

        return True

    def _matches_start_end_relation(self, start_node: str, end_node: str, profile: Dict) -> bool:
        start_building = extract_building_id(start_node)
        end_building = extract_building_id(end_node)
        start_floor = self._extract_floor_number(start_node)
        end_floor = self._extract_floor_number(end_node)

        if profile.get("same_building_as_start") and start_building != end_building:
            return False
        if profile.get("different_building_from_start") and start_building == end_building:
            return False
        if profile.get("same_floor_as_start"):
            if start_floor is None or end_floor is None or start_floor != end_floor:
                return False
        if profile.get("different_floor_from_start"):
            if start_floor is None or end_floor is None or start_floor == end_floor:
                return False
        return True

    def _extract_floor_number(self, node_id: str) -> Optional[int]:
        for part in node_id.split("_"):
            if part.endswith("F") and part[:-1].isdigit():
                return int(part[:-1])
        return None

    def _build_llm_prompt(
        self,
        instruction: str,
        scenario_json: Dict,
        target_item: str,
        item_profile: Dict,
        start_node: str,
        end_node: str,
        reference_paths: List[List[str]],
    ) -> str:
        return build_planning_prompt(
            instruction=instruction,
            scenario_json=scenario_json,
            target_item=target_item,
            item_profile=item_profile,
            start_node=start_node,
            end_node=end_node,
            reference_paths=reference_paths,
        )


if __name__ == "__main__":
    import os

    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)

    engine = EmbodiedGraphEngine(
        macro_node_file="宏观建筑物节点信息.json",
        micro_node_files=["W1-W4节点信息.json", "E1-E4节点信息.json"],
        edge_file="所有建筑物边信息.json",
    )
    generator = EnglishTaskGenerator(engine)

    for _ in range(3):
        sample = generator.generate_task()
        print("=" * 80)
        print(json.dumps(sample, ensure_ascii=False, indent=2)[:3000])
