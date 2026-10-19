from __future__ import annotations

import itertools
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import networkx as nx

from planning_schema import DOOR_CONSTRAINTS


def _candidate_source_jsons(base_dir: Path) -> List[Path]:
    ignored_prefixes = (
        "canonical_",
        "dpo_",
        "phase",
        "app_benchmark",
        "tmp_",
        "analysis_",
        "smoke_",
    )
    ignored_suffixes = (
        "_report.json",
        "_dropped.json",
        "_results.json",
        "_summary.json",
        "_tasks.json",
    )
    ignored_names = {"filter_report.json"}

    candidates = []
    for path in base_dir.glob("*.json"):
        if path.name in ignored_names:
            continue
        if path.name.startswith(ignored_prefixes):
            continue
        if any(path.name.endswith(suffix) for suffix in ignored_suffixes):
            continue
        candidates.append(path)
    return sorted(candidates)


def resolve_graph_data_files(base_dir: Path | None = None) -> Dict[str, Any]:
    base_dir = (base_dir or Path(__file__).resolve().parent).resolve()
    candidates = _candidate_source_jsons(base_dir)

    w_file = next((path for path in candidates if path.name.startswith("W1-W4")), None)
    e_file = next((path for path in candidates if path.name.startswith("E1-E4")), None)

    if not w_file or not e_file:
        raise FileNotFoundError("Could not resolve W1-W4 or E1-E4 node json files.")

    remaining = [path for path in candidates if path not in {w_file, e_file}]
    if len(remaining) < 2:
        raise FileNotFoundError("Could not resolve macro-node json and edge json files.")

    remaining = sorted(remaining, key=lambda path: path.stat().st_size)
    macro_file = remaining[0]
    edge_file = remaining[-1]

    return {
        "macro_node_file": str(macro_file),
        "micro_node_files": [str(w_file), str(e_file)],
        "edge_file": str(edge_file),
    }


def build_default_graph_engine(base_dir: Path | None = None) -> "EmbodiedGraphEngine":
    return EmbodiedGraphEngine(**resolve_graph_data_files(base_dir))


class EmbodiedGraphEngine:
    def __init__(self, macro_node_file, micro_node_files, edge_file):
        normalized_files = self._normalize_source_files(
            macro_node_file,
            micro_node_files,
            edge_file,
        )
        macro_node_file = normalized_files["macro_node_file"]
        micro_node_files = normalized_files["micro_node_files"]
        edge_file = normalized_files["edge_file"]

        self.G = nx.Graph()
        self.nodes_data: Dict[str, Dict[str, Any]] = {}
        self.synthetic_nodes: set[str] = set()

        self._load_nodes(macro_node_file)
        for micro_file in micro_node_files:
            self._load_nodes(micro_file)

        raw_edges = self._read_edges(edge_file)
        self._synthesize_missing_lobby_nodes(raw_edges)
        self._load_edges(raw_edges)
        self._add_entry_edges()
        self._auto_stitch_rooms()
        self._add_aerial_links()

        print(
            "[OK] Graph constructed successfully! "
            f"Loaded {self.G.number_of_nodes()} nodes, {self.G.number_of_edges()} edges."
        )

    def _normalize_source_files(self, macro_node_file, micro_node_files, edge_file) -> Dict[str, Any]:
        macro_path = Path(str(macro_node_file))
        micro_paths = [Path(str(path)) for path in micro_node_files]
        edge_path = Path(str(edge_file))
        all_paths = [macro_path, *micro_paths, edge_path]

        if all(path.exists() for path in all_paths):
            return {
                "macro_node_file": str(macro_path),
                "micro_node_files": [str(path) for path in micro_paths],
                "edge_file": str(edge_path),
            }

        candidate_dirs: List[Path] = []
        for path in all_paths:
            parent = path.parent if str(path.parent) not in {"", "."} else None
            if parent:
                resolved_parent = parent.resolve()
                if resolved_parent not in candidate_dirs:
                    candidate_dirs.append(resolved_parent)

        for fallback_dir in (Path.cwd(), Path(__file__).resolve().parent):
            resolved_dir = fallback_dir.resolve()
            if resolved_dir not in candidate_dirs:
                candidate_dirs.append(resolved_dir)

        for base_dir in candidate_dirs:
            try:
                return resolve_graph_data_files(base_dir)
            except FileNotFoundError:
                continue

        return {
            "macro_node_file": str(macro_path),
            "micro_node_files": [str(path) for path in micro_paths],
            "edge_file": str(edge_path),
        }

    def validate_graph_logic(self):
        print("\n" + "=" * 52)
        print("Starting graph logic validation...")

        error_edges = []
        warning_edges = []

        for u, v, data in self.G.edges(data=True):
            edge_type = data.get("edge_type", "FlatGround")

            u_floor_match = re.search(r"_(\d)F", u)
            v_floor_match = re.search(r"_(\d)F", v)

            if u_floor_match and v_floor_match:
                u_floor = u_floor_match.group(1)
                v_floor = v_floor_match.group(1)

                if u_floor != v_floor and edge_type == "FlatGround":
                    error_edges.append(
                        f"FlatGround cross-floor edge: {u} <-> {v} ({edge_type})"
                    )

                if (
                    abs(int(u_floor) - int(v_floor)) > 1
                    and edge_type == "Stairs"
                ):
                    warning_edges.append(
                        f"Large stairs floor jump: {u} <-> {v} ({edge_type})"
                    )

            u_bldg_match = re.match(r"^([WE]\d)", u)
            v_bldg_match = re.match(r"^([WE]\d)", v)

            if (
                u_bldg_match
                and v_bldg_match
                and u_bldg_match.group(1) != v_bldg_match.group(1)
                and edge_type not in {"AerialLink"}
            ):
                error_edges.append(
                    f"Unexpected direct inter-building edge: {u} <-> {v} ({edge_type})"
                )

        if not error_edges and not warning_edges:
            print("Graph logic validation passed.")
        else:
            if error_edges:
                print("\nErrors:")
                for err in error_edges:
                    print(err)
            if warning_edges:
                print("\nWarnings:")
                for warn in warning_edges:
                    print(warn)
        print("=" * 52)

    def _read_json_safely(self, filepath):
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                content = f.read()
            content = re.sub(r"^\s*//.*$", "", content, flags=re.MULTILINE)
            content = re.sub(r"//.*$", "", content, flags=re.MULTILINE)
            content = content.strip()

            if content.startswith('"nodes"') or content.startswith('"edges"'):
                content = "{\n" + content + "\n}"

            content = re.sub(r",\s*\]", "]", content)
            content = re.sub(r",\s*\}", "}", content)

            try:
                return json.loads(content)
            except json.JSONDecodeError:
                try:
                    return json.loads("[\n" + content + "\n]")
                except json.JSONDecodeError:
                    return None
        except Exception:
            return None

    def _read_edges(self, filepath) -> List[Dict[str, Any]]:
        data = self._read_json_safely(filepath)
        if not data:
            return []
        return data if isinstance(data, list) else data.get("edges", [])

    def _load_nodes(self, filepath):
        data = self._read_json_safely(filepath)
        if not data:
            print(f"[WARNING] Failed to load nodes from {filepath}")
            return
        nodes_list = data.get("nodes", []) if isinstance(data, dict) else data
        print(f"[DEBUG] Loading {len(nodes_list)} nodes from {filepath}")
        for node in nodes_list:
            node_id = node["node_id"]
            payload = dict(node)
            payload.setdefault("synthetic", False)
            payload.setdefault("provenance", "original")
            self.nodes_data[node_id] = payload
            self.G.add_node(node_id, **payload)

    def _ensure_node(self, node_id: str, node_data: Dict[str, Any]):
        if node_id in self.nodes_data:
            return
        payload = dict(node_data)
        payload["node_id"] = node_id
        payload.setdefault("synthetic", True)
        payload.setdefault("provenance", "synthetic")
        self.synthetic_nodes.add(node_id)
        self.nodes_data[node_id] = payload
        self.G.add_node(node_id, **payload)

    def _synthetic_lobby_node(self, node_id: str) -> Dict[str, Any]:
        building_prefix = node_id.split("_", 1)[0]
        category = "Academic" if re.match(r"^[WE]\d$", building_prefix) else "Public"
        floor_match = re.search(r"_(\dF)_Lobby$", node_id)
        floor_label = floor_match.group(1) if floor_match else "UnknownFloor"
        return {
            "type": "Indoor",
            "category": category,
            "spawns": [],
            "functions": ["navigate_floor", "handover_item"],
            "location_desc": f"Synthetic lobby node for {building_prefix} {floor_label}.",
            "synthetic": True,
            "provenance": "synthetic_lobby",
        }

    def _entry_node_payload(self, entry_node: str, building_node: str) -> Dict[str, Any]:
        category = self.nodes_data.get(building_node, {}).get("category", "Public")
        return {
            "type": "Indoor",
            "category": category,
            "spawns": [],
            "functions": ["enter_building", "handover_item", "navigate_floor"],
            "location_desc": f"Synthetic entry lobby for {building_node}.",
            "synthetic": True,
            "provenance": "synthetic_entry",
        }

    def _synthesize_missing_lobby_nodes(self, raw_edges: Iterable[Dict[str, Any]]):
        missing = set()
        for edge in raw_edges:
            for node_id in (edge["source"], edge["target"]):
                if node_id not in self.nodes_data and node_id.endswith("_Lobby"):
                    missing.add(node_id)

        for node_id in sorted(missing):
            self._ensure_node(node_id, self._synthetic_lobby_node(node_id))

    def _normalize_edge_record(self, edge: Dict[str, Any]) -> Dict[str, Any]:
        edge_type = edge.get("edge_type", "FlatGround")
        traversable_by = list(edge.get("traversable_by", []))
        if edge_type == "Elevator":
            traversable_by = ["chassis_car"]
        elif edge_type == "Stairs":
            traversable_by = ["dog_robot"]
        elif not traversable_by:
            traversable_by = ["chassis_car", "dog_robot"]

        constraints = list(edge.get("constraints", []))
        cooperation_logic = edge.get("cooperation_logic", "no_cooperation_needed")
        difficulty = edge.get("difficulty")
        if not difficulty:
            difficulty = "medium" if edge_type in {"Elevator", "Stairs"} else "low"

        return {
            "weight": edge.get("length_m", 10),
            "traversable_by": traversable_by,
            "edge_type": edge_type,
            "constraints": constraints,
            "cooperation_logic": cooperation_logic,
            "difficulty": difficulty,
            "synthetic": False,
            "provenance": "original",
        }

    def _load_edges(self, raw_edges: Iterable[Dict[str, Any]]):
        for edge in raw_edges:
            normalized = self._normalize_edge_record(edge)
            self.G.add_edge(edge["source"], edge["target"], **normalized)

    def _add_entry_edges(self):
        for node_id, data in list(self.nodes_data.items()):
            for entry in data.get("entry_nodes", []):
                self._ensure_node(entry, self._entry_node_payload(entry, node_id))
                if not self.G.has_edge(node_id, entry):
                    self.G.add_edge(
                        node_id,
                        entry,
                        weight=1,
                        edge_type="Virtual_Entry",
                        traversable_by=["chassis_car", "dog_robot"],
                        constraints=[],
                        cooperation_logic="building interface to 1F lobby",
                        difficulty="low",
                        synthetic=True,
                        provenance="synthetic_entry_edge",
                    )

    def _room_autostitch_weight(self, node_id: str, node_type: str) -> int:
        if "Lab" in node_id:
            return 12
        if "MeetingRoom" in node_id:
            return 10
        if "Office" in node_id:
            return 8
        if "Toilet" in node_id:
            return 6
        if node_type == "Room":
            return 8
        return 8

    def _room_autostitch_constraints(self, node_id: str) -> List[str]:
        if "Lab" in node_id:
            return ["requires_arm_to_open_door", "requires_arm_to_swipe_access_card"]
        if any(token in node_id for token in ("Office", "MeetingRoom", "Toilet")):
            return ["requires_arm_to_open_door"]
        return list(DOOR_CONSTRAINTS)

    def _auto_stitch_rooms(self):
        isolates = list(nx.isolates(self.G))
        for iso_node in isolates:
            if iso_node not in self.nodes_data:
                continue
            parts = iso_node.split("_")
            if len(parts) < 3 or not parts[1].endswith("F"):
                continue

            building = parts[0]
            floor = parts[1]
            lobby_node = f"{building}_{floor}_Lobby"
            self._ensure_node(lobby_node, self._synthetic_lobby_node(lobby_node))

            if self.G.has_edge(lobby_node, iso_node):
                continue

            node_type = self.nodes_data[iso_node].get("type", "Unknown")
            self.G.add_edge(
                lobby_node,
                iso_node,
                weight=self._room_autostitch_weight(iso_node, node_type),
                edge_type="FlatGround",
                traversable_by=["chassis_car", "dog_robot"],
                constraints=self._room_autostitch_constraints(iso_node),
                cooperation_logic="synthetic room access edge",
                difficulty="low",
                synthetic=True,
                provenance="synthetic_room_edge",
            )

    def _eligible_aerial_nodes(self) -> List[str]:
        eligible = []
        for node_id, data in self.nodes_data.items():
            if data.get("type") in {"Building", "Outdoor"}:
                eligible.append(node_id)
        return sorted(set(eligible))

    def _ground_distance(self, source: str, target: str) -> float:
        try:
            return nx.shortest_path_length(self.G, source, target, weight="weight")
        except nx.NetworkXNoPath:
            return 60.0

    def _add_aerial_links(self):
        eligible_nodes = self._eligible_aerial_nodes()
        for source, target in itertools.combinations(eligible_nodes, 2):
            if self.G.has_edge(source, target):
                continue
            ground_distance = self._ground_distance(source, target)
            aerial_weight = max(10.0, round(ground_distance * 0.55, 2))
            self.G.add_edge(
                source,
                target,
                weight=aerial_weight,
                edge_type="AerialLink",
                traversable_by=["delivery_drone"],
                constraints=["outdoor_only", "requires_drone_landing_zone"],
                cooperation_logic="delivery_drone flies between outdoor-compatible interfaces",
                difficulty="medium",
                synthetic=True,
                provenance="synthetic_aerial",
            )

    def get_edge_data(self, source: str, target: str) -> Dict[str, Any] | None:
        if not self.G.has_edge(source, target):
            return None
        return dict(self.G.edges[source, target])

    def get_local_scenario(self, start_node, end_node, k=3):
        if start_node not in self.G or end_node not in self.G:
            return None, f"Invalid start/end node: {start_node}, {end_node}"
        try:
            path_generator = nx.shortest_simple_paths(
                self.G, start_node, end_node, weight="weight"
            )
            k_shortest_paths = []
            for i, path in enumerate(path_generator):
                if i >= k:
                    break
                k_shortest_paths.append(path)
        except nx.NetworkXNoPath:
            return None, "No path found in graph"

        if not k_shortest_paths:
            return None, "No path found in graph"

        subgraph_nodes = set()
        for path in k_shortest_paths:
            subgraph_nodes.update(path)
            for node in path:
                subgraph_nodes.update(self.G.neighbors(node))

        local_G = self.G.subgraph(subgraph_nodes)
        scenario_data = {
            "task_start": start_node,
            "task_end": end_node,
            "local_nodes": [],
            "local_edges": [],
        }

        for node_id in local_G.nodes():
            node_data = self.nodes_data.get(node_id, {})
            scenario_data["local_nodes"].append(
                {
                    "id": node_id,
                    "type": node_data.get("type", "Unknown"),
                    "category": node_data.get("category", "Unknown"),
                    "spawns": node_data.get("spawns", []),
                    "functions": node_data.get("functions", []),
                    "synthetic": node_data.get("synthetic", False),
                    "provenance": node_data.get("provenance", "unknown"),
                }
            )

        for u, v, edge_data in local_G.edges(data=True):
            scenario_data["local_edges"].append(
                {
                    "source": u,
                    "target": v,
                    "edge_type": edge_data.get("edge_type"),
                    "weight": edge_data.get("weight"),
                    "traversable_by": edge_data.get("traversable_by"),
                    "constraints": edge_data.get("constraints"),
                    "difficulty": edge_data.get("difficulty"),
                    "synthetic": edge_data.get("synthetic", False),
                    "provenance": edge_data.get("provenance", "unknown"),
                }
            )

        return scenario_data, k_shortest_paths


if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parent
    engine = EmbodiedGraphEngine(
        macro_node_file=str(base_dir / "宏观建筑物节点信息.json"),
        micro_node_files=[
            str(base_dir / "W1-W4节点信息.json"),
            str(base_dir / "E1-E4节点信息.json"),
        ],
        edge_file=str(base_dir / "所有建筑物边信息.json"),
    )
    engine.validate_graph_logic()
    print(f"Nodes: {engine.G.number_of_nodes()}")
    print(f"Edges: {engine.G.number_of_edges()}")
