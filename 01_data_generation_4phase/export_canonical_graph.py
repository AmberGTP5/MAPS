from __future__ import annotations

import json
from pathlib import Path

from graph_engine import EmbodiedGraphEngine


def main():
    base_dir = Path(__file__).resolve().parent
    engine = EmbodiedGraphEngine(
        macro_node_file=str(base_dir / "宏观建筑物节点信息.json"),
        micro_node_files=[
            str(base_dir / "W1-W4节点信息.json"),
            str(base_dir / "E1-E4节点信息.json"),
        ],
        edge_file=str(base_dir / "所有建筑物边信息.json"),
    )

    nodes = []
    for node_id, node_data in sorted(engine.nodes_data.items()):
        payload = dict(node_data)
        payload["node_id"] = node_id
        nodes.append(payload)

    edges = []
    for source, target, edge_data in sorted(engine.G.edges(data=True)):
        payload = dict(edge_data)
        payload["source"] = source
        payload["target"] = target
        edges.append(payload)

    (base_dir / "canonical_nodes.json").write_text(
        json.dumps({"nodes": nodes}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (base_dir / "canonical_edges.json").write_text(
        json.dumps({"edges": edges}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print("Exported canonical_nodes.json and canonical_edges.json")


if __name__ == "__main__":
    main()
