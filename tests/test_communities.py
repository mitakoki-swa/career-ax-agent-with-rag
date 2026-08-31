from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from build_graph_byog import TRAVERSE_RELATIONS, load_byog_graph
from communities import detect_communities


class CommunitiesTest(unittest.TestCase):
    def test_byog_communities_exclude_skill_nodes(self) -> None:
        graph = load_byog_graph(ROOT / "data")

        communities = detect_communities(
            graph,
            relations=TRAVERSE_RELATIONS,
            allowed_node_kinds=frozenset({"職種", "業務"}),
        )

        self.assertTrue(communities)
        self.assertTrue(
            all(
                graph.nodes[node_id].kind in {"職種", "業務"}
                for community in communities
                for node_id in community.node_ids
            )
        )
        self.assertFalse(
            any(
                graph.nodes[node_id].kind == "スキル"
                for community in communities
                for node_id in community.node_ids
            )
        )
        self.assertTrue(all(community.project_ids for community in communities))


if __name__ == "__main__":
    unittest.main()
