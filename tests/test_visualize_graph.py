from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from visualize_graph import (
    build_visualization_payload,
    render_html,
    write_visualization,
)


class VisualizeGraphTest(unittest.TestCase):
    def test_byog_payload_contains_graph_and_project_tags(self) -> None:
        payload = build_visualization_payload(ROOT / "data", "byog")

        self.assertEqual(payload["source"], "byog")
        self.assertEqual(payload["stats"]["conceptNodes"], 60)
        self.assertEqual(payload["stats"]["projectNodes"], 28)
        self.assertGreater(payload["stats"]["graphEdges"], 0)
        self.assertGreater(payload["stats"]["tagEdges"], 0)
        self.assertTrue(
            any(edge["relation"] == "案件タグ" for edge in payload["edges"])
        )

    def test_html_is_self_contained(self) -> None:
        payload = build_visualization_payload(ROOT / "data", "byog")
        html = render_html(payload)

        self.assertIn("BYOG（人手オントロジー）", html)
        self.assertIn("const DATA =", html)
        self.assertNotIn("https://", html)
        self.assertNotIn("<script src=", html)

    def test_writes_separate_source_file(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = write_visualization(
                ROOT / "data", Path(directory), "byog"
            )

            self.assertEqual(path.name, "byog.html")
            self.assertTrue(path.exists())

    def test_llm_payload_when_cache_exists(self) -> None:
        llm_dir = ROOT / "data" / "llm_generated"
        if not (llm_dir / "nodes.csv").exists():
            self.skipTest("LLM Graph cache is not generated")

        payload = build_visualization_payload(ROOT / "data", "llm")
        self.assertEqual(payload["source"], "llm")
        self.assertEqual(payload["title"], "LLM生成Graph")
        self.assertGreater(payload["stats"]["conceptNodes"], 0)


if __name__ == "__main__":
    unittest.main()
