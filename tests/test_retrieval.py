from __future__ import annotations

import math
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from build_graph_byog import load_byog_graph
from byog_search import ByogSearcher, CandidateResult
from graph_core import Question, load_questions
from retrieval_core import (
    CachedEmbedder,
    ProjectReranker,
    SearchHit,
    SearchPlan,
    load_search_plans,
    metrics_at_k,
)
from search_rag import (
    candidate_evaluation,
    evaluation_record,
    summarize_by_category,
)


class CountingEmbedder:
    name = "counting"
    model = "v1"

    def __init__(self) -> None:
        self.calls = 0

    def embed(self, texts: list[str]) -> list[list[float]]:
        self.calls += 1
        return [[float(len(text)), 1.0] for text in texts]


class RetrievalCoreTest(unittest.TestCase):
    def test_metrics_at_k_uses_rank_order_for_ndcg(self) -> None:
        metric = metrics_at_k(("A", "B"), ["B", "C", "A"], 2)

        self.assertEqual(metric.recall, 0.5)
        self.assertEqual(metric.precision, 0.5)
        expected = 1.0 / (1.0 + 1.0 / math.log2(3))
        self.assertAlmostEqual(metric.ndcg, expected)

    def test_embedding_cache_avoids_duplicate_calls(self) -> None:
        embedder = CountingEmbedder()
        with tempfile.TemporaryDirectory() as directory:
            cached = CachedEmbedder(embedder, Path(directory) / "cache.json")
            first = cached.embed(["同じ文", "別の文"])
            second = cached.embed(["同じ文"])

        self.assertEqual(embedder.calls, 1)
        self.assertEqual(second[0], first[0])

    def test_evaluation_record_contains_candidates_and_explicit_ranks(self) -> None:
        question = Question("QX", "質問", ("P1",), "周辺", "QX")
        hits = [
            SearchHit("P1", 0.8, vector_score=0.8),
            SearchHit("P2", 0.5, vector_score=0.5),
        ]
        metric = metrics_at_k(question.gold_ids, ["P1", "P2"], 2)
        record = evaluation_record(
            question,
            None,
            hits,
            {
                "2": {
                    "k": metric.k,
                    "recall": metric.recall,
                    "precision": metric.precision,
                    "f1": metric.f1,
                    "ndcg": metric.ndcg,
                }
            },
            candidates=candidate_evaluation(question.gold_ids, {"P1", "P2"}),
        )

        self.assertEqual(record["candidates"]["candidate_count"], 2)
        self.assertEqual([item["rank"] for item in record["hits"]], [1, 2])
        categories = summarize_by_category([record], (2,))
        self.assertEqual(categories["周辺"]["candidate"]["recall"], 1.0)


class ByogPlanTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.graph = load_byog_graph(ROOT / "data")
        cls.plans = load_search_plans(ROOT / "data" / "search_plans.draft.json")
        cls.searcher = ByogSearcher(cls.graph)

    def test_cross_condition_is_and(self) -> None:
        hits = self.searcher.execute(
            "Kubernetesを使うモデル配信案件", self.plans["Q06"]
        )

        self.assertEqual(
            {hit.project_id for hit in hits}, {"P004", "P015", "P028"}
        )

    def test_candidates_are_available_before_rerank(self) -> None:
        candidates = self.searcher.candidates(self.plans["Q06"])
        self.assertEqual(
            candidates.project_ids, frozenset({"P004", "P015", "P028"})
        )
        graph_hits = self.searcher.graph_only_hits(candidates)
        self.assertEqual(
            [hit.project_id for hit in graph_hits],
            ["P004", "P015", "P028"],
        )

    def test_graph_score_precedes_vector_score_and_vector_breaks_ties(self) -> None:
        reranker = Mock(spec=ProjectReranker)
        reranker.rank.return_value = [
            SearchHit("A", 0.9, vector_score=0.9, graph_score=0.5),
            SearchHit("C", 0.8, vector_score=0.8, graph_score=1.0),
            SearchHit("B", 0.2, vector_score=0.2, graph_score=1.0),
        ]
        searcher = ByogSearcher(self.graph, reranker)
        candidates = CandidateResult(
            project_ids=frozenset({"A", "B", "C"}),
            graph_scores={"A": 0.5, "B": 1.0, "C": 1.0},
            matched_conditions={},
            reasons={},
            paths={},
        )

        hits = searcher.rerank_candidates(
            "質問",
            SearchPlan(mode="graph_expand", start_node_ids=("N_JOB_MLOPS",)),
            candidates,
        )

        self.assertEqual([hit.project_id for hit in hits], ["C", "B", "A"])
        self.assertEqual([hit.score for hit in hits], [1.0, 1.0, 0.5])

    def test_negative_query_returns_no_candidate(self) -> None:
        hits = self.searcher.execute(
            "量子コンピュータの案件", self.plans["Q17"]
        )

        self.assertEqual(hits, [])

    def test_all_draft_plans_reference_existing_nodes(self) -> None:
        known_nodes = set(self.graph.nodes)
        for plan in self.plans.values():
            plan.validate(known_nodes)

    def test_questions_have_valid_splits_gold_ids_and_plans(self) -> None:
        questions = load_questions(ROOT / "data")

        self.assertEqual(
            sum(question.evaluation_split == "development" for question in questions),
            20,
        )
        self.assertEqual(
            sum(question.evaluation_split == "holdout" for question in questions),
            17,
        )
        for question in questions:
            with self.subTest(question_id=question.id):
                self.assertTrue(set(question.gold_ids) <= set(self.graph.projects))
                self.assertIn(question.plan_id, self.plans)
                if question.category == "該当なし":
                    self.assertEqual(question.gold_ids, ())

    def test_deterministic_plans_match_gold_candidates(self) -> None:
        excluded = {"Q10", "Q13"}
        for question in load_questions(ROOT / "data"):
            if question.evaluation_split != "development" or question.id in excluded:
                continue
            with self.subTest(question_id=question.id):
                self.assertTrue(question.plan_id)
                hits = self.searcher.execute(
                    question.text, self.plans[question.plan_id]
                )
                self.assertEqual(
                    {hit.project_id for hit in hits}, set(question.gold_ids)
                )


if __name__ == "__main__":
    unittest.main()
