"""検索計画を決定的に実行し、候補だけをベクトルで rerank する BYOG 検索。"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, replace

from graph_core import KnowledgeGraph
from retrieval_core import (
    ProjectReranker,
    SearchHit,
    SearchPlan,
    filter_project_ids,
)

GRAPH_FIRST_MODES = frozenset({"exact", "filter", "graph_expand", "graph_bridge"})


@dataclass(frozen=True)
class TraversalResult:
    distances: dict[str, int]
    paths: dict[str, tuple[str, ...]]


@dataclass(frozen=True)
class CandidateResult:
    project_ids: frozenset[str]
    graph_scores: dict[str, float]
    matched_conditions: dict[str, tuple[str, ...]]
    reasons: dict[str, tuple[str, ...]]
    paths: dict[str, tuple[str, ...]]


class ByogSearcher:
    def __init__(
        self, graph: KnowledgeGraph, reranker: ProjectReranker | None = None
    ) -> None:
        self.graph = graph
        self.reranker = reranker

    def execute(
        self,
        query: str,
        plan: SearchPlan,
        k: int | None = None,
        min_score: float = 0.0,
        global_candidate_ids: set[str] | None = None,
    ) -> list[SearchHit]:
        plan.validate(set(self.graph.nodes))
        candidates = self.candidates(
            plan, global_candidate_ids
        )
        return self.rerank_candidates(query, plan, candidates, k, min_score)

    def rerank_candidates(
        self,
        query: str,
        plan: SearchPlan,
        candidates: CandidateResult,
        k: int | None = None,
        min_score: float = 0.0,
    ) -> list[SearchHit]:
        """Graph順位を保ち、同じGの候補内をVでrerankする。"""

        if not candidates.project_ids:
            return []

        if self.reranker is None:
            return self.graph_only_hits(candidates, k)

        hits = self.reranker.rank(
            query=plan.augmented_query(query),
            candidate_ids=candidates.project_ids,
            k=None,
            min_score=min_score,
            graph_scores=candidates.graph_scores,
            matched_conditions=candidates.matched_conditions,
            reasons=candidates.reasons,
            paths=candidates.paths,
            vector_weight=1.0,
            graph_weight=0.0,
        )
        if plan.mode in GRAPH_FIRST_MODES:
            hits = [replace(hit, score=hit.graph_score) for hit in hits]
            hits.sort(
                key=lambda hit: (
                    -hit.graph_score,
                    -hit.vector_score,
                    hit.project_id,
                )
            )
        return hits[:k] if k is not None else hits

    def candidates(
        self,
        plan: SearchPlan,
        global_candidate_ids: set[str] | None = None,
    ) -> CandidateResult:
        """rerank・min-score・kを適用する前の候補とGraph根拠を返す。"""

        plan.validate(set(self.graph.nodes))
        projects_by_node = {
            node_id: set(project_ids)
            for node_id, project_ids in self.graph.tags_by_node.items()
        }
        condition_ids, matched = filter_project_ids(
            self.graph.projects, plan, projects_by_node
        )

        if plan.mode == "global_summary":
            graph_ids = (
                set(global_candidate_ids)
                if global_candidate_ids is not None
                else set(self.graph.projects)
            )
            traversal = TraversalResult({}, {})
        elif plan.mode == "natural":
            graph_ids = set(self.graph.projects)
            traversal = TraversalResult({}, {})
        elif plan.mode in {"exact", "filter"}:
            graph_ids = condition_ids
            traversal = TraversalResult({}, {})
        elif plan.mode == "graph_expand":
            traversal = self._traverse(
                plan.start_node_ids, plan.relations, plan.max_hops
            )
            graph_ids = self._projects_for_nodes(set(traversal.distances))
        elif plan.mode == "graph_bridge":
            traversal = self._bridge(plan)
            useful_nodes = set(traversal.distances) - set(plan.start_node_ids)
            graph_ids = self._projects_for_nodes(useful_nodes)
        else:  # SearchPlan.validate が先に弾くが型の追加漏れを安全側に倒す
            raise ValueError(f"未対応のBYOG検索モードです: {plan.mode}")

        if plan.conditions and plan.mode not in {"exact", "filter"}:
            candidates = graph_ids & condition_ids
        else:
            candidates = graph_ids

        graph_scores: dict[str, float] = {}
        reasons: dict[str, tuple[str, ...]] = {}
        paths: dict[str, tuple[str, ...]] = {}
        for project_id in candidates:
            tagged_nodes = self.graph.tags_by_project.get(project_id, set())
            traversed_tags = [
                node_id for node_id in tagged_nodes if node_id in traversal.distances
            ]
            if traversed_tags:
                best_node = min(
                    traversed_tags,
                    key=lambda node_id: (traversal.distances[node_id], node_id),
                )
                distance = traversal.distances[best_node]
                graph_scores[project_id] = 1.0 / (distance + 1)
                paths[project_id] = traversal.paths[best_node]
                reasons[project_id] = (f"graph:{best_node}:distance={distance}",)
            elif matched.get(project_id):
                count = len(matched[project_id])
                denominator = max(len(plan.conditions), 1)
                graph_scores[project_id] = count / denominator
                reasons[project_id] = tuple(
                    f"condition:{label}" for label in matched[project_id]
                )
            else:
                graph_scores[project_id] = 0.0
                reasons[project_id] = (plan.mode,)

        return CandidateResult(
            project_ids=frozenset(candidates),
            graph_scores=graph_scores,
            matched_conditions=matched,
            reasons=reasons,
            paths=paths,
        )

    @staticmethod
    def graph_only_hits(
        candidates: CandidateResult, k: int | None = None
    ) -> list[SearchHit]:
        """Vを使わずGだけで候補を順位付けする比較用ベースライン。"""

        hits = [
            SearchHit(
                project_id=project_id,
                score=candidates.graph_scores.get(project_id, 0.0),
                graph_score=candidates.graph_scores.get(project_id, 0.0),
                matched_conditions=candidates.matched_conditions.get(project_id, ()),
                reasons=candidates.reasons.get(project_id, ()),
                path=candidates.paths.get(project_id, ()),
            )
            for project_id in candidates.project_ids
        ]
        hits.sort(key=lambda item: (-item.score, item.project_id))
        return hits[:k] if k is not None else hits

    def _traverse(
        self,
        start_node_ids: tuple[str, ...],
        relations: tuple[str, ...],
        max_hops: int,
    ) -> TraversalResult:
        allowed = set(relations)
        distances = {node_id: 0 for node_id in start_node_ids}
        paths = {node_id: (node_id,) for node_id in start_node_ids}
        queue = deque(start_node_ids)

        while queue:
            current = queue.popleft()
            distance = distances[current]
            if distance >= max_hops:
                continue
            edges = list(self.graph.outgoing.get(current, []))
            edges.extend(self.graph.incoming.get(current, []))
            for edge in edges:
                if edge.relation not in allowed:
                    continue
                neighbor = edge.target if edge.source == current else edge.source
                if neighbor in distances:
                    continue
                distances[neighbor] = distance + 1
                paths[neighbor] = paths[current] + (neighbor,)
                queue.append(neighbor)
        return TraversalResult(distances, paths)

    def _bridge(self, plan: SearchPlan) -> TraversalResult:
        from_start = self._traverse(
            plan.start_node_ids, plan.relations, plan.max_hops
        )
        from_target = self._traverse(
            plan.target_node_ids, plan.relations, plan.max_hops
        )
        distances: dict[str, int] = {}
        paths: dict[str, tuple[str, ...]] = {}
        common_nodes = set(from_start.distances) & set(from_target.distances)
        for node_id in common_nodes:
            total_distance = (
                from_start.distances[node_id] + from_target.distances[node_id]
            )
            if total_distance > plan.max_hops:
                continue
            start_path = from_start.paths[node_id]
            target_path = from_target.paths[node_id]
            full_path = start_path + tuple(reversed(target_path[:-1]))
            distances[node_id] = from_start.distances[node_id]
            paths[node_id] = full_path
        return TraversalResult(distances, paths)

    def _projects_for_nodes(self, node_ids: set[str]) -> set[str]:
        project_ids: set[str] = set()
        for node_id in node_ids:
            project_ids.update(self.graph.tags_by_node.get(node_id, set()))
        return project_ids
