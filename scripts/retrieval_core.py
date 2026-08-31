"""検索計画、共通検索結果、rerank、@k 評価の共通処理。"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol


SEARCH_MODES = frozenset(
    {"natural", "exact", "filter", "graph_expand", "graph_bridge", "global_summary"}
)
OPERATORS = frozenset({"AND", "OR"})
CONDITION_FIELDS = frozenset(
    {"project_id", "role", "domain", "cloud", "skills", "text", "concept"}
)
ALLOWED_RELATIONS = frozenset({"包含", "関連", "使用"})


@dataclass(frozen=True)
class SearchCondition:
    field: str
    value: str
    node_id: str = ""
    node_ids: tuple[str, ...] = ()

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "SearchCondition":
        condition = cls(
            field=str(payload.get("field") or "").strip(),
            value=str(payload.get("value") or "").strip(),
            node_id=str(payload.get("node_id") or "").strip(),
            node_ids=_string_tuple(payload.get("node_ids")),
        )
        if condition.field not in CONDITION_FIELDS:
            raise ValueError(f"未対応の検索条件です: {condition.field}")
        if not condition.value and not condition.node_id and not condition.node_ids:
            raise ValueError("検索条件には value、node_id、node_ids のいずれかが必要です")
        return condition

    def to_dict(self) -> dict[str, Any]:
        return {
            "field": self.field,
            "value": self.value,
            "node_id": self.node_id,
            "node_ids": list(self.node_ids),
        }


@dataclass(frozen=True)
class SearchPlan:
    mode: str = "natural"
    operator: str = "AND"
    conditions: tuple[SearchCondition, ...] = ()
    start_node_ids: tuple[str, ...] = ()
    target_node_ids: tuple[str, ...] = ()
    relations: tuple[str, ...] = ("包含", "関連")
    max_hops: int = 3
    query_terms: tuple[str, ...] = ()
    confidence: float = 1.0
    status: str = "draft"
    note: str = ""

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "SearchPlan":
        plan = cls(
            mode=str(payload.get("mode") or "natural").strip(),
            operator=str(payload.get("operator") or "AND").strip().upper(),
            conditions=tuple(
                SearchCondition.from_dict(item)
                for item in (payload.get("conditions") or [])
            ),
            start_node_ids=_string_tuple(payload.get("start_node_ids")),
            target_node_ids=_string_tuple(payload.get("target_node_ids")),
            relations=_string_tuple(payload.get("relations")) or ("包含", "関連"),
            max_hops=int(payload.get("max_hops") or 3),
            query_terms=_string_tuple(payload.get("query_terms")),
            confidence=float(payload.get("confidence", 1.0)),
            status=str(payload.get("status") or "draft").strip(),
            note=str(payload.get("note") or "").strip(),
        )
        plan.validate()
        return plan

    def validate(self, known_node_ids: set[str] | None = None) -> None:
        if self.mode not in SEARCH_MODES:
            raise ValueError(f"未対応の検索モードです: {self.mode}")
        if self.operator not in OPERATORS:
            raise ValueError(f"未対応の結合方法です: {self.operator}")
        if not 0 <= self.confidence <= 1:
            raise ValueError("confidence は 0〜1 で指定してください")
        if not 0 <= self.max_hops <= 5:
            raise ValueError("max_hops は 0〜5 で指定してください")
        invalid_relations = set(self.relations) - ALLOWED_RELATIONS
        if invalid_relations:
            raise ValueError(f"未対応の関係です: {sorted(invalid_relations)}")
        if self.mode in {"exact", "filter"} and not self.conditions:
            raise ValueError(f"{self.mode} には conditions が必要です")
        if self.mode == "graph_expand" and not self.start_node_ids:
            raise ValueError("graph_expand には start_node_ids が必要です")
        if self.mode == "graph_bridge" and (
            not self.start_node_ids or not self.target_node_ids
        ):
            raise ValueError(
                "graph_bridge には start_node_ids と target_node_ids が必要です"
            )
        if known_node_ids is not None:
            referenced = set(self.start_node_ids) | set(self.target_node_ids)
            referenced.update(
                condition.node_id for condition in self.conditions if condition.node_id
            )
            referenced.update(
                node_id
                for condition in self.conditions
                for node_id in condition.node_ids
            )
            unknown = referenced - known_node_ids
            if unknown:
                raise ValueError(f"検索計画に未知のノードがあります: {sorted(unknown)}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "operator": self.operator,
            "conditions": [item.to_dict() for item in self.conditions],
            "start_node_ids": list(self.start_node_ids),
            "target_node_ids": list(self.target_node_ids),
            "relations": list(self.relations),
            "max_hops": self.max_hops,
            "query_terms": list(self.query_terms),
            "confidence": self.confidence,
            "status": self.status,
            "note": self.note,
        }

    def augmented_query(self, original: str) -> str:
        terms = [original]
        terms.extend(item.value for item in self.conditions if item.value)
        terms.extend(self.query_terms)
        return " ".join(dict.fromkeys(part.strip() for part in terms if part.strip()))


@dataclass(frozen=True)
class SearchHit:
    project_id: str
    score: float
    vector_score: float = 0.0
    graph_score: float = 0.0
    matched_conditions: tuple[str, ...] = ()
    reasons: tuple[str, ...] = ()
    path: tuple[str, ...] = ()


@dataclass(frozen=True)
class MetricAtK:
    k: int
    recall: float
    precision: float
    f1: float
    ndcg: float


class Embedder(Protocol):
    name: str

    def embed(self, texts: list[str]) -> list[list[float]]:
        ...


def _string_tuple(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return tuple(part.strip() for part in value.split(",") if part.strip())
    return tuple(str(part).strip() for part in value if str(part).strip())


def load_search_plans(path: Path) -> dict[str, SearchPlan]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} は質問IDをキーにしたJSON objectにしてください")
    return {
        str(question_id): SearchPlan.from_dict(plan)
        for question_id, plan in payload.items()
    }


def load_node_labels(path: Path) -> dict[str, str]:
    """検索計画の参照ノードIDと正規名を評価用に読み込む。"""

    import csv

    with path.open(encoding="utf-8-sig", newline="") as fh:
        return {
            (row.get("ノードID") or "").strip(): (row.get("正規名") or "").strip()
            for row in csv.DictReader(fh)
            if (row.get("ノードID") or "").strip()
            and (row.get("正規名") or "").strip()
        }


def _normalize_node_name(value: str) -> str:
    return re.sub(r"\s+", "", value).lower()


class SemanticNodeResolver:
    """別Graphのノード名を語彙一致と埋め込み類似度で1対多解決する。"""

    def __init__(
        self,
        graph: Any,
        embedder: Embedder,
        top_k: int = 5,
        min_score: float = 0.55,
    ) -> None:
        if top_k < 1:
            raise ValueError("node top-k は1以上にしてください")
        if not -1.0 <= min_score <= 1.0:
            raise ValueError("node min-score は-1〜1で指定してください")
        self.graph = graph
        self.embedder = embedder
        self.top_k = top_k
        self.min_score = min_score
        self.node_ids = sorted(graph.nodes)
        self.node_texts = [
            " ".join(
                part
                for part in (
                    graph.nodes[node_id].name,
                    graph.nodes[node_id].kind,
                    graph.nodes[node_id].description,
                )
                if part
            )
            for node_id in self.node_ids
        ]
        self.node_vectors = embedder.embed(self.node_texts)
        self._details: dict[str, list[dict[str, Any]]] = {}

    def resolve(
        self, label: str, expected_kind: str = ""
    ) -> tuple[str, ...]:
        normalized_label = _normalize_node_name(label)
        lexical = {
            node_id
            for node_id in self.node_ids
            if normalized_label
            and (
                normalized_label
                in _normalize_node_name(self.graph.nodes[node_id].name)
                or _normalize_node_name(self.graph.nodes[node_id].name)
                in normalized_label
            )
        }
        query_vector = self.embedder.embed([label])[0]
        scored = sorted(
            (
                (cosine_similarity(query_vector, node_vector), node_id)
                for node_id, node_vector in zip(
                    self.node_ids, self.node_vectors
                )
            ),
            key=lambda item: (-item[0], item[1]),
        )
        semantic = [
            node_id
            for score, node_id in scored
            if score >= self.min_score
            and (
                not expected_kind
                or self.graph.nodes[node_id].kind == expected_kind
            )
        ][: self.top_k]
        matches = tuple(
            sorted(
                lexical | set(semantic),
                key=lambda node_id: (
                    -next(
                        (
                            score
                            for score, scored_id in scored
                            if scored_id == node_id
                        ),
                        0.0,
                    ),
                    node_id,
                ),
            )
            [: self.top_k]
        )
        score_by_id = {node_id: score for score, node_id in scored}
        self._details[label] = [
            {
                "node_id": node_id,
                "name": self.graph.nodes[node_id].name,
                "kind": self.graph.nodes[node_id].kind,
                "score": score_by_id.get(node_id, 0.0),
                "lexical_match": node_id in lexical,
            }
            for node_id in matches
        ]
        return matches

    def details(self, label: str) -> list[dict[str, Any]]:
        return self._details.get(label, [])


def adapt_search_plan_to_graph(
    plan: SearchPlan,
    authoring_node_labels: dict[str, str],
    graph: Any,
    resolver: SemanticNodeResolver | None = None,
    min_graph_hops: int | None = None,
) -> tuple[SearchPlan, dict[str, Any]]:
    """固定オントロジーの計画を、1対多で別GraphのノードIDへ写像する。"""

    name_index: dict[str, list[str]] = {}
    for node_id, node in graph.nodes.items():
        name_index.setdefault(_normalize_node_name(node.name), []).append(node_id)

    resolved: dict[str, list[str]] = {}

    def resolve(node_id: str) -> tuple[str, ...]:
        label = authoring_node_labels.get(node_id, "")
        if not label:
            return ()
        if resolver is None:
            normalized_label = _normalize_node_name(label)
            matches = sorted(
                candidate_id
                for normalized_name, candidate_ids in name_index.items()
                if normalized_label
                and (
                    normalized_label in normalized_name
                    or normalized_name in normalized_label
                )
                for candidate_id in candidate_ids
            )
        else:
            expected_kind = (
                "職種"
                if node_id.startswith("N_JOB_")
                else "業務"
                if node_id.startswith("N_WORK_")
                else "スキル"
                if node_id.startswith("N_SKILL_")
                else ""
            )
            matches = list(resolver.resolve(label, expected_kind))
        if not matches:
            return ()
        resolved[node_id] = matches
        return tuple(matches)

    unresolved_condition_ids: list[str] = []
    conditions: list[SearchCondition] = []
    for condition in plan.conditions:
        mapped_ids = resolve(condition.node_id) if condition.node_id else ()
        if condition.node_id and not mapped_ids:
            unresolved_condition_ids.append(condition.node_id)
        conditions.append(
            SearchCondition(
                field=condition.field,
                value=condition.value,
                node_ids=mapped_ids,
            )
        )

    start_node_ids = tuple(
        mapped
        for node_id in plan.start_node_ids
        for mapped in resolve(node_id)
    )
    target_node_ids = tuple(
        mapped
        for node_id in plan.target_node_ids
        for mapped in resolve(node_id)
    )
    unresolved_start_ids = [
        node_id for node_id in plan.start_node_ids if node_id not in resolved
    ]
    unresolved_target_ids = [
        node_id for node_id in plan.target_node_ids if node_id not in resolved
    ]

    missing_required_anchor = (
        plan.mode == "graph_expand" and not start_node_ids
    ) or (
        plan.mode == "graph_bridge"
        and (not start_node_ids or not target_node_ids)
    )
    if missing_required_anchor:
        adapted = SearchPlan(
            mode="filter",
            operator="AND",
            conditions=(
                SearchCondition(
                    field="project_id",
                    value="__UNRESOLVED_GRAPH_ANCHOR__",
                ),
            ),
            query_terms=plan.query_terms,
            confidence=plan.confidence,
            status=plan.status,
            note=f"{plan.note} / unresolved graph anchor",
        )
        status = "unresolved"
    else:
        adapted = SearchPlan(
            mode=plan.mode,
            operator=plan.operator,
            conditions=tuple(conditions),
            start_node_ids=start_node_ids,
            target_node_ids=target_node_ids,
            relations=plan.relations,
            max_hops=(
                max(plan.max_hops, min_graph_hops)
                if min_graph_hops is not None
                and plan.mode in {"graph_expand", "graph_bridge"}
                else plan.max_hops
            ),
            query_terms=plan.query_terms,
            confidence=plan.confidence,
            status=plan.status,
            note=plan.note,
        )
        status = (
            "partial"
            if (
                unresolved_condition_ids
                or unresolved_start_ids
                or unresolved_target_ids
            )
            else "resolved"
        )
    adapted.validate(set(graph.nodes))
    return adapted, {
        "status": status,
        "original_mode": plan.mode,
        "adapted_mode": adapted.mode,
        "original_max_hops": plan.max_hops,
        "adapted_max_hops": adapted.max_hops,
        "resolved_node_ids": resolved,
        "resolution_details": {
            node_id: resolver.details(authoring_node_labels[node_id])
            for node_id in resolved
            if resolver is not None
        },
        "unresolved_condition_node_ids": unresolved_condition_ids,
        "unresolved_start_node_ids": unresolved_start_ids,
        "unresolved_target_node_ids": unresolved_target_ids,
    }


def save_search_plans(path: Path, plans: dict[str, SearchPlan]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        question_id: plan.to_dict() for question_id, plan in sorted(plans.items())
    }
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def _normalize(text: str) -> str:
    return "".join(text.lower().split())


def condition_matches_project(project: Any, condition: SearchCondition) -> bool:
    needle = _normalize(condition.value)
    if not needle:
        return True
    if condition.field == "project_id":
        source = getattr(project, "id", "")
        return _normalize(source) == needle
    if condition.field == "role":
        source = getattr(project, "role", "")
    elif condition.field == "domain":
        source = getattr(project, "domain", "")
    elif condition.field == "cloud":
        source = getattr(project, "cloud", "")
    elif condition.field == "skills":
        source = getattr(project, "skills", "")
    else:
        source = project.document()
    return needle in _normalize(source)


def filter_project_ids(
    projects: dict[str, Any],
    plan: SearchPlan,
    projects_by_node: dict[str, set[str]] | None = None,
) -> tuple[set[str], dict[str, tuple[str, ...]]]:
    """条件を決定的に適用する。node_id はタグ、value は案件列を評価する。"""

    projects_by_node = projects_by_node or {}
    if not plan.conditions:
        return set(projects), {}

    condition_sets: list[set[str]] = []
    labels: list[str] = []
    for condition in plan.conditions:
        condition_node_ids = (
            condition.node_ids
            or ((condition.node_id,) if condition.node_id else ())
        )
        if condition_node_ids:
            matched = set().union(
                *(
                    projects_by_node.get(node_id, set())
                    for node_id in condition_node_ids
                )
            )
        else:
            matched = {
                project_id
                for project_id, project in projects.items()
                if condition_matches_project(project, condition)
            }
        condition_sets.append(matched)
        labels.append(
            "|".join(condition_node_ids)
            or f"{condition.field}={condition.value}"
        )

    if plan.operator == "AND":
        found = set.intersection(*condition_sets) if condition_sets else set(projects)
    else:
        found = set.union(*condition_sets) if condition_sets else set()

    matched_labels: dict[str, tuple[str, ...]] = {}
    for project_id in found:
        matched_labels[project_id] = tuple(
            label
            for label, matched in zip(labels, condition_sets)
            if project_id in matched
        )
    return found, matched_labels


class CachedEmbedder:
    """テキスト単位の埋め込みキャッシュ。モデル・入力が同じならAPIを再実行しない。"""

    def __init__(self, embedder: Embedder, path: Path) -> None:
        self.embedder = embedder
        self.path = path
        model = str(getattr(embedder, "model", ""))
        self.namespace = f"{embedder.name}:{model}"
        self.name = f"{embedder.name}-cached"
        self._cache = self._load()

    def _load(self) -> dict[str, list[float]]:
        if not self.path.exists():
            return {}
        payload = json.loads(self.path.read_text(encoding="utf-8"))
        return {
            str(key): [float(value) for value in vector]
            for key, vector in payload.items()
        }

    def _key(self, text: str) -> str:
        raw = f"{self.namespace}\0{text}".encode("utf-8")
        return hashlib.sha256(raw).hexdigest()

    def embed(self, texts: list[str]) -> list[list[float]]:
        keys = [self._key(text) for text in texts]
        missing_texts: list[str] = []
        missing_keys: list[str] = []
        for key, text in zip(keys, texts):
            if key not in self._cache:
                missing_keys.append(key)
                missing_texts.append(text)
        if missing_texts:
            vectors = self.embedder.embed(missing_texts)
            for key, vector in zip(missing_keys, vectors):
                self._cache[key] = vector
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(
                json.dumps(self._cache, ensure_ascii=False), encoding="utf-8"
            )
        return [self._cache[key] for key in keys]


class ProjectReranker:
    def __init__(self, projects: dict[str, Any], embedder: Embedder) -> None:
        self.projects = projects
        self.embedder = embedder
        self.project_ids = list(projects)
        self.project_vectors = dict(
            zip(
                self.project_ids,
                embedder.embed(
                    [projects[project_id].document() for project_id in self.project_ids]
                ),
            )
        )

    def rank(
        self,
        query: str,
        candidate_ids: set[str] | list[str] | tuple[str, ...] | None = None,
        k: int | None = None,
        min_score: float = 0.0,
        graph_scores: dict[str, float] | None = None,
        matched_conditions: dict[str, tuple[str, ...]] | None = None,
        reasons: dict[str, tuple[str, ...]] | None = None,
        paths: dict[str, tuple[str, ...]] | None = None,
        vector_weight: float = 1.0,
        graph_weight: float = 0.0,
    ) -> list[SearchHit]:
        query_vector = self.embedder.embed([query])[0]
        candidates = (
            set(candidate_ids) if candidate_ids is not None else set(self.project_ids)
        )
        graph_scores = graph_scores or {}
        matched_conditions = matched_conditions or {}
        reasons = reasons or {}
        paths = paths or {}
        hits: list[SearchHit] = []
        for project_id in candidates:
            project_vector = self.project_vectors.get(project_id)
            if project_vector is None:
                continue
            vector_score = cosine_similarity(query_vector, project_vector)
            if vector_score < min_score:
                continue
            graph_score = graph_scores.get(project_id, 0.0)
            score = vector_weight * vector_score + graph_weight * graph_score
            hits.append(
                SearchHit(
                    project_id=project_id,
                    score=score,
                    vector_score=vector_score,
                    graph_score=graph_score,
                    matched_conditions=matched_conditions.get(project_id, ()),
                    reasons=reasons.get(project_id, ()),
                    path=paths.get(project_id, ()),
                )
            )
        hits.sort(key=lambda item: (-item.score, item.project_id))
        return hits[:k] if k is not None else hits


def cosine_similarity(left: list[float], right: list[float]) -> float:
    if not left or not right or len(left) != len(right):
        return 0.0
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return sum(a * b for a, b in zip(left, right)) / (left_norm * right_norm)


def metrics_at_k(
    gold_ids: tuple[str, ...], ranked_ids: list[str], k: int
) -> MetricAtK:
    predicted = ranked_ids[:k]
    gold = set(gold_ids)
    if not gold:
        recall_value = 1.0 if not predicted else 0.0
        precision_value = 1.0 if not predicted else 0.0
    else:
        hit = len(gold & set(predicted))
        recall_value = hit / len(gold)
        precision_value = hit / len(predicted) if predicted else 0.0
    f1_value = (
        0.0
        if recall_value + precision_value == 0
        else 2
        * recall_value
        * precision_value
        / (recall_value + precision_value)
    )
    return MetricAtK(
        k=k,
        recall=recall_value,
        precision=precision_value,
        f1=f1_value,
        ndcg=ndcg_at_k(gold_ids, ranked_ids, k),
    )


def ndcg_at_k(gold_ids: tuple[str, ...], ranked_ids: list[str], k: int) -> float:
    gold = set(gold_ids)
    if not gold:
        return 1.0 if not ranked_ids[:k] else 0.0
    dcg = sum(
        1.0 / math.log2(rank + 1)
        for rank, project_id in enumerate(ranked_ids[:k], start=1)
        if project_id in gold
    )
    ideal_hits = min(len(gold), k)
    ideal = sum(
        1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1)
    )
    return dcg / ideal if ideal else 0.0
