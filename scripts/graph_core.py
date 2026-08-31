"""BYOG / LLM Graph 共通のグラフ構造と探索。"""

from __future__ import annotations

import csv
import os
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"


def load_env(path: Path | None = None) -> None:
    """リポジトリ直下の .env を環境変数に載せる。既存の値は上書きしない。"""
    env_path = path or (ROOT / ".env")
    if not env_path.exists():
        return
    for raw in env_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value
    bedrock_key = os.environ.get("BEDROCK_API_KEY", "").strip()
    if bedrock_key and not os.environ.get("AWS_BEARER_TOKEN_BEDROCK"):
        os.environ["AWS_BEARER_TOKEN_BEDROCK"] = bedrock_key


def bedrock_api_key() -> str:
    return (
        os.environ.get("AWS_BEARER_TOKEN_BEDROCK", "").strip()
        or os.environ.get("BEDROCK_API_KEY", "").strip()
    )


def bedrock_region() -> str:
    return (
        os.environ.get("AWS_REGION", "").strip()
        or os.environ.get("BEDROCK_REGION", "").strip()
        or "ap-northeast-1"
    )


DEFAULT_BEDROCK_CHAT_MODEL = "jp.anthropic.claude-sonnet-4-5-20250929-v1:0"
_INFERENCE_PROFILE_PREFIXES = frozenset({"us", "eu", "apac", "jp", "global"})
_LEGACY_CHAT_MODELS = frozenset(
    {
        "anthropic.claude-3-5-sonnet-20241022-v2:0",
        "apac.anthropic.claude-3-5-sonnet-20241022-v2:0",
        "us.anthropic.claude-3-5-sonnet-20241022-v2:0",
        "eu.anthropic.claude-3-5-sonnet-20241022-v2:0",
        "jp.anthropic.claude-3-5-sonnet-20241022-v2:0",
    }
)
_REGION_INFERENCE_PREFIX = {
    "us-east-1": "us",
    "us-east-2": "us",
    "us-west-2": "us",
    "eu-central-1": "eu",
    "eu-west-1": "eu",
    "eu-west-2": "eu",
    "eu-west-3": "eu",
    "eu-north-1": "eu",
    "ap-northeast-1": "jp",
    "ap-northeast-2": "apac",
    "ap-northeast-3": "apac",
    "ap-southeast-1": "apac",
    "ap-southeast-2": "apac",
    "ap-south-1": "apac",
}


def bedrock_chat_model_id(model: str = "") -> str:
    """Claude は foundation model ID では呼べないことが多いので推論プロファイルにする。"""
    chosen = (model or os.environ.get("BEDROCK_CHAT_MODEL") or DEFAULT_BEDROCK_CHAT_MODEL).strip()
    if not chosen:
        chosen = DEFAULT_BEDROCK_CHAT_MODEL
    if chosen in _LEGACY_CHAT_MODELS:
        return DEFAULT_BEDROCK_CHAT_MODEL
    if chosen.startswith("arn:"):
        return chosen
    prefix, _, _rest = chosen.partition(".")
    if prefix in _INFERENCE_PROFILE_PREFIXES:
        return chosen
    if chosen.startswith("anthropic."):
        geo = _REGION_INFERENCE_PREFIX.get(bedrock_region(), "jp")
        return f"{geo}.{chosen}"
    return chosen


def bedrock_runtime_client():
    key = bedrock_api_key()
    if not key:
        raise RuntimeError(
            "BEDROCK_API_KEY がありません。.env に BEDROCK_API_KEY=... を書いてください。"
        )
    os.environ["AWS_BEARER_TOKEN_BEDROCK"] = key
    try:
        import boto3
    except ImportError as exc:
        raise RuntimeError("boto3 がありません。pip install boto3") from exc
    return boto3.client("bedrock-runtime", region_name=bedrock_region())


@dataclass(frozen=True)
class Node:
    id: str
    name: str
    kind: str
    description: str


@dataclass(frozen=True)
class Edge:
    id: str
    source: str
    target: str
    relation: str
    weight: float


@dataclass(frozen=True)
class Project:
    id: str
    name: str
    role: str
    domain: str
    cloud: str
    summary: str
    skills: str

    def document(self) -> str:
        return " ".join(
            part
            for part in (
                self.name,
                self.role,
                self.domain,
                self.cloud,
                self.summary,
                self.skills,
            )
            if part
        )


@dataclass(frozen=True)
class Question:
    id: str
    text: str
    gold_ids: tuple[str, ...]
    category: str
    plan_id: str = ""
    evaluation_split: str = "development"


@dataclass
class KnowledgeGraph:
    nodes: dict[str, Node]
    edges: list[Edge]
    projects: dict[str, Project]
    tags_by_project: dict[str, set[str]]
    tags_by_node: dict[str, set[str]]
    outgoing: dict[str, list[Edge]]
    incoming: dict[str, list[Edge]]


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"CSV がありません: {path}")
    with path.open(encoding="utf-8-sig", newline="") as fh:
        return list(csv.DictReader(fh))


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def required(row: dict[str, str], key: str, path: Path) -> str:
    value = (row.get(key) or "").strip()
    if not value:
        raise ValueError(f"{path.name} に空の {key} があります: {row}")
    return value


def load_projects(data_dir: Path = DATA_DIR) -> dict[str, Project]:
    path = data_dir / "projects.csv"
    projects: dict[str, Project] = {}
    for row in read_csv(path):
        project_id = (row.get("案件ID") or "").strip()
        if not project_id:
            continue
        if project_id in projects:
            raise ValueError(f"案件ID が重複しています: {project_id}")
        projects[project_id] = Project(
            id=project_id,
            name=(row.get("案件名") or "").strip(),
            role=(row.get("想定職種") or "").strip(),
            domain=(row.get("ドメイン") or "").strip(),
            cloud=(row.get("クラウド") or "").strip(),
            summary=(row.get("案件概要") or "").strip(),
            skills=(row.get("必要スキル") or "").strip(),
        )
    return projects


def load_questions(data_dir: Path = DATA_DIR) -> list[Question]:
    path = data_dir / "questions.csv"
    questions: list[Question] = []
    for row in read_csv(path):
        question_id = (row.get("質問ID") or "").strip()
        text = (row.get("質問") or "").strip()
        if not question_id or not text:
            continue
        raw_gold = (row.get("正解案件ID") or "").strip()
        gold_ids = tuple(part.strip() for part in raw_gold.split(",") if part.strip())
        evaluation_split = (row.get("評価区分") or "development").strip().lower()
        if evaluation_split not in {"development", "holdout"}:
            raise ValueError(
                f"{question_id} の評価区分が不正です: {evaluation_split}"
            )
        questions.append(
            Question(
                id=question_id,
                text=text,
                gold_ids=gold_ids,
                category=(row.get("分類") or "").strip(),
                plan_id=(row.get("検索計画") or question_id).strip(),
                evaluation_split=evaluation_split,
            )
        )
    return questions


def build_graph(
    nodes: dict[str, Node],
    edges: list[Edge],
    projects: dict[str, Project],
    tags_by_project: dict[str, set[str]],
) -> KnowledgeGraph:
    outgoing: dict[str, list[Edge]] = defaultdict(list)
    incoming: dict[str, list[Edge]] = defaultdict(list)
    for edge in edges:
        outgoing[edge.source].append(edge)
        incoming[edge.target].append(edge)
    tags_by_node: dict[str, set[str]] = defaultdict(set)
    for project_id, node_ids in tags_by_project.items():
        for node_id in node_ids:
            tags_by_node[node_id].add(project_id)
    return KnowledgeGraph(
        nodes=nodes,
        edges=edges,
        projects=projects,
        tags_by_project=dict(tags_by_project),
        tags_by_node=dict(tags_by_node),
        outgoing=dict(outgoing),
        incoming=dict(incoming),
    )


def validate(graph: KnowledgeGraph) -> list[str]:
    warnings: list[str] = []
    for edge in graph.edges:
        if edge.source not in graph.nodes:
            warnings.append(f"エッジ {edge.id} の出発 {edge.source} がノードにありません")
        if edge.target not in graph.nodes:
            warnings.append(f"エッジ {edge.id} の到着 {edge.target} がノードにありません")
    for project_id, node_ids in graph.tags_by_project.items():
        if project_id not in graph.projects:
            warnings.append(f"タグの案件 {project_id} が projects.csv にありません")
        for node_id in node_ids:
            if node_id not in graph.nodes:
                warnings.append(f"タグ {project_id} -> {node_id} のノードがありません")
    return warnings


def collect_neighborhood(
    graph: KnowledgeGraph,
    start_id: str,
    relations: frozenset[str] | None = None,
    bidirectional: bool = False,
) -> set[str]:
    if start_id not in graph.nodes:
        raise KeyError(f"起点ノードがありません: {start_id}")

    seen = {start_id}
    stack = [start_id]
    while stack:
        current = stack.pop()
        candidates = list(graph.outgoing.get(current, []))
        if bidirectional:
            candidates.extend(graph.incoming.get(current, []))
        for edge in candidates:
            if relations is not None and edge.relation not in relations:
                continue
            nxt = edge.target if edge.source == current else edge.source
            if nxt in seen:
                continue
            seen.add(nxt)
            stack.append(nxt)
    return seen


def projects_for_nodes(graph: KnowledgeGraph, node_ids: set[str]) -> list[str]:
    found: set[str] = set()
    for node_id in node_ids:
        found.update(graph.tags_by_node.get(node_id, set()))
    return sorted(found)


def print_stats(graph: KnowledgeGraph) -> None:
    kinds: dict[str, int] = defaultdict(int)
    for node in graph.nodes.values():
        kinds[node.kind] += 1
    relations: dict[str, int] = defaultdict(int)
    for edge in graph.edges:
        relations[edge.relation] += 1

    print(f"ノード: {len(graph.nodes)}  {dict(kinds)}")
    print(f"エッジ: {len(graph.edges)}  {dict(relations)}")
    print(f"案件: {len(graph.projects)}")
    print(f"タグ: {sum(len(v) for v in graph.tags_by_project.values())}")


def print_query(
    graph: KnowledgeGraph,
    start_id: str,
    relations: frozenset[str] | None = None,
    bidirectional: bool = False,
) -> None:
    neighborhood = collect_neighborhood(
        graph, start_id, relations=relations, bidirectional=bidirectional
    )
    project_ids = projects_for_nodes(graph, neighborhood)
    start = graph.nodes[start_id]

    print(f"起点: {start.id} ({start.name})")
    print("辿ったノード:")
    for node_id in sorted(neighborhood):
        node = graph.nodes[node_id]
        print(f"  - {node.id}  {node.kind}  {node.name}")
    print(f"ヒット案件: {len(project_ids)}")
    for project_id in project_ids:
        project = graph.projects.get(project_id)
        name = project.name if project else ""
        print(f"  - {project_id}  {name}")


def recall(gold: tuple[str, ...], predicted: list[str]) -> float:
    if not gold:
        return 1.0 if not predicted else 0.0
    hit = len(set(gold) & set(predicted))
    return hit / len(gold)


def precision(gold: tuple[str, ...], predicted: list[str]) -> float:
    if not predicted:
        return 1.0 if not gold else 0.0
    hit = len(set(gold) & set(predicted))
    return hit / len(predicted)


def f1_score(gold: tuple[str, ...], predicted: list[str]) -> float:
    rec = recall(gold, predicted)
    prec = precision(gold, predicted)
    if rec + prec == 0:
        return 0.0
    return 2 * rec * prec / (rec + prec)
