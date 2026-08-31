"""コミュニティ検出と要約。BYOG / LLM Graph で同じ手順を使う。"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

from graph_core import (
    DATA_DIR,
    KnowledgeGraph,
    bedrock_runtime_client,
    projects_for_nodes,
    read_csv,
    write_csv,
)

SUMMARY_DIR = DATA_DIR / "community_summaries"
GLOBAL_CATEGORIES = frozenset({"全体像"})


@dataclass
class Community:
    id: str
    title: str
    node_ids: tuple[str, ...]
    project_ids: tuple[str, ...]
    summary: str = ""


def detect_communities(
    graph: KnowledgeGraph,
    relations: frozenset[str] | None = None,
    allowed_node_kinds: frozenset[str] | None = None,
) -> list[Community]:
    import networkx as nx
    from networkx.algorithms.community import louvain_communities

    nx_graph = nx.Graph()
    nx_graph.add_nodes_from(
        node_id
        for node_id, node in graph.nodes.items()
        if allowed_node_kinds is None or node.kind in allowed_node_kinds
    )
    for edge in graph.edges:
        if relations is not None and edge.relation not in relations:
            continue
        if nx_graph.has_node(edge.source) and nx_graph.has_node(edge.target):
            nx_graph.add_edge(edge.source, edge.target)

    raw = louvain_communities(nx_graph, seed=42)
    communities: list[Community] = []
    for index, node_set in enumerate(sorted(raw, key=len, reverse=True), start=1):
        node_ids = tuple(sorted(node_set))
        project_ids = tuple(projects_for_nodes(graph, set(node_ids)))
        if not project_ids:
            continue
        names = [graph.nodes[node_id].name for node_id in node_ids if node_id in graph.nodes]
        title = " / ".join(names[:3]) if names else f"community-{index}"
        communities.append(
            Community(
                id=f"C{index:02d}",
                title=title,
                node_ids=node_ids,
                project_ids=project_ids,
            )
        )
    return communities


def _community_prompt(graph: KnowledgeGraph, community: Community) -> str:
    lines = [
        "次の案件群を「コミュニティ」として要約してください。",
        "個別の列挙ではなく、よく出る業務・スキル・傾向を200〜400字で書いてください。",
        f"コミュニティ名の候補: {community.title}",
        "案件:",
    ]
    for project_id in community.project_ids:
        project = graph.projects[project_id]
        lines.append(f"- {project_id} {project.name} / {project.role} / {project.summary} / {project.skills}")
    return "\n".join(lines)


def summarize_communities(graph: KnowledgeGraph, communities: list[Community], model: str) -> list[Community]:
    client = bedrock_runtime_client()
    filled: list[Community] = []
    for community in communities:
        response = client.converse(
            modelId=model,
            system=[{"text": "案件検索用のコミュニティ要約だけを日本語で返す。前置きは不要。"}],
            messages=[{"role": "user", "content": [{"text": _community_prompt(graph, community)}]}],
            inferenceConfig={"temperature": 0, "maxTokens": 800},
        )
        text = response["output"]["message"]["content"][0]["text"].strip()
        filled.append(
            Community(
                id=community.id,
                title=community.title,
                node_ids=community.node_ids,
                project_ids=community.project_ids,
                summary=text,
            )
        )
    return filled


def save_communities(path: Path, communities: list[Community]) -> None:
    write_csv(
        path,
        ["community_id", "title", "node_ids", "project_ids", "summary"],
        [
            {
                "community_id": item.id,
                "title": item.title,
                "node_ids": ",".join(item.node_ids),
                "project_ids": ",".join(item.project_ids),
                "summary": item.summary,
            }
            for item in communities
        ],
    )


def load_communities(path: Path) -> list[Community]:
    if not path.exists():
        return []
    items: list[Community] = []
    for row in read_csv(path):
        summary = (row.get("summary") or "").strip()
        if not summary:
            continue
        items.append(
            Community(
                id=(row.get("community_id") or "").strip(),
                title=(row.get("title") or "").strip(),
                node_ids=tuple(part for part in (row.get("node_ids") or "").split(",") if part),
                project_ids=tuple(part for part in (row.get("project_ids") or "").split(",") if part),
                summary=summary,
            )
        )
    return items


def _cosine(left: list[float], right: list[float]) -> float:
    if not left or not right or len(left) != len(right):
        return 0.0
    return sum(a * b for a, b in zip(left, right))


def _embed_texts(texts: list[str]) -> list[list[float]]:
    client = bedrock_runtime_client()
    model = os.environ.get("BEDROCK_EMBED_MODEL", "amazon.titan-embed-text-v2:0")
    vectors: list[list[float]] = []
    for text in texts:
        response = client.invoke_model(
            modelId=model,
            contentType="application/json",
            accept="application/json",
            body=json.dumps({"inputText": text or " ", "normalize": True}),
        )
        payload = json.loads(response["body"].read())
        vectors.append(payload["embedding"])
    return vectors


def rank_communities(query: str, communities: list[Community], top_k: int = 3) -> list[Community]:
    if not communities:
        return []
    texts = [f"{item.title}\n{item.summary}" for item in communities]
    doc_vectors = _embed_texts(texts)
    query_vector = _embed_texts([query])[0]
    ranked = sorted(
        zip(communities, doc_vectors),
        key=lambda pair: _cosine(query_vector, pair[1]),
        reverse=True,
    )
    return [community for community, _vector in ranked[:top_k]]


def search_by_summaries(query: str, communities: list[Community], top_k: int = 3) -> list[str]:
    found: list[str] = []
    seen: set[str] = set()
    for community in rank_communities(query, communities, top_k=top_k):
        for project_id in community.project_ids:
            if project_id not in seen:
                seen.add(project_id)
                found.append(project_id)
    return found
