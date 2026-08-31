"""LLM にオントロジーを作らせる GraphRAG。

人が作った nodes/edges/tags は使わない。案件文だけを渡し、
LangChain の LLMGraphTransformer がノードと関係を抽出する。
Microsoft GraphRAG と同じ系統（AIがグラフを invent する）の軽量版。
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from communities import (
    GLOBAL_CATEGORIES,
    SUMMARY_DIR,
    load_communities,
    rank_communities,
    search_by_summaries,
)
from graph_core import (
    DATA_DIR,
    Edge,
    Node,
    bedrock_chat_model_id,
    bedrock_runtime_client,
    build_graph,
    collect_neighborhood,
    f1_score,
    load_projects,
    load_env,
    load_questions,
    precision,
    print_query,
    print_stats,
    projects_for_nodes,
    read_csv,
    recall,
    validate,
    write_csv,
)

GENERATED_DIRNAME = "llm_generated"


def node_id_from_name(name: str) -> str:
    slug = re.sub(r"[^0-9A-Za-z一-龥ぁ-んァ-ン]+", "_", name).strip("_")
    if not slug:
        slug = f"X{abs(hash(name)) % 10000}"
    return f"N_{slug.upper()[:40]}"


def _normalize_name(name: str) -> str:
    return re.sub(r"\s+", "", name).lower()


class ExtractedGraph:
    def __init__(self) -> None:
        self.nodes: dict[str, Node] = {}
        self.edges: list[Edge] = []
        self.tags_by_project: dict[str, set[str]] = defaultdict(set)
        self._edge_keys: set[tuple[str, str, str]] = set()
        self._name_index: dict[str, str] = {}

    def add_node(self, name: str, kind: str, description: str = "") -> str:
        key = _normalize_name(name)
        if key in self._name_index:
            return self._name_index[key]
        node_id = node_id_from_name(name)
        suffix = 2
        while node_id in self.nodes:
            node_id = f"{node_id_from_name(name)}_{suffix}"
            suffix += 1
        self.nodes[node_id] = Node(id=node_id, name=name.strip(), kind=kind or "概念", description=description)
        self._name_index[key] = node_id
        return node_id

    def add_edge(self, source_name: str, target_name: str, relation: str) -> None:
        source_id = self.add_node(source_name, "概念")
        target_id = self.add_node(target_name, "概念")
        rel = (relation or "関連").strip() or "関連"
        key = (source_id, target_id, rel)
        if key in self._edge_keys:
            return
        self._edge_keys.add(key)
        self.edges.append(
            Edge(
                id=f"E{len(self.edges) + 1:03d}",
                source=source_id,
                target=target_id,
                relation=rel,
                weight=1.0,
            )
        )

    def tag(self, project_id: str, node_id: str) -> None:
        self.tags_by_project[project_id].add(node_id)


def extract_local(projects: dict) -> ExtractedGraph:
    """APIなしの仮抽出。職種・スキル・ドメインをノードにするだけ。比較に使わない。"""
    extracted = ExtractedGraph()
    for project in projects.values():
        if project.role:
            extracted.tag(project.id, extracted.add_node(project.role, "職種"))
        if project.domain:
            extracted.tag(project.id, extracted.add_node(project.domain, "ドメイン"))
        for skill in (part.strip() for part in project.skills.split("/") if part.strip()):
            skill_id = extracted.add_node(skill, "スキル")
            extracted.tag(project.id, skill_id)
            if project.role:
                extracted.add_edge(project.role, skill, "使用")
    return extracted


def extract_bedrock_json(projects: dict, model: str) -> ExtractedGraph:
    """Claude on Bedrock にノードと関係の JSON を出させる。"""
    client = bedrock_runtime_client()
    model = bedrock_chat_model_id(model)
    print(f"Bedrock chat: {model}")
    extracted = ExtractedGraph()
    system = (
        "案件説明から知識グラフを作る。人が用意したオントロジーは使わない。"
        "JSONだけ返す。"
        '{"nodes":[{"name":"","type":"職種|業務|スキル|その他"}],'
        '"edges":[{"source":"","target":"","relation":"包含|関連|使用"}]}'
    )
    for project in projects.values():
        response = client.converse(
            modelId=model,
            system=[{"text": system}],
            messages=[{"role": "user", "content": [{"text": project.document()}]}],
            inferenceConfig={"temperature": 0, "maxTokens": 2048},
        )
        content = response["output"]["message"]["content"][0]["text"]
        payload = _parse_json_content(content)
        for node in payload.get("nodes") or []:
            name = str(node.get("name") or "").strip()
            if not name:
                continue
            node_id = extracted.add_node(name, str(node.get("type") or "概念"))
            extracted.tag(project.id, node_id)
        for edge in payload.get("edges") or []:
            source = str(edge.get("source") or "").strip()
            target = str(edge.get("target") or "").strip()
            if source and target:
                extracted.add_edge(source, target, str(edge.get("relation") or "関連"))
    return extracted


def _parse_json_content(text: str) -> dict:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped)
        stripped = re.sub(r"\s*```$", "", stripped)
    return json.loads(stripped)


def extract_langchain(projects: dict, model: str) -> ExtractedGraph:
    """LangChain LLMGraphTransformer。許可リストは渡さず、AIに自由に作らせる。"""
    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY がありません。キーを設定するか --llm local を使ってください。"
        )
    try:
        from langchain_core.documents import Document
        from langchain_experimental.graph_transformers import LLMGraphTransformer
        from langchain_openai import ChatOpenAI
    except ImportError as exc:
        raise RuntimeError(
            "langchain 一式がありません。pip install langchain-experimental langchain-openai langchain-core"
        ) from exc

    llm = ChatOpenAI(api_key=api_key, model=model, temperature=0)
    transformer = LLMGraphTransformer(llm=llm)
    extracted = ExtractedGraph()

    for project in projects.values():
        documents = [Document(page_content=project.document(), metadata={"案件ID": project.id})]
        graph_docs = transformer.convert_to_graph_documents(documents)
        for graph_doc in graph_docs:
            for node in graph_doc.nodes:
                node_id = extracted.add_node(node.id, getattr(node, "type", "") or "概念")
                extracted.tag(project.id, node_id)
            for rel in graph_doc.relationships:
                source_name = getattr(rel.source, "id", str(rel.source))
                target_name = getattr(rel.target, "id", str(rel.target))
                extracted.add_edge(source_name, target_name, getattr(rel, "type", "") or "関連")
                extracted.tag(project.id, extracted.add_node(source_name, "概念"))
                extracted.tag(project.id, extracted.add_node(target_name, "概念"))
    return extracted


def extract_openai_json(projects: dict, model: str) -> ExtractedGraph:
    """LangChain が使えないときの同等抽出。案件文からノードと関係の JSON を出させる。"""
    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY がありません。キーを設定するか --llm local を使ってください。"
        )
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError("openai パッケージがありません。pip install openai") from exc

    client = OpenAI(api_key=api_key)
    extracted = ExtractedGraph()
    system = (
        "案件説明から知識グラフを作る。人が用意したオントロジーは使わない。"
        "JSONだけ返す。"
        '{"nodes":[{"name":"","type":"職種|業務|スキル|その他"}],'
        '"edges":[{"source":"","target":"","relation":"包含|関連|使用"}]}'
    )
    for project in projects.values():
        response = client.chat.completions.create(
            model=model,
            temperature=0,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": project.document()},
            ],
        )
        content = response.choices[0].message.content or "{}"
        payload = json.loads(content)
        for node in payload.get("nodes") or []:
            name = str(node.get("name") or "").strip()
            if not name:
                continue
            node_id = extracted.add_node(name, str(node.get("type") or "概念"))
            extracted.tag(project.id, node_id)
        for edge in payload.get("edges") or []:
            source = str(edge.get("source") or "").strip()
            target = str(edge.get("target") or "").strip()
            if source and target:
                extracted.add_edge(source, target, str(edge.get("relation") or "関連"))
    return extracted


def save_extracted(extracted: ExtractedGraph, output_dir: Path) -> None:
    write_csv(
        output_dir / "nodes.csv",
        ["ノードID", "正規名", "種別", "説明"],
        [
            {
                "ノードID": node.id,
                "正規名": node.name,
                "種別": node.kind,
                "説明": node.description,
            }
            for node in extracted.nodes.values()
        ],
    )
    write_csv(
        output_dir / "edges.csv",
        ["エッジID", "出発ノードID", "到着ノードID", "関係", "強さ"],
        [
            {
                "エッジID": edge.id,
                "出発ノードID": edge.source,
                "到着ノードID": edge.target,
                "関係": edge.relation,
                "強さ": str(edge.weight),
            }
            for edge in extracted.edges
        ],
    )
    tag_rows = [
        {"案件ID": project_id, "ノードID": node_id}
        for project_id, node_ids in sorted(extracted.tags_by_project.items())
        for node_id in sorted(node_ids)
    ]
    write_csv(output_dir / "tags.csv", ["案件ID", "ノードID"], tag_rows)


def load_extracted(output_dir: Path) -> ExtractedGraph:
    extracted = ExtractedGraph()
    for row in read_csv(output_dir / "nodes.csv"):
        node_id = (row.get("ノードID") or "").strip()
        name = (row.get("正規名") or "").strip()
        if not node_id or not name:
            continue
        extracted.nodes[node_id] = Node(
            id=node_id,
            name=name,
            kind=(row.get("種別") or "概念").strip(),
            description=(row.get("説明") or "").strip(),
        )
        extracted._name_index[_normalize_name(name)] = node_id
    for row in read_csv(output_dir / "edges.csv"):
        source = (row.get("出発ノードID") or "").strip()
        target = (row.get("到着ノードID") or "").strip()
        relation = (row.get("関係") or "関連").strip()
        if not source or not target:
            continue
        extracted.edges.append(
            Edge(
                id=(row.get("エッジID") or f"E{len(extracted.edges) + 1:03d}"),
                source=source,
                target=target,
                relation=relation,
                weight=float((row.get("強さ") or "1").strip() or "1"),
            )
        )
        extracted._edge_keys.add((source, target, relation))
    for row in read_csv(output_dir / "tags.csv"):
        project_id = (row.get("案件ID") or "").strip()
        node_id = (row.get("ノードID") or "").strip()
        if project_id and node_id:
            extracted.tag(project_id, node_id)
    extracted.tags_by_project = defaultdict(set, extracted.tags_by_project)
    if not extracted.nodes:
        raise ValueError(f"{output_dir} の生成結果が空です")
    return extracted


def cache_exists(output_dir: Path) -> bool:
    return (output_dir / "nodes.csv").exists() and (output_dir / "nodes.csv").stat().st_size > 20


def find_start_ids(graph, query: str) -> list[str]:
    needle = _normalize_name(query)
    return [
        node.id
        for node in graph.nodes.values()
        if _normalize_name(node.name) and _normalize_name(node.name) in needle
    ]


def search_projects(graph, query: str) -> list[str]:
    start_ids = find_start_ids(graph, query)
    found: set[str] = set()
    for start_id in start_ids:
        found.update(projects_for_nodes(graph, collect_neighborhood(graph, start_id, bidirectional=True)))
    return sorted(found)


def run_eval(graph, questions, k: int, communities=None, summary_k: int = 3) -> None:
    from collections import defaultdict

    communities = communities or []
    by_category: dict[str, list[tuple[float, float, float]]] = defaultdict(list)
    mode = "ローカル + 全体像は要約" if communities else "ローカル探索のみ"
    print(f"評価: {len(questions)} 問  （LLMグラフ、{mode}）")
    if communities:
        print(f"全体像は要約上位 {summary_k} 件")
    for question in questions:
        if communities and question.category in GLOBAL_CATEGORIES:
            predicted = search_by_summaries(question.text, communities, top_k=summary_k)
            used = "global"
        else:
            predicted = search_projects(graph, question.text)[:k]
            used = "local"
        rec = recall(question.gold_ids, predicted)
        prec = precision(question.gold_ids, predicted)
        f1 = f1_score(question.gold_ids, predicted)
        by_category[question.category or "未分類"].append((rec, prec, f1))
        gold = ",".join(question.gold_ids) if question.gold_ids else "該当なし"
        print(f"{question.id}  {question.category}  {used}  recall={rec:.2f}  precision={prec:.2f}  f1={f1:.2f}")
        print(f"  gold: {gold}")
        print(f"  llm:  {','.join(predicted) if predicted else '該当なし'}")
        if used == "global":
            for item in rank_communities(question.text, communities, top_k=summary_k):
                print(f"  要約 {item.id} {item.title}: {item.summary[:180]}")
    print("分類別")
    for category, scores in by_category.items():
        n = len(scores)
        avg_rec = sum(item[0] for item in scores) / n
        avg_prec = sum(item[1] for item in scores) / n
        avg_f1 = sum(item[2] for item in scores) / n
        print(f"  {category}: recall={avg_rec:.2f}  precision={avg_prec:.2f}  f1={avg_f1:.2f}  ({n}問)")


def main() -> int:
    parser = argparse.ArgumentParser(description="LLM がオントロジーを作る GraphRAG")
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--start", default="", help="起点のノード名またはID。例: MLOps")
    parser.add_argument("--query", default="", help="質問文。名前が含まれるノードから辿る")
    parser.add_argument("--eval", action="store_true", help="questions.csv で一括採点する")
    parser.add_argument("--k", type=int, default=28, help="評価時に使う最大件数")
    parser.add_argument("--summary-k", type=int, default=3, help="全体像で使うコミュニティ要約の件数")
    parser.add_argument("--llm", choices=("local", "bedrock", "openai"), default="local")
    parser.add_argument(
        "--extractor",
        choices=("langchain", "openai-json"),
        default="langchain",
        help="openai 時の抽出器。langchain が無ければ openai-json に倒す",
    )
    parser.add_argument("--model", default="", help="未指定なら BEDROCK_CHAT_MODEL または gpt-4o-mini")
    parser.add_argument("--rebuild", action="store_true", help="キャッシュを無視して再抽出する")
    args = parser.parse_args()
    load_env()

    output_dir = args.data_dir / GENERATED_DIRNAME
    try:
        projects = load_projects(args.data_dir)
    except (FileNotFoundError, ValueError) as exc:
        print(f"読み込みエラー: {exc}", file=sys.stderr)
        return 1

    if not projects:
        print("projects.csv が空です。", file=sys.stderr)
        return 1

    chat_model = bedrock_chat_model_id(args.model)
    openai_model = args.model or "gpt-4o-mini"

    try:
        if cache_exists(output_dir) and not args.rebuild:
            print(f"キャッシュを使用: {output_dir}")
            extracted = load_extracted(output_dir)
        elif args.llm == "local":
            extracted = extract_local(projects)
            save_extracted(extracted, output_dir)
        elif args.llm == "bedrock":
            extracted = extract_bedrock_json(projects, chat_model)
            save_extracted(extracted, output_dir)
        elif args.extractor == "openai-json":
            extracted = extract_openai_json(projects, openai_model)
            save_extracted(extracted, output_dir)
        else:
            try:
                extracted = extract_langchain(projects, openai_model)
            except RuntimeError as exc:
                if "langchain" in str(exc):
                    print(f"警告: {exc}  openai-json に切り替えます。", file=sys.stderr)
                    extracted = extract_openai_json(projects, openai_model)
                else:
                    raise
            save_extracted(extracted, output_dir)
    except (RuntimeError, ValueError, json.JSONDecodeError) as exc:
        print(f"抽出エラー: {exc}", file=sys.stderr)
        return 1

    graph = build_graph(extracted.nodes, extracted.edges, projects, dict(extracted.tags_by_project))
    print(f"生成元: {'cache' if cache_exists(output_dir) and not args.rebuild else args.llm}")
    print_stats(graph)
    for warning in validate(graph):
        print(f"警告: {warning}", file=sys.stderr)
    print(f"生成CSV: {output_dir}")

    if args.start:
        start_id = args.start if args.start in graph.nodes else None
        if start_id is None:
            matches = find_start_ids(graph, args.start)
            if not matches:
                print(f"起点が見つかりません: {args.start}", file=sys.stderr)
                return 1
            start_id = matches[0]
            print(f"名前から解決: {args.start} -> {start_id}")
        print_query(graph, start_id, bidirectional=True)

    if args.query:
        print(f"質問: {args.query}")
        hits = search_projects(graph, args.query)
        print(f"ヒット案件: {len(hits)}")
        for project_id in hits:
            project = graph.projects.get(project_id)
            print(f"  - {project_id}  {project.name if project else ''}")

    if args.eval:
        questions = load_questions(args.data_dir)
        if not questions:
            print("questions.csv が空です。", file=sys.stderr)
            return 1
        communities = load_communities(SUMMARY_DIR / "llm.csv")
        if communities:
            print(f"コミュニティ要約: {len(communities)} 件を全体像に使用")
        run_eval(graph, questions, k=args.k, communities=communities, summary_k=args.summary_k)

    if not args.start and not args.query and not args.eval:
        print("使い方: --start MLOps  /  --query \"...\"  /  --eval")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
