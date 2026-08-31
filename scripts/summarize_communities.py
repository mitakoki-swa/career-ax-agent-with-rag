"""グラフからコミュニティを切り、Claude で要約する。BYOG と LLM Graph で同じ処理。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from build_graph_byog import TRAVERSE_RELATIONS, load_byog_graph
from build_graph_llm import GENERATED_DIRNAME, load_extracted
from communities import (
    SUMMARY_DIR,
    detect_communities,
    save_communities,
    summarize_communities,
)
from graph_core import DATA_DIR, bedrock_chat_model_id, build_graph, load_env, load_projects


def load_llm_graph(data_dir: Path):
    projects = load_projects(data_dir)
    extracted = load_extracted(data_dir / GENERATED_DIRNAME)
    return build_graph(extracted.nodes, extracted.edges, projects, dict(extracted.tags_by_project))


def main() -> int:
    parser = argparse.ArgumentParser(description="コミュニティ要約を作る")
    parser.add_argument("--source", choices=("byog", "llm"), required=True)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--model", default="")
    args = parser.parse_args()
    load_env()
    model = bedrock_chat_model_id(args.model)

    if args.source == "byog":
        graph = load_byog_graph(args.data_dir)
        relations: frozenset[str] | None = TRAVERSE_RELATIONS
        allowed_node_kinds: frozenset[str] | None = frozenset({"職種", "業務"})
    else:
        graph = load_llm_graph(args.data_dir)
        relations = None
        allowed_node_kinds = None

    communities = detect_communities(
        graph,
        relations=relations,
        allowed_node_kinds=allowed_node_kinds,
    )
    print(f"コミュニティ: {len(communities)}  案件つきのみ要約します")
    summarized = summarize_communities(graph, communities, model)
    output = SUMMARY_DIR / f"{args.source}.csv"
    save_communities(output, summarized)
    for item in summarized:
        print(f"- {item.id}  {item.title}  案件{len(item.project_ids)}件")
        print(f"  {item.summary[:120]}...")
    print(f"保存: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
