"""人が作ったオントロジー（BYOG）からグラフを組む。"""

from __future__ import annotations

import argparse
import json
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
from byog_search import ByogSearcher
from graph_core import (
    DATA_DIR,
    Edge,
    Node,
    build_graph,
    collect_neighborhood,
    f1_score,
    load_projects,
    load_questions,
    precision,
    print_query,
    print_stats,
    projects_for_nodes,
    read_csv,
    recall,
    required,
    validate,
)
from retrieval_core import (
    CachedEmbedder,
    ProjectReranker,
    cosine_similarity,
    load_search_plans,
    metrics_at_k,
)
from search_rag import (
    build_embedder,
    candidate_evaluation,
    candidate_evaluation_text,
    evaluation_record,
    metrics_dict,
    parse_ks,
    print_category_summary,
    summarize_by_category,
    summarize_candidates,
)

TRAVERSE_RELATIONS = frozenset({"包含", "関連"})


def load_byog_graph(data_dir: Path = DATA_DIR):
    node_rows = read_csv(data_dir / "nodes.csv")
    edge_rows = read_csv(data_dir / "edges.csv")
    tag_rows = read_csv(data_dir / "tags.csv")
    projects = load_projects(data_dir)

    nodes: dict[str, Node] = {}
    for row in node_rows:
        node_id = required(row, "ノードID", data_dir / "nodes.csv")
        if node_id in nodes:
            raise ValueError(f"ノードID が重複しています: {node_id}")
        nodes[node_id] = Node(
            id=node_id,
            name=required(row, "正規名", data_dir / "nodes.csv"),
            kind=required(row, "種別", data_dir / "nodes.csv"),
            description=(row.get("説明") or "").strip(),
        )

    edges: list[Edge] = []
    for row in edge_rows:
        edges.append(
            Edge(
                id=required(row, "エッジID", data_dir / "edges.csv"),
                source=required(row, "出発ノードID", data_dir / "edges.csv"),
                target=required(row, "到着ノードID", data_dir / "edges.csv"),
                relation=required(row, "関係", data_dir / "edges.csv"),
                weight=float((row.get("強さ") or "1").strip() or "1"),
            )
        )

    tags_by_project: dict[str, set[str]] = defaultdict(set)
    for row in tag_rows:
        tags_by_project[required(row, "案件ID", data_dir / "tags.csv")].add(
            required(row, "ノードID", data_dir / "tags.csv")
        )

    return build_graph(nodes, edges, projects, dict(tags_by_project))


def _norm(text: str) -> str:
    return re.sub(r"\s+", "", text).lower()


def search_projects(graph, query: str) -> list[str]:
    found: set[str] = set()
    for project_id in graph.projects:
        if project_id.lower() in query.lower():
            found.add(project_id)

    needle = _norm(query)
    start_ids = [
        node.id
        for node in graph.nodes.values()
        if _norm(node.name) and _norm(node.name) in needle
    ]
    for start_id in start_ids:
        neighborhood = collect_neighborhood(graph, start_id, relations=TRAVERSE_RELATIONS)
        found.update(projects_for_nodes(graph, neighborhood))
    return sorted(found)


def run_eval(graph, questions, communities=None, summary_k: int = 3) -> None:
    communities = communities or []
    by_category: dict[str, list[tuple[float, float, float]]] = defaultdict(list)
    mode = "ローカル探索 + 全体像はコミュニティ要約" if communities else "ローカル探索のみ"
    print(f"評価: {len(questions)} 問  （BYOG、{mode}）")
    if communities:
        print(f"全体像は要約上位 {summary_k} 件")
    for question in questions:
        if communities and question.category in GLOBAL_CATEGORIES:
            predicted = search_by_summaries(question.text, communities, top_k=summary_k)
            used = "global"
        else:
            predicted = search_projects(graph, question.text)
            used = "local"
        rec = recall(question.gold_ids, predicted)
        prec = precision(question.gold_ids, predicted)
        f1 = f1_score(question.gold_ids, predicted)
        by_category[question.category or "未分類"].append((rec, prec, f1))
        gold = ",".join(question.gold_ids) if question.gold_ids else "該当なし"
        print(f"{question.id}  {question.category}  {used}  recall={rec:.2f}  precision={prec:.2f}  f1={f1:.2f}")
        print(f"  gold: {gold}")
        print(f"  byog: {','.join(predicted) if predicted else '該当なし'}")
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


def run_plan_eval(
    graph,
    questions,
    plans,
    searcher: ByogSearcher,
    ks: tuple[int, ...],
    min_score: float,
    communities=None,
    summary_k: int = 3,
) -> dict[str, object]:
    communities = communities or []
    totals = {k: [0.0, 0.0, 0.0, 0.0] for k in ks}
    graph_only_totals = {k: [0.0, 0.0, 0.0, 0.0] for k in ks}
    graph_only_count = 0
    positive_count = 0
    no_answer_count = 0
    no_answer_correct = 0
    records: list[dict[str, object]] = []
    print(
        f"\n--- BYOG 計画評価 + Graph-first/Titan tie-break "
        f"k={','.join(map(str, ks))} "
        f"summary-k={summary_k} ---"
    )
    for question in questions:
        plan = plans.get(question.plan_id)
        if plan is None:
            raise ValueError(f"{question.id} の検索計画がありません")
        global_candidates = None
        if plan.mode == "global_summary":
            if not communities:
                raise ValueError(
                    f"{question.id} は global_summary ですが要約CSVがありません"
                )
            if searcher.reranker is None:
                raise ValueError("global_summary にはrerankerが必要です")
            global_candidates = community_candidate_ids(
                query=plan.augmented_query(question.text),
                communities=communities,
                embedder=searcher.reranker.embedder,
                top_k=summary_k,
            )
        candidates = searcher.candidates(plan, global_candidates)
        candidate_info = candidate_evaluation(
            question.gold_ids, candidates.project_ids
        )
        hits = searcher.rerank_candidates(
            query=question.text,
            plan=plan,
            candidates=candidates,
            k=max(ks),
            min_score=min_score,
        )
        graph_only_applicable = plan.mode != "global_summary"
        graph_only_hits = (
            searcher.graph_only_hits(candidates, max(ks))
            if graph_only_applicable
            else []
        )
        predicted = [hit.project_id for hit in hits]
        graph_only_predicted = [hit.project_id for hit in graph_only_hits]
        print(
            f"{question.id} [{question.category}] "
            f"mode={plan.mode} status={plan.status}"
        )
        print(f"  正解={list(question.gold_ids)}")
        print(f"  候補={candidate_evaluation_text(candidate_info)}")
        print(
            "  予測="
            + str(
                [
                    (
                        f"{hit.project_id}:{hit.score:.3f}"
                        f"(v={hit.vector_score:.3f},g={hit.graph_score:.3f})"
                    )
                    for hit in hits
                ]
            )
        )
        if graph_only_hits:
            print(
                "  Graph-only="
                + str(
                    [
                        f"{hit.project_id}:g={hit.graph_score:.3f}"
                        for hit in graph_only_hits
                    ]
                )
            )
        if not question.gold_ids:
            no_answer_count += 1
            no_answer_correct += int(not predicted)
            print(f"  No-answer={'正解' if not predicted else '不正解'}")
            records.append(
                evaluation_record(
                    question,
                    plan,
                    hits,
                    {},
                    not predicted,
                    candidates=candidate_info,
                    graph_only_hits=graph_only_hits,
                )
            )
            continue
        positive_count += 1
        question_metrics: dict[str, dict[str, float | int]] = {}
        graph_only_metrics: dict[str, dict[str, float | int]] = {}
        if graph_only_applicable:
            graph_only_count += 1
        for k in ks:
            metric = metrics_at_k(question.gold_ids, predicted, k)
            totals[k][0] += metric.recall
            totals[k][1] += metric.precision
            totals[k][2] += metric.f1
            totals[k][3] += metric.ndcg
            print(
                f"  @{k} Recall={metric.recall:.3f} "
                f"Precision={metric.precision:.3f} "
                f"F1={metric.f1:.3f} nDCG={metric.ndcg:.3f}"
            )
            question_metrics[str(k)] = metrics_dict(metric)
            if graph_only_applicable:
                graph_metric = metrics_at_k(
                    question.gold_ids, graph_only_predicted, k
                )
                graph_only_metrics[str(k)] = metrics_dict(graph_metric)
                graph_only_totals[k][0] += graph_metric.recall
                graph_only_totals[k][1] += graph_metric.precision
                graph_only_totals[k][2] += graph_metric.f1
                graph_only_totals[k][3] += graph_metric.ndcg
                print(
                    f"    Graph-only @{k} Recall={graph_metric.recall:.3f} "
                    f"Precision={graph_metric.precision:.3f} "
                    f"F1={graph_metric.f1:.3f} nDCG={graph_metric.ndcg:.3f}"
                )
        records.append(
            evaluation_record(
                question,
                plan,
                hits,
                question_metrics,
                candidates=candidate_info,
                graph_only_hits=graph_only_hits,
                graph_only_metrics=graph_only_metrics,
            )
        )

    print("正解案件ありの平均:")
    summary: dict[str, object] = {"positive_questions": positive_count}
    if positive_count:
        averages: dict[str, dict[str, float]] = {}
        for k in ks:
            values = totals[k]
            averages[str(k)] = {
                "recall": values[0] / positive_count,
                "precision": values[1] / positive_count,
                "f1": values[2] / positive_count,
                "ndcg": values[3] / positive_count,
            }
            print(
                f"  @{k} Recall={values[0] / positive_count:.3f} "
                f"Precision={values[1] / positive_count:.3f} "
                f"F1={values[2] / positive_count:.3f} "
                f"nDCG={values[3] / positive_count:.3f}"
            )
        summary["metrics"] = averages
    if graph_only_count:
        graph_averages: dict[str, dict[str, float]] = {}
        print("Graph-only平均（global_summary除外）:")
        for k in ks:
            values = graph_only_totals[k]
            graph_averages[str(k)] = {
                "recall": values[0] / graph_only_count,
                "precision": values[1] / graph_only_count,
                "f1": values[2] / graph_only_count,
                "ndcg": values[3] / graph_only_count,
            }
            print(
                f"  @{k} Recall={values[0] / graph_only_count:.3f} "
                f"Precision={values[1] / graph_only_count:.3f} "
                f"F1={values[2] / graph_only_count:.3f} "
                f"nDCG={values[3] / graph_only_count:.3f}"
            )
        summary["graph_only"] = {
            "positive_questions": graph_only_count,
            "metrics": graph_averages,
        }
    if no_answer_count:
        summary["no_answer"] = {
            "correct": no_answer_correct,
            "total": no_answer_count,
            "accuracy": no_answer_correct / no_answer_count,
        }
        print(
            f"No-answer accuracy="
            f"{no_answer_correct / no_answer_count:.3f} "
            f"({no_answer_correct}/{no_answer_count})"
        )
    candidate_summary = summarize_candidates(records)
    summary["candidate"] = candidate_summary
    print(
        f"Candidate平均: Recall={candidate_summary['recall']:.3f} "
        f"Precision={candidate_summary['precision']:.3f}"
    )
    categories = summarize_by_category(records, ks)
    summary["categories"] = categories
    print_category_summary(categories)
    return {
        "system": "byog",
        "mode": "plan",
        "ranking_strategy": "graph_first_vector_tiebreak",
        "ks": list(ks),
        "questions": records,
        "summary": summary,
    }


def community_candidate_ids(query, communities, embedder, top_k: int) -> set[str]:
    texts = [f"{item.title}\n{item.summary}" for item in communities]
    community_vectors = embedder.embed(texts)
    query_vector = embedder.embed([query])[0]
    ranked = sorted(
        zip(communities, community_vectors),
        key=lambda pair: cosine_similarity(query_vector, pair[1]),
        reverse=True,
    )
    return {
        project_id
        for community, _vector in ranked[:top_k]
        for project_id in community.project_ids
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="人が作ったオントロジーから BYOG グラフを組む")
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--start", default="", help="周辺検索の起点ノードID。例: N_JOB_MLOPS")
    parser.add_argument("--eval", action="store_true", help="questions.csv で RAG と同じ採点をする")
    parser.add_argument(
        "--eval-split",
        choices=("development", "holdout", "all"),
        default="all",
        help="評価対象の区分。デフォルトは全件",
    )
    parser.add_argument("--summary-k", type=int, default=3, help="全体像で使うコミュニティ要約の件数")
    parser.add_argument(
        "--eval-mode",
        choices=("legacy", "plan", "both"),
        default="legacy",
        help="従来検索、検索計画+rerank、または両方を評価する",
    )
    parser.add_argument(
        "--reranker",
        choices=("local", "bedrock", "openai"),
        default="bedrock",
        help="plan評価のrerank。bedrock は Titan Embeddings",
    )
    parser.add_argument(
        "--ks", default="5,10,20", help="評価する k の一覧（カンマ区切り）"
    )
    parser.add_argument(
        "--plans",
        default="search_plans.draft.json",
        help="data-dir 配下の検索計画JSON",
    )
    parser.add_argument("--min-score", type=float, default=0.0)
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument(
        "--output", type=Path, default=None, help="計画評価を共通JSON形式で保存"
    )
    args = parser.parse_args()

    try:
        graph = load_byog_graph(args.data_dir)
    except (FileNotFoundError, ValueError) as exc:
        print(f"読み込みエラー: {exc}", file=sys.stderr)
        return 1

    if not graph.nodes:
        print("data/ のオントロジーCSVが空です。", file=sys.stderr)
        return 1

    print_stats(graph)
    for warning in validate(graph):
        print(f"警告: {warning}", file=sys.stderr)

    if args.start:
        try:
            print_query(graph, args.start, relations=TRAVERSE_RELATIONS)
        except KeyError as exc:
            print(f"検索エラー: {exc}", file=sys.stderr)
            return 1

    if args.eval:
        from graph_core import load_env

        load_env()
        try:
            questions = load_questions(args.data_dir)
        except (FileNotFoundError, ValueError) as exc:
            print(f"読み込みエラー: {exc}", file=sys.stderr)
            return 1
        if args.eval_split != "all":
            questions = [
                question
                for question in questions
                if question.evaluation_split == args.eval_split
            ]
        if not questions:
            print(
                f"questions.csv に評価区分={args.eval_split}の質問がありません。",
                file=sys.stderr,
            )
            return 1
        communities = load_communities(SUMMARY_DIR / "byog.csv")
        if communities:
            print(f"コミュニティ要約: {len(communities)} 件を全体像に使用")
        else:
            print("コミュニティ要約がありません。全体像もローカル探索です。先に scripts/summarize_communities.py --source byog")
        if args.eval_mode in {"legacy", "both"}:
            run_eval(graph, questions, communities, summary_k=args.summary_k)
        if args.eval_mode in {"plan", "both"}:
            try:
                plans = load_search_plans(args.data_dir / args.plans)
                embedder = build_embedder(args.reranker)
                if not args.no_cache:
                    embedder = CachedEmbedder(
                        embedder,
                        args.data_dir
                        / "cache"
                        / f"embeddings-{args.reranker}.json",
                    )
                reranker = ProjectReranker(graph.projects, embedder)
                result = run_plan_eval(
                    graph=graph,
                    questions=questions,
                    plans=plans,
                    searcher=ByogSearcher(graph, reranker),
                    ks=parse_ks(args.ks),
                    min_score=args.min_score,
                    communities=communities,
                    summary_k=args.summary_k,
                )
                if args.output:
                    args.output.parent.mkdir(parents=True, exist_ok=True)
                    args.output.write_text(
                        json.dumps(
                            {
                                "evaluation_split": args.eval_split,
                                "runs": [result],
                            },
                            ensure_ascii=False,
                            indent=2,
                        )
                        + "\n",
                        encoding="utf-8",
                    )
                    print(f"評価結果: {args.output}")
            except (ValueError, RuntimeError) as exc:
                print(f"計画評価エラー: {exc}", file=sys.stderr)
                return 1

    if not args.start and not args.eval:
        print("使い方: --start N_JOB_MLOPS  または  --eval")
        print("全体像まで揃えるなら先に: poetry run python scripts/summarize_communities.py --source byog")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
