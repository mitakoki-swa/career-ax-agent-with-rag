"""同じ案件・同じ質問を使うベクトルRAGの骨組み。

検索経路は常に「文書をベクトル化 → 質問をベクトル化 → コサイン類似度」。
今は API が無いので LocalHashEmbedder で同じ経路を通す。
キーを置いたら --embedder openai に切り替える。
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

sys.path.insert(0, str(Path(__file__).resolve().parent))

from graph_core import bedrock_runtime_client, f1_score, load_env, precision, recall
from retrieval_core import (
    CachedEmbedder,
    ProjectReranker,
    SearchHit,
    SearchPlan,
    filter_project_ids,
    load_search_plans,
    metrics_at_k,
)

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"

LATIN_TOKEN = re.compile(r"[a-z0-9][a-z0-9_+.#-]*", re.I)
CJK_CHAR = re.compile(r"[\u3040-\u30ff\u4e00-\u9fff]")


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


class Embedder(Protocol):
    name: str

    def embed(self, texts: list[str]) -> list[list[float]]:
        """テキストを同じ次元のベクトルにする。"""


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"CSV がありません: {path}")
    with path.open(encoding="utf-8-sig", newline="") as fh:
        return list(csv.DictReader(fh))


def load_projects(data_dir: Path = DATA_DIR) -> dict[str, Project]:
    path = data_dir / "projects.csv"
    projects: dict[str, Project] = {}
    for row in _read_csv(path):
        project_id = (row.get("案件ID") or "").strip()
        if not project_id:
            continue
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
    for row in _read_csv(path):
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


def tokenize(text: str) -> list[str]:
    lowered = text.lower()
    tokens = [token.lower() for token in LATIN_TOKEN.findall(lowered)]
    cjk = CJK_CHAR.findall(text)
    tokens.extend("".join(cjk[i : i + 2]) for i in range(len(cjk) - 1))
    return tokens


def _l2_normalize(vector: list[float]) -> list[float]:
    norm = math.sqrt(sum(value * value for value in vector))
    if norm == 0:
        return vector
    return [value / norm for value in vector]


class LocalHashEmbedder:
    """APIなしで embed() と同じ入出力を返す仮実装。精度比較の本番には使わない。"""

    name = "local-hash"
    model = "sha256-token-bigram-v1"
    dim = 256

    def embed(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for text in texts:
            vector = [0.0] * self.dim
            for token in tokenize(text):
                digest = hashlib.sha256(token.encode("utf-8")).digest()
                index = int.from_bytes(digest[:4], "big") % self.dim
                vector[index] += 1.0
            vectors.append(_l2_normalize(vector))
        return vectors


class BedrockEmbedder:
    """Bedrock の Titan 埋め込み。Claude は埋め込みできないので別モデルを使う。"""

    name = "bedrock"

    def __init__(self, model: str | None = None) -> None:
        self.model = model or os.environ.get("BEDROCK_EMBED_MODEL", "amazon.titan-embed-text-v2:0")

    def embed(self, texts: list[str]) -> list[list[float]]:
        client = bedrock_runtime_client()
        vectors: list[list[float]] = []
        for text in texts:
            response = client.invoke_model(
                modelId=self.model,
                contentType="application/json",
                accept="application/json",
                body=json.dumps({"inputText": text or " ", "normalize": True}),
            )
            payload = json.loads(response["body"].read())
            vectors.append(payload["embedding"])
        return vectors


class OpenAIEmbedder:
    """OPENAI_API_KEY を置いたら使う本番用。キーが無いとここで明示的に止める。"""

    name = "openai"

    def __init__(self, model: str = "text-embedding-3-small") -> None:
        self.model = model

    def embed(self, texts: list[str]) -> list[list[float]]:
        api_key = os.environ.get("OPENAI_API_KEY", "").strip()
        if not api_key:
            raise RuntimeError(
                "OPENAI_API_KEY がありません。キーを設定するか --embedder local を使ってください。"
            )
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError("openai パッケージがありません。pip install openai") from exc

        client = OpenAI(api_key=api_key)
        response = client.embeddings.create(model=self.model, input=texts)
        return [item.embedding for item in response.data]


def build_embedder(kind: str) -> Embedder:
    if kind == "local":
        return LocalHashEmbedder()
    if kind == "bedrock":
        return BedrockEmbedder()
    if kind == "openai":
        return OpenAIEmbedder()
    raise ValueError(f"未知の embedder です: {kind}")


class VectorIndex:
    """文書ベクトルを持ち、質問ベクトルとのコサインで案件IDを返す。"""

    def __init__(self, documents: dict[str, str], embedder: Embedder) -> None:
        self.embedder = embedder
        self.doc_ids = list(documents)
        texts = [documents[doc_id] for doc_id in self.doc_ids]
        self.doc_vectors = embedder.embed(texts)

    def search(self, query: str, k: int = 10) -> list[tuple[str, float]]:
        query_vector = self.embedder.embed([query])[0]
        scored: list[tuple[str, float]] = []
        for doc_id, doc_vector in zip(self.doc_ids, self.doc_vectors):
            score = _cosine(query_vector, doc_vector)
            if score > 0:
                scored.append((doc_id, score))
        scored.sort(key=lambda item: item[1], reverse=True)
        return scored[:k]


def _cosine(left: list[float], right: list[float]) -> float:
    if not left or not right or len(left) != len(right):
        return 0.0
    return sum(a * b for a, b in zip(left, right))


def print_hits(projects: dict[str, Project], hits: list[tuple[str, float]]) -> None:
    if not hits:
        print("ヒット案件: 0")
        return
    print(f"ヒット案件: {len(hits)}")
    for project_id, score in hits:
        project = projects.get(project_id)
        name = project.name if project else ""
        print(f"  - {project_id}  {score:.3f}  {name}")


def run_eval(index: VectorIndex, questions: list[Question], k: int) -> None:
    by_category: dict[str, list[tuple[float, float, float]]] = defaultdict(list)
    print(f"評価: {len(questions)} 問  top-{k}  （recall=取りこぼし / precision=ノイズ）")
    for question in questions:
        predicted = [project_id for project_id, _score in index.search(question.text, k=k)]
        rec = recall(question.gold_ids, predicted)
        prec = precision(question.gold_ids, predicted)
        f1 = f1_score(question.gold_ids, predicted)
        by_category[question.category or "未分類"].append((rec, prec, f1))
        gold = ",".join(question.gold_ids) if question.gold_ids else "該当なし"
        print(f"{question.id}  {question.category}  recall={rec:.2f}  precision={prec:.2f}  f1={f1:.2f}")
        print(f"  gold: {gold}")
        print(f"  rag:  {','.join(predicted) if predicted else '該当なし'}")
    print("分類別")
    for category, scores in by_category.items():
        n = len(scores)
        avg_rec = sum(item[0] for item in scores) / n
        avg_prec = sum(item[1] for item in scores) / n
        avg_f1 = sum(item[2] for item in scores) / n
        print(f"  {category}: recall={avg_rec:.2f}  precision={avg_prec:.2f}  f1={avg_f1:.2f}  ({n}問)")


def parse_ks(value: str) -> tuple[int, ...]:
    ks = tuple(sorted({int(part.strip()) for part in value.split(",") if part.strip()}))
    if not ks or any(k <= 0 for k in ks):
        raise ValueError("--ks は正の整数をカンマ区切りで指定してください")
    return ks


def plan_candidates(
    projects: dict[str, Project], plan: SearchPlan
) -> tuple[set[str], dict[str, tuple[str, ...]]]:
    """RAG は案件列で表現できる条件だけを決定的フィルタとして使う。"""

    applicable = tuple(
        type(condition)(field=condition.field, value=condition.value)
        for condition in plan.conditions
        if condition.value
        and condition.field
        in {"project_id", "role", "domain", "cloud", "skills", "text"}
    )
    if not applicable:
        return set(projects), {}
    rag_plan = SearchPlan(
        mode="filter",
        operator=plan.operator,
        conditions=applicable,
        query_terms=plan.query_terms,
        confidence=plan.confidence,
        status=plan.status,
        note=plan.note,
    )
    return filter_project_ids(projects, rag_plan)


def candidate_evaluation(
    gold_ids: tuple[str, ...], candidate_ids: set[str] | frozenset[str]
) -> dict[str, object]:
    ordered_ids = sorted(candidate_ids)
    gold = set(gold_ids)
    hits = len(gold & set(candidate_ids))
    return {
        "candidate_count": len(ordered_ids),
        "candidate_ids": ordered_ids,
        "candidate_recall": hits / len(gold) if gold else None,
        "candidate_precision": (
            hits / len(ordered_ids)
            if ordered_ids
            else (1.0 if not gold else 0.0)
        ),
        "candidate_no_answer_correct": not ordered_ids if not gold else None,
    }


def candidate_evaluation_text(info: dict[str, object]) -> str:
    recall_value = info["candidate_recall"]
    precision_value = info["candidate_precision"]
    recall_text = "N/A" if recall_value is None else f"{float(recall_value):.3f}"
    precision_text = (
        "N/A" if precision_value is None else f"{float(precision_value):.3f}"
    )
    return (
        f"{info['candidate_count']}件 "
        f"Recall={recall_text} Precision={precision_text}"
    )


def metrics_dict(metric) -> dict[str, float | int]:
    return {
        "k": metric.k,
        "recall": metric.recall,
        "precision": metric.precision,
        "f1": metric.f1,
        "ndcg": metric.ndcg,
    }


def summarize_candidates(
    records: list[dict[str, object]],
) -> dict[str, float | int]:
    positive = [record for record in records if record["gold_ids"]]
    if not positive:
        return {"positive_questions": 0, "recall": 0.0, "precision": 0.0}
    return {
        "positive_questions": len(positive),
        "recall": sum(
            float(record["candidates"]["candidate_recall"])
            for record in positive
        )
        / len(positive),
        "precision": sum(
            float(record["candidates"]["candidate_precision"])
            for record in positive
        )
        / len(positive),
    }


def summarize_by_category(
    records: list[dict[str, object]], ks: tuple[int, ...]
) -> dict[str, object]:
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for record in records:
        grouped[str(record["category"])].append(record)

    summaries: dict[str, object] = {}
    for category, items in sorted(grouped.items()):
        positive = [item for item in items if item["gold_ids"]]
        negative = [item for item in items if not item["gold_ids"]]
        category_summary: dict[str, object] = {
            "questions": len(items),
            "positive_questions": len(positive),
        }
        if positive:
            category_summary["candidate"] = {
                "recall": sum(
                    float(item["candidates"]["candidate_recall"])
                    for item in positive
                )
                / len(positive),
                "precision": sum(
                    float(item["candidates"]["candidate_precision"])
                    for item in positive
                )
                / len(positive),
            }
            category_summary["metrics"] = {
                str(k): {
                    metric_name: sum(
                        float(item["metrics"][str(k)][metric_name])
                        for item in positive
                    )
                    / len(positive)
                    for metric_name in ("recall", "precision", "f1", "ndcg")
                }
                for k in ks
            }
            graph_items = [
                item for item in positive if item.get("graph_only_metrics")
            ]
            if graph_items:
                category_summary["graph_only_questions"] = len(graph_items)
                category_summary["graph_only_metrics"] = {
                    str(k): {
                        metric_name: sum(
                            float(
                                item["graph_only_metrics"][str(k)][metric_name]
                            )
                            for item in graph_items
                        )
                        / len(graph_items)
                        for metric_name in ("recall", "precision", "f1", "ndcg")
                    }
                    for k in ks
                }
        if negative:
            correct = sum(bool(item["no_answer_correct"]) for item in negative)
            category_summary["no_answer"] = {
                "correct": correct,
                "total": len(negative),
                "accuracy": correct / len(negative),
            }
        summaries[category] = category_summary
    return summaries


def print_category_summary(categories: dict[str, object]) -> None:
    print("分類別:")
    for category, raw_summary in categories.items():
        summary = raw_summary
        print(f"  {category}:")
        if "candidate" in summary:
            candidate = summary["candidate"]
            print(
                f"    Candidate Recall={candidate['recall']:.3f} "
                f"Precision={candidate['precision']:.3f}"
            )
        for k, metric in summary.get("metrics", {}).items():
            print(
                f"    @{k} Recall={metric['recall']:.3f} "
                f"Precision={metric['precision']:.3f} "
                f"F1={metric['f1']:.3f} nDCG={metric['ndcg']:.3f}"
            )
        for k, metric in summary.get("graph_only_metrics", {}).items():
            print(
                f"    Graph-only @{k} Recall={metric['recall']:.3f} "
                f"Precision={metric['precision']:.3f} "
                f"F1={metric['f1']:.3f} nDCG={metric['ndcg']:.3f}"
            )
        if "no_answer" in summary:
            no_answer = summary["no_answer"]
            print(
                f"    No-answer accuracy={no_answer['accuracy']:.3f} "
                f"({no_answer['correct']}/{no_answer['total']})"
            )


def run_ranked_eval(
    reranker: ProjectReranker,
    questions: list[Question],
    projects: dict[str, Project],
    ks: tuple[int, ...],
    min_score: float,
    mode: str,
    plans: dict[str, SearchPlan],
) -> dict[str, object]:
    max_k = max(ks)
    totals = {k: [0.0, 0.0, 0.0, 0.0] for k in ks}
    positive_count = 0
    no_answer_count = 0
    no_answer_correct = 0
    records: list[dict[str, object]] = []
    print(f"\n--- RAG 評価 mode={mode} k={','.join(map(str, ks))} ---")

    for question in questions:
        plan = plans.get(question.plan_id)
        if mode == "plan" and plan is None:
            raise ValueError(f"{question.id} の検索計画がありません")
        if mode == "plan" and plan is not None:
            candidate_ids, matched = plan_candidates(projects, plan)
            query = plan.augmented_query(question.text)
        else:
            candidate_ids, matched = set(projects), {}
            query = question.text

        candidate_info = candidate_evaluation(question.gold_ids, candidate_ids)
        hits = reranker.rank(
            query=query,
            candidate_ids=candidate_ids,
            k=max_k,
            min_score=min_score,
            matched_conditions=matched,
        )
        predicted = [hit.project_id for hit in hits]
        print(
            f"{question.id} [{question.category}] "
            f"status={plan.status if plan and mode == 'plan' else '-'}"
        )
        print(f"  正解={list(question.gold_ids)}")
        print(f"  候補={candidate_evaluation_text(candidate_info)}")
        print(
            "  予測="
            + str([f"{hit.project_id}:{hit.score:.3f}" for hit in hits])
        )
        if not question.gold_ids:
            no_answer_count += 1
            no_answer_correct += int(not predicted)
            print(f"  No-answer={'正解' if not predicted else '不正解'}")
            records.append(
                evaluation_record(
                    question,
                    plan if mode == "plan" else None,
                    hits,
                    {},
                    not predicted,
                    candidates=candidate_info,
                )
            )
            continue
        positive_count += 1
        question_metrics: dict[str, dict[str, float | int]] = {}
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
            question_metrics[str(k)] = {
                "k": metric.k,
                "recall": metric.recall,
                "precision": metric.precision,
                "f1": metric.f1,
                "ndcg": metric.ndcg,
            }
        records.append(
            evaluation_record(
                question,
                plan if mode == "plan" else None,
                hits,
                question_metrics,
                candidates=candidate_info,
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
        "system": "rag",
        "mode": mode,
        "ks": list(ks),
        "questions": records,
        "summary": summary,
    }


def evaluation_record(
    question: Question,
    plan: SearchPlan | None,
    hits: list[SearchHit],
    metrics: dict[str, dict[str, float | int]],
    no_answer_correct: bool | None = None,
    *,
    candidates: dict[str, object] | None = None,
    graph_only_hits: list[SearchHit] | None = None,
    graph_only_metrics: dict[str, dict[str, float | int]] | None = None,
) -> dict[str, object]:
    return {
        "question_id": question.id,
        "category": question.category,
        "evaluation_split": question.evaluation_split,
        "plan_id": question.plan_id,
        "plan_mode": plan.mode if plan else "",
        "plan_status": plan.status if plan else "",
        "gold_ids": list(question.gold_ids),
        "candidates": candidates
        or candidate_evaluation(question.gold_ids, {hit.project_id for hit in hits}),
        "hits": [
            {
                "rank": rank,
                "project_id": hit.project_id,
                "score": hit.score,
                "vector_score": hit.vector_score,
                "graph_score": hit.graph_score,
                "matched_conditions": list(hit.matched_conditions),
                "reasons": list(hit.reasons),
                "path": list(hit.path),
            }
            for rank, hit in enumerate(hits, start=1)
        ],
        "graph_only_hits": [
            {
                "rank": rank,
                "project_id": hit.project_id,
                "score": hit.score,
                "graph_score": hit.graph_score,
                "matched_conditions": list(hit.matched_conditions),
                "reasons": list(hit.reasons),
                "path": list(hit.path),
            }
            for rank, hit in enumerate(graph_only_hits or [], start=1)
        ],
        "graph_only_metrics": graph_only_metrics or {},
        "metrics": metrics,
        "no_answer_correct": no_answer_correct,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="ベクトルRAGの骨組み")
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--query", default="", help="自由文で検索する")
    parser.add_argument("--eval", action="store_true", help="questions.csv で一括採点する")
    parser.add_argument(
        "--eval-split",
        choices=("development", "holdout", "all"),
        default="all",
        help="評価対象の区分。デフォルトは全件",
    )
    parser.add_argument("--k", type=int, default=10, help="返す案件数")
    parser.add_argument(
        "--eval-mode",
        choices=("natural", "plan", "both"),
        default="natural",
        help="自然文、検索計画、または両方を評価する",
    )
    parser.add_argument(
        "--ks", default="5,10,20", help="評価する k の一覧（カンマ区切り）"
    )
    parser.add_argument(
        "--plans",
        default="search_plans.draft.json",
        help="data-dir 配下の検索計画JSON",
    )
    parser.add_argument(
        "--min-score",
        type=float,
        default=0.0,
        help="この類似度未満の案件を返さない",
    )
    parser.add_argument(
        "--no-cache", action="store_true", help="埋め込みキャッシュを使わない"
    )
    parser.add_argument(
        "--output", type=Path, default=None, help="評価結果を共通JSON形式で保存"
    )
    parser.add_argument(
        "--embedder",
        choices=("local", "bedrock", "openai"),
        default="local",
        help="local は仮ベクトル。bedrock は BEDROCK_API_KEY。openai は OPENAI_API_KEY",
    )
    args = parser.parse_args()
    load_env()

    try:
        projects = load_projects(args.data_dir)
        embedder = build_embedder(args.embedder)
        if not args.no_cache:
            embedder = CachedEmbedder(
                embedder,
                args.data_dir / "cache" / f"embeddings-{args.embedder}.json",
            )
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        print(f"起動エラー: {exc}", file=sys.stderr)
        return 1

    if not projects:
        print("projects.csv が空です。Graph と同じ案件データを置いてください。", file=sys.stderr)
        return 1

    documents = {project.id: project.document() for project in projects.values()}
    try:
        index = VectorIndex(documents, embedder)
    except RuntimeError as exc:
        print(f"埋め込みエラー: {exc}", file=sys.stderr)
        return 1

    print(f"案件: {len(projects)}  embedder: {embedder.name}")

    if args.query:
        print(f"質問: {args.query}")
        hits = index.search(args.query, k=len(projects))
        hits = [hit for hit in hits if hit[1] >= args.min_score][: args.k]
        print_hits(projects, hits)

    if args.eval:
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
        try:
            plans = load_search_plans(args.data_dir / args.plans)
            ks = parse_ks(args.ks)
            reranker = ProjectReranker(projects, embedder)
            modes = (
                ("natural", "plan")
                if args.eval_mode == "both"
                else (args.eval_mode,)
            )
            runs = []
            for mode in modes:
                runs.append(
                    run_ranked_eval(
                        reranker=reranker,
                        questions=questions,
                        projects=projects,
                        ks=ks,
                        min_score=args.min_score,
                        mode=mode,
                        plans=plans,
                    )
                )
            if args.output:
                args.output.parent.mkdir(parents=True, exist_ok=True)
                args.output.write_text(
                    json.dumps(
                        {"evaluation_split": args.eval_split, "runs": runs},
                        ensure_ascii=False,
                        indent=2,
                    )
                    + "\n",
                    encoding="utf-8",
                )
                print(f"評価結果: {args.output}")
        except (ValueError, RuntimeError) as exc:
            print(f"評価エラー: {exc}", file=sys.stderr)
            return 1

    if not args.query and not args.eval:
        print("使い方: --query \"想定職種がMLOpsの案件を教えて\"  または  --eval")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
