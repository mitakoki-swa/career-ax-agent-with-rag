# PoC：ベクトルRAG / BYOG GraphRAG / LLM GraphRAG の比較

ステータス：PoC Aの比較基盤実装済み / 検索計画ドラフトのレビュー待ち  
対象ブランチ：`feature/graph-rag-poc`  
関連：

- `docs/vision.md` Step1
- `docs/adr/0002-作るもののイメージ固め.md`
- `docs/adr/0003-検索PoCの評価方針.md`
- `docs/poc/検索方式とCLI引数.md`
- `docs/poc/GraphRAG-PoC最終比較.md`

この文書は、案件検索基盤のPoCで「何を検証し、何をもって判断し、どこまで作るか」を記録する実験の正本である。最終的な採用方式は、比較条件を揃えた実験結果を基に別途ADRで決定する。

## 1. PoCの目的

案件情報を横断的に検索し、職種名の完全一致だけでは見つからない隣接案件を提示することで、エンジニアがキャリアの選択肢を広げられるかを検証する。

主な仮説は次のとおり。

1. 事実引きはベクトルRAGまたは構造化検索で十分である
2. 「インフラ経験をMLOpsにつなげたい」などの周辺・キャリアブリッジ検索では、業務の関係を持つGraphが有効である
3. 人が枠組みを定義するBYOGは、LLMが自由にGraphを生成する方式より精度・再現性・コストを制御しやすい
4. 本番候補は、Graphで候補を取得し、Gを優先して同じGの中をTitanで順位付けするHybrid構成になる可能性が高い

ADR-002のとおり、検証対象は「どういう案件が存在するか」という事実の開示である。参画可能性、ジュニア可否、現在の空き、個人とのマッチ度は対象外とする。

## 2. 現在のデータと実装

### 2.1 共通データ

```text
data/projects.csv          合成案件28件
data/questions.csv         開発20問・holdout 17問と正答案件ID
data/search_plans.draft.json
                          37問の手動検索計画ドラフト
data/nodes.csv             BYOGの人手オントロジー
data/edges.csv
data/tags.csv
data/llm_generated/*.csv   Claudeが案件文から生成したGraph
data/community_summaries/  BYOG / LLM Graphのコミュニティ要約
```

Q01〜Q20は方式調整と回帰確認に使う `development`、Q21〜Q37は構成固定後の `holdout` とし、`--eval-split` で集計を分離する。

| 分類 | 問 | 評価したいこと |
|---|---|---|
| 事実引き | Q01〜Q04 | 職種・スキル・案件IDなどの明確な事実 |
| 横断 | Q05〜Q08 | 複数条件のAND |
| 周辺 | Q09〜Q12 | 職種列に書かれていない隣接業務を辿れるか |
| 全体像 | Q13〜Q16 | 案件群の傾向をコミュニティから説明できるか |
| 該当なし | Q17〜Q20 | 無関係な案件を返さず、存在しないと言えるか |

本命はQ09「MLOpsとして専門性を上げるために経験できる案件は？」である。学習・配信・監視・特徴量に関わるインフラ、データエンジニア、ソフトウェアエンジニア案件も拾い、単なるKubernetes運用やDWH案件は除外する。

### 2.2 ベクトルRAG

`scripts/search_rag.py`

```text
案件文をTitanで埋め込み
→ 質問を埋め込み
→ コサイン類似度で順位付け
→ 上位k件
```

案件文は `案件名 + 職種 + ドメイン + クラウド + 概要 + 必要スキル` である。自然文だけの評価と、検索計画で属性候補を絞ってから順位付けする評価を切り替えられる。案件・質問の埋め込みは `data/cache/` へ保存し、同じモデル・同じ入力のAPI呼び出しを再利用する。`--min-score` で「該当なし」用の類似度しきい値も指定できるが、採用値は負例を使って別途決める。

### 2.3 BYOGローカル検索

`scripts/build_graph_byog.py`

```text
質問に含まれるノード名を文字列一致
→ 「包含」「関連」を辿る
→ 到達ノードのタグから案件IDを収集
```

- Graphのノード60件、エッジ60件、案件タグ188件
- `使用`は辿らない。共通スキルだけで無関係な案件へ広がることを防ぐため
- Q09では正解23件をすべて取得し、P005 / P007 / P009 / P012 / P025を除外できた
- 従来ローカル検索は互換性確認用として残している
- 新しい計画実行器は `exact / filter / graph_expand / graph_bridge / global_summary` を許可リストで検証し、条件を決定的に実行する
- `filter` は明示的なAND / ORを扱い、Graph検索では候補集合を条件で絞る
- BYOG候補だけのTitan類似度を計算し、Gを優先して同じGの中をVで並べる。`SearchHit` にvector score、graph score、一致条件、理由、Graph経路を保持する

### 2.4 BYOGグローバル検索

```text
BYOGをLouvainでコミュニティ分割
→ Claudeで事前要約
→ 質問と要約をTitanで比較
→ 上位kコミュニティの案件を返す
```

ローカル検索とグローバル検索は同じBYOGを使うが、質問時の処理は別である。グローバル検索がローカル検索を途中まで実行するわけではない。

現在はコミュニティ検出から `使用` を除外するため、スキルノードが孤立し、BYOGでは48件の要約が生成された。意味のあるコミュニティより細かく分割されており、コスト比較を歪めている。

検索計画の `global_summary` 経路では、案件rerankと同じ埋め込み・キャッシュを使って要約を順位付けする。全体像に分類したQ13-Q16・Q32-Q34はこの経路へ統一する。従来評価経路は互換性確認用であり、キャッシュなしのため最終比較には使わない。

### 2.5 LLM Graph

`scripts/build_graph_llm.py`

```text
案件28件をClaudeへ1件ずつ渡す
→ ノード・エッジを自由生成
→ Graphをキャッシュ
→ relation表記を包含・関連・使用へ正規化
→ 評価用SearchPlanの正規名をTitanで複数のLLMノードへ変換
→ LLM Graph用の探索深さで候補生成
→ BYOGと同じmode実行・Graph-first＋Titan同点解消・共通指標で評価
```

生成結果はノード284件、エッジ557件、タグ455件、コミュニティ11件である。Graphが密につながり、一般的なノードからほぼ全案件へ到達する。

「LLM Graph」はGraphを作ったのがLLMという意味であり、ローカル検索時にLLMが推論しているわけではない。検索時はキャッシュしたCSVをPythonが決定的に走査する。Claudeが呼ばれるのはGraph再構築とコミュニティ要約作成時である。

holdout比較ではRAG / BYOGと同じ `k=5/10/20`、Candidate Recall / Precision、Recall / Precision / F1 / nDCG、No-answer accuracy、分類別平均、Titanを使う。Plan変換は評価用の意味合わせであり、1つの概念を語彙一致とTitanで複数ノードへ解決する。relationの表記ゆれを正規化し、細粒度Graph用にhop数を増やす。各設定と解決状況を出力JSONへ残し、developmentで固定してからholdoutを比較する。

### 2.6 GraphのHTML可視化

`scripts/visualize_graph.py` は、BYOGとLLM生成Graphを別々の自己完結HTMLへ変換する。外部JavaScriptやCDNを使わないため、生成ファイルをそのままブラウザで開ける。

```powershell
# BYOGとLLM生成Graphを両方生成
poetry run python scripts/visualize_graph.py --source both

# 片方だけ生成
poetry run python scripts/visualize_graph.py --source byog
poetry run python scripts/visualize_graph.py --source llm
```

出力先：

```text
artifacts/graph_visualizations/byog.html
artifacts/graph_visualizations/llm.html
```

画面では、ノード名・ID・案件IDの検索、ノード種別と関係種別の絞り込み、案件タグ線の表示、ズーム、移動、隣接ノード限定表示ができる。ノードを選ぶと説明、接続案件、隣接ノードを確認できる。LLM版はノードとエッジが多いため、最初に案件ノードを非表示にし、必要な種別へ絞る。

## 3. 現時点で分かったこと

### 3.1 BYOGの効果

- Q09では Recall 1.00 / Precision 1.00となり、オントロジーによる包含・除外が機能した
- ベクトルRAGを全28件まで広げるとQ09のRecallは1.00になるが、BYOGが除外した5案件も返す
- LLM GraphもQ09では全28件を返し、同じ5案件を除外できない
- BYOGは「正解を増やす」だけでなく、「共通スキルを持つが目的の業務ではない案件を落とす」点で価値がある

### 3.2 現在の高Recallをそのまま評価できない理由

- RAGは類似度順の上位k件、Graphは到達した全件であり、返却件数が揃っていない
- Q09の正解は23件なので、RAGのRecall@10は最大でも `10 / 23 = 0.43`
- Graphにはランキングがなく、単純に上位10件へ切ると案件ID順になる
- 横断検索はANDではなく案件集合の和になっている
- 合成案件・質問・正解・オントロジーが同じ開発文脈で作られており、本番データより表現が揃っている

したがって、現状はBYOGの可能性を確認した段階であり、GraphRAGがRAGより総合的に高精度だとはまだ結論付けない。

### 3.3 コミュニティ要約

全体像質問で参照する要約数を増やすとRecallが上がりやすく、Precisionは下がりやすい。

観測例：

- BYOG local：Recall 0.75 / Precision 0.33
- BYOG 要約上位3件：Recall 0.83 / Precision 0.40
- LLM Graph local：Recall 0.75 / Precision 0.27
- LLM Graph 要約上位4件：Recall 0.88 / Precision 0.32

LLM GraphのRecall上昇は、大きなコミュニティを追加したことで多くの案件をまとめて取得した影響が大きい。コミュニティの分割品質が改善したとは限らない。

### 3.4 コスト

現状のAPI呼び出し構造は次のとおり。

- BYOG local：API不要
- RAG：Titanで案件と質問を埋め込み
- LLM Graph再構築：Claudeを案件ごとに28回
- LLM Graphコミュニティ要約：Claudeを11回
- BYOGコミュニティ要約：Claudeを48回
- グローバル評価：Titanで要約と質問を埋め込み

現在のままでは、BYOGが必ず低コストとは証明できない。BYOGのスキル単体コミュニティを除外し、案件・要約ベクトルをキャッシュし、APIのcalls / input tokens / output tokens / latencyを記録する必要がある。

## 4. これから実施するPoC

検索器、質問解析、案件タグ付けを分離して評価する。最初からすべてをLLMで自動化すると、失敗原因がGraph・質問理解・タグのどこにあるか判別できないためである。

### PoC A：理想的な検索計画で検索器を評価

自然な質問文とは別に、人が正解の検索計画を定義する。

20問の初期計画は `data/search_plans.draft.json` に作成済みで、`questions.csv` の `検索計画` 列から計画IDを参照する。これは正解ラベルの一部なので、評価結果を見る前に人がレビューし、`status` を `approved` にした版を固定する。ドラフトを調整して得点を上げ続けると評価データへの過学習になるため、調整履歴を残す。

例：

```json
{
  "mode": "filter",
  "conditions": [
    {"field": "skills", "value": "Python", "node_id": "N_SKILL_PYTHON"},
    {"field": "domain", "value": "金融"}
  ],
  "operator": "AND",
  "status": "approved"
}
```

検索モードは少なくとも次を扱う。

| mode | 用途 |
|---|---|
| `exact` | 案件ID、職種、スキルなどの直接検索 |
| `filter` | 複数属性のAND / OR |
| `graph_expand` | MLOps周辺の関連業務を広く取得 |
| `graph_bridge` | インフラからMLOpsなど、起点と目標をつなぐ |
| `global_summary` | 案件群の傾向 |

実行経路：

```text
人が作った検索計画
→ BYOGで候補案件を取得
→ Titanで候補案件をrerank
→ 上位k件を評価
```

この段階では質問解析AIを使わない。正しい検索計画があればBYOGが高精度になるかを検証する。

実行例：

```powershell
# RAG：自然文と検索計画の両方を同じkで評価
poetry run python scripts/search_rag.py --eval --eval-mode both --embedder bedrock --ks 5,10,20 --output data/eval_results/rag.json

# BYOG：計画で候補生成し、Titanでrerank
poetry run python scripts/build_graph_byog.py --eval --eval-mode plan --reranker bedrock --ks 5,10,20 --output data/eval_results/byog.json

# APIを使わず、処理経路だけ確認
poetry run python scripts/search_rag.py --eval --eval-mode both --embedder local --ks 5,10,20
poetry run python scripts/build_graph_byog.py --eval --eval-mode plan --reranker local --ks 5,10,20
```

`data/cache/` はGit管理外である。Titanモデルまたは案件文を変更するとキャッシュキーが変わり、必要なテキストだけ再埋め込みされる。

`--output` のJSONはRAG / BYOGで同じ `question_id / gold_ids / hits / metrics` を持つ。各hitには `score / vector_score / graph_score / matched_conditions / reasons / path` を保存するため、順位だけでなくBYOGの候補理由も比較できる。生成結果は `data/eval_results/` に置き、Git管理外とする。

### PoC B：質問から検索計画を生成できるか

PoC Aで検索器と検索計画の形式を固定した後、新しい自然な質問をLLMへ渡し、制約付きJSONとして検索計画を生成させる。

評価項目：

- 検索モードの正解率
- AND / ORの正解率
- 条件、起点ノード、対象関係の抽出精度
- BYOGに存在しないノードを勝手に生成していないか
- 人手計画とLLM計画での検索精度差

この段階の質問は、既存20問とは別のhold-out質問を使う。可能であれば対象エンジニアが自然に書いた質問を使い、AI生成質問だけを最終評価には使わない。

### PoC C：新規案件を固定オントロジーへタグ付けできるか

本番では職種・スキル・案件が増えるため、BYOGのタグを人手だけで維持しない。

```text
新規案件
→ 固定オントロジーをClaudeへ提示
→ 既存ノードへのタグ候補とconfidenceを出力
→ 未知概念だけ人が確認
```

オントロジーは、個別の職種名より次の上位構造を安定させる。

- 職種
- 業務・能力
- スキル
- ドメイン
- クラウド

新しい職種は既存の業務・能力へ分解し、対応できない名称だけ新規ノードまたはalias候補としてレビューする。

評価項目：

- 人手タグに対するPrecision / Recall / F1
- 未知概念を正しく検出できた割合
- 人手レビューが必要な案件割合
- 1案件あたりのAPI費用とレビュー時間

### PoC D：実データと定性的な体験

合成データで実装を固めた後、匿名化した実案件へ置き換える。

- 対象エンジニアが作成した質問で検索する
- RAG / BYOG / Hybridを同じ画面または出力で比較する
- 「知らなかったが妥当な隣接案件を発見できたか」を確認する
- Graphの経路を「なぜこの案件が出たか」として表示する
- 「参画できる」という判断ではなく、存在する案件の発見であることを明示する

## 5. 比較方法

### 5.1 候補取得と画面提示を分ける

```text
候補取得：正解案件を広く落とさず取得する
画面提示：候補を順位付けし、上位だけ表示する
```

BYOGはGraphで候補を取得し、Graph関連度Gを優先して同じGの中をTitan類似度Vで並べる。RAGは全案件をTitanで順位付けする。

```text
RAG   ：全案件 → Titan順位付け
BYOG  ：Graph候補 → G降順 → 同じGをTitanで順位付け
全体像：コミュニティ候補 → Titan順位付け（案件ごとのGなし）
```

PoC Aでは、RAGの候補を先にk件へ絞ってからGraphを検索する方式は採用しない。RAGが落とした隣接案件をGraphが救えなくなるためである。

### 5.2 kを揃える

案件を関連度順に並べた後、各方式を同じ `k=5 / 10 / 20` で評価する。

- `Precision@k`：上位k件のうち正解だった割合
- `Recall@k`：全正解のうち上位k件で取得できた割合
- `nDCG@k`：正解案件が上位に並んでいる度合い
- `F1@k`：PrecisionとRecallのバランス

Q09のように正解数がkより多い質問では、Recall@10だけで方式を判断しない。候補全体のRecallと、上位k件のPrecision / nDCGを併記する。

正答案件がないQ17〜Q20は、Precision / Recallの平均へ混ぜず `No-answer accuracy` として別集計する。該当なし質問を通常質問の平均へ混ぜると、空集合を返すだけで平均値が過大になるためである。

### 5.3 同じ条件にするもの

- 同じ案件データ
- 同じ自然な質問
- 同じ正答案件ID
- 同じ案件文
- 同じTitan埋め込みモデル
- 同じk
- 同じ評価コード
- 同じ該当なし判定方針
- キャッシュ有無とAPI費用の記録方法

### 5.4 暫定成功基準

実データ取得後に再調整する前提で、次をPoCの判断材料とする。

- Graphが必要なQ09〜Q12相当で、BYOGまたはHybridが同じkのRAGよりRecall@10を0.15以上改善する、またはnDCG@10を0.10以上改善する
- Precision@10がRAG比で0.05を超えて悪化しない
- Q09相当で、同じスキルを持つだけの無関係案件を除外できる
- 該当なし質問で無関係案件を返さない
- 検索理由となるGraph経路を説明できる
- 制約付きAIタグ付けがhold-out案件でF1 0.85以上を目安とする
- API費用と人手レビュー工数を1000案件規模へ外挿できる

この数値は業界標準ではなく、PoCの意思決定用の暫定値である。

## 6. 方式ごとの判断

| 結果 | 判断 |
|---|---|
| 周辺検索でBYOG / HybridがRAGを明確に上回る | Graphによるキャリアブリッジ探索を採用候補とする |
| 事実引きはRAG、周辺はGraphが強い | 質問モードを分けるHybridを採用候補とする |
| 同じkで差がほぼない | RAGを優先し、Graphの維持コストを負わない |
| GraphもRAGも実データで低精度 | 検索方式よりデータ品質・情報不足を先に改善する |
| 人手計画では高精度、LLM計画で大きく低下 | 質問解析の改善または確認質問を導入する |
| AIタグ付け精度が低い / レビュー率が高い | BYOGの本番自動運用は見送り、オントロジー縮小を検討する |
| BYOGがLLM Graph以上で、構築・更新コストも低い | 固定オントロジー＋制約付きAIタグ付けを本番候補とする |

LLMが自由生成するGraphは、RAG / BYOG / Hybridの評価軸と検索結果形式が固まるまで再調整を保留する。比較時は同じ `案件ID / score / rank / reason` を返すよう改修する。

## 7. PoCでやらないこと

- 案件と個人のマッチングスコア
- ジュニアが参画可能か、現在参画可能かの判断
- キャリア実現確率、スキルギャップ、学習計画の生成
- Slack / Gmail / 音声データの本格ETL
- 本番用Graph DB、認証、権限管理
- 完全なMicrosoft GraphRAGの階層・map-reduce再実装
- 本番UI、ストリーミング回答、運用監視基盤一式
- LLM Graphの再構築ガチャによる精度改善
- GraphRAGがすべての質問分類でRAGより勝つという結論ありきの調整

## 8. 実装状況と課題

優先順は次のとおり。

実装済み：

1. 質問とは別の検索計画JSONと20問のドラフト
2. `exact / filter / graph_expand / graph_bridge / global_summary` の計画検証・決定的実行
3. 横断質問のAND / ORと案件列条件
4. BYOG候補のTitan rerank
5. 共通 `SearchHit` と `Precision@k / Recall@k / F1@k / nDCG@k`
6. テキスト単位の埋め込みキャッシュ
7. RAGの `--min-score`
8. Graph経路、一致条件、検索理由の保持
9. RAGの自然文評価と検索計画評価の切り替え
10. rerank前の候補件数・候補ID・Candidate Recall / Precision
11. 各hitの明示Rankと質問分類別の平均
12. BYOG候補をGだけで並べるGraph-only比較順位

未完了：

1. 検索計画ドラフトの人手レビューと承認版の固定
2. BYOGのコミュニティ検出からスキル単体要約を除外
3. API calls / tokens / latencyの計測
4. しきい値の学習用質問と最終評価質問の分離
5. PoC A完了後の質問解析LLM、案件タグ付けLLM
6. 最後にLLM Graphを同じ評価形式へ合わせる

## 9. データ上の注意

現在の28案件と20質問は、検索処理・Graph構造・評価方法を検証する開発用データである。

同じAIまたは同じ開発者が案件・質問・正解・オントロジーを作ると、表現や判断基準が揃いすぎる可能性がある。案件数をAI生成で増やすだけでは、本番への一般化を証明できない。

最終テストでは次を分離する。

```text
開発用：
  現在の合成案件・質問

最終評価用：
  未使用の実案件または人が作成した案件
  対象エンジニアが自然に書いた質問
  別の評価者が作った正答案件IDと検索計画
```

オントロジーと検索パラメータは最終評価前に固定し、結果を見てから変更しない。

## 10. 次のアクション

1. `data/search_plans.draft.json` を質問・正解と照合し、承認版として固定する
2. ローカル埋め込みで決定的候補集合と評価処理を回帰テストする
3. Graph-first＋Titan同点解消でBYOGを再評価し、Titan単独順位・Graph-only順位と比較する
4. 現在の4職種・業務オントロジー内のholdout質問を評価し、範囲外職種はNo-answer/OODとして別集計する
5. 負例用の開発セットで `--min-score` を決め、最終評価では固定する
6. 結果とAPI費用をこの文書へ記録する
7. PoC Aが成功した場合のみ、PoC Bの質問解析を実装する
8. 未使用案件でPoC Cの制約付きAIタグ付けを評価する
9. 実案件と対象ユーザーの質問を取得し、定性デモを行う

## 11. 今の到達点

合成28案件・開発20問のTitan比較では、BYOGの候補生成がRAG planより高いCandidate Precisionを示した一方、Vだけのrerankが周辺検索のGraph順位を悪化させた。この結果を受け、BYOGの通常順位をG優先・同じGの中をTitanで並べる方式へ変更した。holdout 17問と評価区分の分離は追加済みである。検索計画の人手承認、holdoutの初回評価、質問解析、タグ付け自動化、実データ評価、コスト計測は未完了である。

次はLLM Graphの再生成ではなく、RAGとBYOGの評価条件を揃え、正しい検索計画が与えられた場合の検索性能を確定する。
