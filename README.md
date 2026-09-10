# Hybrid Search Router Dataset

Build a dataset that captures when sparse, dense, or hybrid retrieval works best—and use it to train and evaluate a query router.

This repository brings together query profiling, dataset composition, verified augmentation, retrieval-based labeling, and router experiments for Qdrant. The central artifact is the dataset: queries connected to a corpus, relevance judgments, and measured retrieval outcomes. The classifiers test whether those outcomes can be predicted from query text alone.

The work follows four stages: **understand queries → select and generate data → train and measure routers → test on an unseen collection**.

## Start with these notebooks

These three notebooks live directly in `src/` and are the main entry points into the deployed datasets and router. Supporting experiments live under `notebooks/`:

| Notebook | What it covers |
| --- | --- |
| [Home Depot top-1 bakeoff](src/home_depot_top1_bakeoff.ipynb) | How the router is evaluated: retrieval quality on Home Depot and comparisons against alternative strategies. |
| [Encoder router volume probe](src/encoder_router_volume_probe.ipynb) | How the router is trained: constructing the combined training population and comparing model variants. |
| [230K union deep dive](src/union_230k_deepdive.ipynb) | Statistics and detailed analysis of the deployed datasets, including composition and query-feature coverage. |

## Why route hybrid search?

Sparse search rewards lexical overlap. Dense search can retrieve semantically related documents even when the wording differs. Reciprocal Rank Fusion (RRF) combines their rankings, but it does not know whether both retrievers are useful for a particular query. When one ranking is misleading, blending it into the other can make the results worse.

The September 2026 project presentation illustrates this with `wood garage door`: sparse retrieval scores 0.631 NDCG@10, while fixed hybrid RRF scores 0.061 on the example collection. That motivates a routing decision, but does not establish a universal rule about short queries or product searches. Which strategy works depends on the query, the documents, and the retrieval stack.

The router chooses among three routes:

| Route | Retrieval strategy |
| --- | --- |
| `sparse_only` | Sparse retrieval with BM25 |
| `dense_only` | Dense vector retrieval |
| `pure_rrf` | Hybrid retrieval with Reciprocal Rank Fusion |

The research question is whether a cheap, query-only classifier can make that choice reliably enough to improve on a fixed strategy—and on an LLM-based selector.

## 1. Understand queries: a shared taxonomy

[Query Taxonomy](src/query-taxonomy/README.md) provides the measurable vocabulary used throughout the pipeline. It detects structured identifiers, sentence markers, logical structures, corruption, language, and statistical properties such as query length, natural-language share, and syntactic depth. Corpus-relative features add measurements such as term rarity and vocabulary overlap against a collection.

A **bank** measures a feature and returns either character spans or scalar statistics. Span banks resolve overlapping claims within their group; different groups can describe the same text independently. This lets a query contain an identifier, negation, and a typo without collapsing those properties into one query category.

The same features serve several purposes: profile source datasets, identify missing query behaviors, select diverse rows, verify generated queries, and diagnose model behavior. **A query's shape is never its retrieval label.** An identifier may suggest that exact matching matters; only retrieval against relevance judgments establishes which route succeeded.

## 2. Build the dataset

### Acquire queries, judgments, and documents

A **lane** is a dataset-specific retrieval and evaluation unit: queries, relevance judgments (*qrels*), and a corpus indexed in Qdrant with dense and sparse representations. The presentation reports 46 indexed lanes acquired across three waves, including ORCAS, MS MARCO, BEIR, BRIGHT, QUEST, CRUMB, RAR-b, FreshStack, LoTTE, CLERC, Amazon ESCI, WANDS, FINDER, TREC CAST, and ToCQA.

Acquisition includes adapting formats, materializing corpora, and establishing what relevance evidence each source actually provides. Official qrels, click-derived signals, answer passages, and generated judgments have different strengths; their provenance matters to the resulting labels. See [the dataset guide](docs/datasets.md).

### Compose for coverage

`composition/` selects queries using **archetype cells**: named combinations of taxonomy features, such as a short identifier lookup or a long conversational request. Recipes control quotas and source contributions so a large corpus cannot silently define the entire dataset.

Cells describe the query behaviors we want to cover. A cell with too few matching queries becomes a generation target. Each row retains its source and the cells it satisfies, making coverage and source imbalance inspectable.

### Generate, verify, and judge

The augmentation loop turns coverage deficits into concrete orders:

1. Measure which cells are underfilled.
2. Select a parent query or a supporting document.
3. Generate or transform a query to meet the requested features.
4. Re-run the taxonomy and reject candidates that fail their targets.
5. Establish relevance support and measure retrieval outcomes.
6. Return accepted rows to the pool and recompose the dataset.

Passing a taxonomy check establishes the requested query shape. Answerability and retrieval quality need separate evidence. When a transformation changes what a query asks for, its original qrels cannot simply be assumed to remain valid.

The [pipeline service](src/pipeline_service/README.md) exposes coverage analysis, generation instructions, verification, augmentation, and qrels expansion over HTTP. Model-backed generation is orchestrated by callers; the service supplies prompts and checks. Its qrels-expansion endpoint can call a paid judge under an explicit spend ceiling.

### Label from retrieval outcomes

The labeling harness runs all three routes against the same corpus and scores them against the query's qrels. The main objective is:

```text
O(route, query) = 0.7 × HitRate@1 + 0.3 × NDCG@10
```

It emphasizes finding a relevant document first while retaining a measure of top-ten ranking quality. Stored rankings and scores support re-evaluation, tie analysis, and comparisons between serving policies.

The later dataset builds use a cascade that adds judgment and retrieval evidence to recover more decisive rows. **Unresolved rows stay unresolved.** A tie-breaking serving policy is not evidence that one retriever is better.

### Dataset snapshots

The September 2026 presentation and volume-probe notebook distinguish two populations:

| Snapshot | Rows | Rows with a stored route | Role |
| --- | ---: | ---: | --- |
| 91K final version | 91,080 | 56,047 (61.5%) | Cascaded and relabeled artifact from the 100K-v2 build |
| 233K combined union | 233,247 | 166,188 (71.2%) | Union of v2, v3, v3-augmented, and 100K-v2, deduplicated by `(dataset, query_id)` |

Both cover 46 lanes. The union contains the 91K population and is not an independent control. Its composition also changes substantially: approximately 34.1% natural, 57.1% synthetic, and 8.6% augmented queries, compared with 81.7%, 16.8%, and 1.6% in the 91K version. More rows therefore do not isolate the effect of dataset size.

The [volume-probe notebook](src/encoder_router_volume_probe.ipynb) constructs the union at runtime. The older 46,142-query, 42-collection dataset belongs to the earlier viability study.

## 3. Train and measure routers

The repository contains the original logistic-regression baseline, encoder-router experiments, feature and branch ablations, and a model-free word-rarity rule.

The default serving arm, `zipf_shape_nocorpus`, uses a BGE query embedding, character n-gram features, and word-frequency/query-shape statistics with an MLP. It requires query text at inference time and makes no LLM call. Two heads estimate whether sparse and dense retrieval are acceptable; thresholds convert those scores into a route, with an RRF hedge when the scores are close.

The [router service](src/router_service/api.py) exposes route predictions and the `p_dense` and `p_sparse` scores. These are separate model outputs, not a three-class probability distribution. Its health endpoint identifies the loaded model and serving configuration.

Evaluation asks more than whether a classifier matches stored labels: does its chosen route improve retrieval over a train-selected constant, fixed RRF, a rarity rule, or the LLM selector? Does the result survive holding out a collection? Do hand-authored probes reveal failures hidden by an aggregate score?

## 4. Test on an unseen collection

The Home Depot bakeoff evaluates the routers on 11,795 real queries and roughly 124K products. The September 2026 presentation reports these relevance-at-one results:

| Strategy | Relevance@1 |
| --- | ---: |
| Deployed encoder router, 233K training pipeline | 0.585 |
| Fixed hybrid RRF | 0.552 |
| Auto-fusion LLM, 3,000-query run | 0.550 |
| Weighted RRF, router probabilities | 0.544 |
| Word-rarity rule | 0.540 |
| Dense-only constant | 0.371 |

The presentation's paired per-query comparisons report gains of **0.033 over fixed RRF** and **0.049 over auto-fusion**. These use matched queries, so the paired auto-fusion gain differs from subtracting the aggregate rows above. The router was statistically indistinguishable from the two strongest small-model contenders in that bakeoff.

The same presentation reports about **15 ms for the MLP routing step**, and **34 ms end to end**, versus 379 ms for the LLM-based route. Its roughly 10× cost reduction uses assumed rates; timing and cost describe that measured setup.

These results support the later encoder router on this collection. The earlier logistic-regression router received a scoped **NO-GO** against its precommitted evaluation bar; [VERDICT.md](VERDICT.md) preserves that result and its limitations. Neither result establishes that routing wins on every collection.

The [collection-to-search-tests demo](src/demo_service/README.md) closes the loop for a collection without an answer key: select a document, request a query behavior, generate and verify a query, compare retrieval routes, and retain an inspectable test with provenance.

## Get started

Use Python 3.11–3.14 and Poetry. From the repository root:

```bash
git submodule update --init --recursive
poetry install
```

Query Taxonomy is a local submodule dependency. To fetch versioned data and trained model artifacts, install DVC with its GCS support separately—DVC is not a project dependency—and run:

```bash
pipx install 'dvc[gs]'
dvc pull -r public src/data.dvc models.dvc
```

The configured public remote uses anonymous GCS access. Data and model artifacts are separate from the Git checkout; source datasets also retain their own access and licensing requirements.

### Run the router

After fetching model artifacts:

```bash
poetry run uvicorn router_service.api:app --host 127.0.0.1 --port 8001
```

Open `http://127.0.0.1:8001/docs`, or classify a batch:

```bash
curl -s http://127.0.0.1:8001/classify \
  -H 'Content-Type: application/json' \
  -d '{"queries": ["wood garage door", "why do cats purr", "CVE-2021-44228"]}'
```

`ROUTER_DIR` selects the saved model directory. `RRF_DELTA` controls the near-tie hedge and defaults to `0.15`. The default arm lives under `models/classifiers_union_200k/zipf_shape_nocorpus/`; the directory name is historical. The sentence encoder may need to download its weights on first use.

### Explore or build data

```bash
poetry run uvicorn pipeline_service.api:app --host 127.0.0.1 --port 8000
```

Open `http://127.0.0.1:8000/docs` and follow the [pipeline walkthrough](src/pipeline_service/README.md). For retrieval and indexing work, start local Qdrant with `docker compose up -d` and configure the relevant variables listed in [.example.env](.example.env). Full taxonomy extraction also needs the language/model dependencies described in the [taxonomy README](src/query-taxonomy/README.md).

Data artifacts live under `src/data/`. Notebook path setup varies; read each notebook's setup cell before running it. Generation, judging, indexing, and some comparison notebooks require external services or paid calls.

## Repository map

| Location                                        | Responsibility                                                         |
| -------------------------------------------------| ------------------------------------------------------------------------|
| `src/query-taxonomy/`                           | Query vocabulary, feature banks, and extraction                        |
| `src/dataset_registry/`                         | Source dataset discovery and profiling                                 |
| `src/composition/`                              | Recipes, archetype cells, quotas, and selection                        |
| `src/taxonomy_generators/`, `src/augmentation/` | Feature surfaces and query transformations                             |
| `src/hybrid_search_rrf_dataset/`                | Retrieval lanes, indexing, objectives, labels, and baseline evaluation |
| `src/relevance_judge/`, `src/rungs/`            | Judgment expansion, labeling cascades, and spend controls              |
| `src/encoder_router/`, `src/router_service/`    | Router training, evaluation, and inference API                         |
| `src/pipeline_service/`, `src/demo_service/`    | Dataset tools and the collection-to-search-tests demo                  |
| `src/scripts/`, `notebooks/`                    | Build entry points, experiments, and recorded analyses                 |
| `tests/`                                        | Automated checks                                                       |

For terminology and design history, read [CONTEXT.md](CONTEXT.md) and [SPEC.md](SPEC.md). For experimental limitations and directions considered after the first router, read [VERDICT.md](VERDICT.md) and [SCOPE_DECISION.md](SCOPE_DECISION.md). These are historical records; individual decisions describe the stage at which they were made.

The recurring lesson is that **dataset construction is part of the experiment**. Query diversity, source balance, judgment depth, and the retrieval stack all affect what a router can learn. More generated queries or more stored route decisions only help when their evidence supports the decision being taught.
