# Dynamic Router for the Hybrid Search 

This repository contains the work for the routing problem between sparse and dense during search. 

## Why a router

Reciprocal Rank Fusion merges result lists by position and nothing else: a
document scores `1/(k + rank)` in each list, and the two are added (Cormack et
al., SIGIR 2009, `k = 60`). Dropping the scores is deliberate, since cosine and
BM25 are not comparable while ranks always are.

The bill arrives at the top. A document dense ranks first and sparse never
returns scores `1/61`. A document both retrievers rank 62nd scores `2/122`, the
same number. Break-even is rank `k+2`, so weak agreement beats a confident first
place, and the score that proved it was confident is already gone.

That hurts when one retriever is plain wrong about a query, which happens by
family: dense collapses on simple entity questions (Sciavolino et al., EMNLP
2021), BM25 leads on most out-of-domain BEIR sets after losing 7 to 18 points
in-domain (Thakur et al., NeurIPS 2021). Weighting per query recovers up to 7.5%
Precision@1 over fixed-weight hybrid (Hsu and Tzeng, 2025).

The catch: nobody has done this well from query text alone. A trained selector
moved NDCG 0.5106 to 0.5121 (Chifu et al., 2025), and predictors fail hardest
exactly where dense and lexical disagree (Faggioli et al., ECIR 2023). Both
studies ran on hundreds of TREC queries. This repo bets that tens of thousands,
labelled by running all three strategies, close that gap.

Survey and full citations:
[docs/research/qpp-retrieval-routing.md](docs/research/qpp-retrieval-routing.md).

The approach we are taking it is: create a classifier that will be able to determine which strategy is better: sparse / dense or returning to the hybrid. For having a good classifier the first task is creating a dataset that will represent the complexity of the decision between the dense vs sparse vector searches. 

Current repository went through a series of improvements and is currently on a way of delivering a third version of the dataset. Each version tries to improve on its predecessor by providing a better and more comprehansive look of the dataset generation. **This means that the central target of this repository is dataset creation, not classification. Classification used more as a tool for the evaluation, than actual result query (through that is the final goal)**   

## Project modules 

This project has multiple modules each having its area of responsibility. We will go from the simplest and most fundamental ones to the most recent ones. 

### Query Taxonomy

Fundamental module of the project. The basic premise starts from the assumptions - we need to capture diverse queries in our dataset. For that reason we have to know which types of queries exist. The taxonomy is split into several ways, which could be summarized as: structural vs implementational. Structral details is the actual taxonomy - it shows which types of queries exist, while implementational - how to achieve their implementation to happen.  

**A bank is one detector for one feature.** It declares which taxonomy member it
detects, an ambiguity tier saying how precise its pattern is, and the pattern
itself. It returns spans, `(start, end, feature)`. `HTTPStatusCodeBank` claims
`500` in "error 500" and `201` in "HTTP/1.1 201", and nothing else. Adding a
feature means adding a bank, and each one ships about two positive and two
negative test cases, with a test that fails when a bank has none.

Banks that return a number instead of a span are stat banks: stopword ratio,
syntactic depth, term rarity. A number has no character range, so they skip the
resolution step below.

**The engine is the machinery a bank runs on**: `regex`, `spacy_model`,
`wordfreq`, `tokenizer`, `langid_model`. It is declared on the class and
readable before any bank is instantiated, so `FeatureExtractor()` on its regex
default never imports spaCy or torch. Asking for more engines costs
dependencies and time per query, so the cheap path stays the default.

**The taxonomy is six groups**, each a vocabulary enum plus a package of banks:

| Group | Banks | Detects |
|---|---|---|
| `structured_identifiers` | 79, over 8 domains | `192.168.0.0/16`, `sk-live-…`, `Q3 2026`, ISBNs, case numbers |
| `statistical_metrics` | 8 | length, stopword ratio, syntactic depth, term rarity, subword fragmentation |
| `sentence_markers` | 6 | negation, greeting, politeness, interjection, comparative, acronym |
| `corruption` | 5 | encoding artifacts, truncation, paste residue, typos |
| `logical_structures` | 4 | operator syntax, temporal, code fragments, math expressions |
| `semantical` | 3 | language set, code-switching |

Inside a group, banks compete for the same characters and the tier settles it:
the version-string bank claims `v1.0.0` first, so the number bank cannot come
back for the `1.0` inside it. Groups never compete with each other, so one query
can carry an identifier span and a corruption span over the same text. A seventh
group, `query_corpus`, scores a query against a specific collection, so it needs
an index and stays out of the default registry.

### Other important modules

**`composition/`** decides which queries enter the dataset. An archetype cell is
a query type written as bands over taxonomy features, say "short, one opaque
identifier, no natural-language signal". `cells.yaml` holds the cells and
`CellFill` fills a quota for each one from the feature catalog, capping how much
any single corpus may contribute so one dataset cannot own a cell. Cells are
written up front from retrieval mechanics and never read off what the catalog
happens to contain, so a cell nobody can fill is a generation target instead of
a mistake. Every number lives in `Recipe`, which is frozen: changing a value
means a new build.

**`augmentation/`** grows the cells that came up short. An order sheet says how
many rows each thin cell still needs, operators pick real parent queries and
transform them (decorate, rewrite into operator syntax, inject an identifier,
corrupt the text), and the result goes back through the taxonomy extractor to
confirm it hits the bands it was ordered to hit. Selection is deterministic and
the LLM only writes surface text. Anything that fails re-measurement never
reaches the pool.

**`hybrid_search_rrf_dataset/`** produces the labels. It indexes each corpus in
Qdrant, runs all three strategies over the same queries, scores every ranking
against that corpus's relevance judgments, and records which strategy won.
`lanes.py` tracks which datasets carry a usable relevance signal at all: qrels,
ORCAS clicks, or a GooAQ answer passage standing in for the gold document.
`router.py` holds the baseline classifier, kept beside the labels so evaluation
runs against the data it was trained on.

### Current state of the dataset

A golden routing dataset for a dense/sparse/hybrid Strategy Router: ~46K
queries across 42 corpora, each labelled with which retrieval route
(`dense_only` / `pure_rrf` / `sparse_only`) actually retrieved best, measured
by running all three against an indexed corpus — never by asking a model
which route looks right.

## How to use it

```bash
poetry install
dvc pull                # fetches src/data — labels, corpus snapshots, caches
```

Two entry points, both in `src/`:

- **[src/dataset_showcase.ipynb](src/dataset_showcase.ipynb)** — the dataset
  in four stops: what's in it, the decisive core (~5K clear wins), the
  acceptability labels that turn ties into training signal (~38K rows,
  SPEC d60), and how augmentation grows the thin archetypes.
- **[src/route_experiments.ipynb](src/route_experiments.ipynb)** — train,
  validate and probe routers on it. Edit the config cell, Run All.

Design decisions for LLM live in [SPEC.md](SPEC.md), vocabulary in
[CONTEXT.md](CONTEXT.md). Working and archive notebooks are under
[notebooks/](notebooks/); they read `data/` relatively, so run them with
`src/` as the working directory.
