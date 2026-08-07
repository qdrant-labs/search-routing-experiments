# hybrid-search-rrf-dataset

A golden routing dataset for a dense/sparse/hybrid Strategy Router: ~46K
queries across 42 corpora, each labelled with which retrieval route
(`dense_only` / `pure_rrf` / `sparse_only`) actually retrieved best, measured
by running all three against an indexed corpus — never by asking a model
which route looks right.

## Use

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
