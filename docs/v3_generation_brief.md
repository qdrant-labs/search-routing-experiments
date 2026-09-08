# v3 dataset generation — objectives & how cells fit

*Briefing / context prompt. Read this before working on v3 generation.*

## What we're building and why

A query dataset that **trains a query router** to pick the best retrieval
route per query: **dense**, **sparse**, or **hybrid (RRF)**. The dataset is not
an end in itself — it exists to teach that decision. So "a big diverse dataset"
is not the goal; a dataset that *maximally helps identify where routing goes
wrong* is.

## The three objectives

Every generation/selection choice serves one of these:

1. **Diversity** — cover the space of query archetypes (short keyword queries,
   deep questions, identifier-bearing queries, corrupted queries, rare-vocab
   queries, …). Breadth of query *shapes*.
2. **Representativity** — the queries look like real-world traffic, not
   synthetic templates. Natural phrasing, real corpora.
3. **Utility** — the dataset teaches the router. Operationalized as:
   **maximize decisive rows** (a route clearly wins) and **cover every
   (lane × winning-route) combination**. A diverse, realistic query that every
   route answers equally teaches nothing — it has diversity and
   representativity but zero utility.

## The unit of coverage: cell × corpus-stats × corruption (ADDITIVE strata)

The thing we balance coverage over is richer than a query archetype alone. It
is three **independent** axes, balanced separately (NOT a multiplicative grid —
a full cross-product was rejected as unfillable):

- **cell** — the query archetype (see below).
- **corpus-stats** — how the query sits against its collection (term rarity /
  IDF, query↔corpus overlap, PMI). Per-query, corpus-relative.
- **corruption** — degree of text damage (clean / typo / artifact).

## What a cell is, and how cells fit

A **cell is an a-priori query archetype** — a conjunction of bands on
measurable query features (e.g. "length ≤ 6 words AND no identifier" = a short
keyword query; "term_rarity.rare_share ≥ 0.25" = a rare-vocabulary query).
Cells come from **retrieval mechanics and prior knowledge, never from what the
current data happens to contain.** Consequences:

- A **thin cell is a generation target**, never something to prune. If an
  archetype matters for routing and we have few of it, we generate more.
- Cells are the **diversity + representativity** half of the objective: they
  guarantee we cover the archetypes, drawn from real queries.
- Cells are **NOT the utility half.** Measured finding (`shape_by_cell.ipynb`):
  outcome shape (does a route win?) is a **dataset/lane** property, not a cell
  property — a cell explains ~0.5% of shape variance after dataset. So *which
  route wins* is decided by the query↔corpus↔retriever interaction (the lane),
  not by the cell. **Utility is targeted on the lane × route axis; diversity is
  targeted on cells.** Cells alone cannot make the dataset useful.

There are **44 v2 cells (frozen)** plus **5 new v3 cells** on the new taxonomy
dimensions: `rare_term`, `fragmented` (subword), `typo_bearing`, `high_oov`,
`artifact_corrupted`. The v3 cells are additive.

## Route classes and the 45/45/10 split

- **dense** = dense-decisive rows (routes_differ, dense wins by margin ≥ 0.4).
- **sparse** = sparse-decisive.
- **hybrid** = **genuine ties + rrf-decisive**. A "genuine tie" is all three
  routes tied across *deep* qrels (≥2 judged docs); a **fake tie** (everyone
  ranks the single judged doc at ceiling) is a qrels-depth artifact and is
  **waste**, not hybrid.
- **Target global split ≈ 45/45/10** (relaxed from an initial 40/40/20). Why:
  hybrid genuinely wins only ~4–10% of the time; the 32.5% of rows that "tie"
  are 95.6% fake (single-answer QA/FAQ datasets), so a clean 20% hybrid is not
  reachable and forcing it would fill the hybrid class with noise. RRF-as-hedge
  is the right prior when dense and sparse genuinely tie.

## Constraints the generation/selection loop enforces

- **Global route split** ≈ 45/45/10.
- **Per-dataset decisive floor** — each dataset must contribute ≥ F decisive
  rows (dataset diversity, not just cell diversity). Today most datasets are
  under any floor: fixable by *labelling more* (under-labelled, good yield) for
  some lanes, by *new data* for low-yield hard-reasoning lanes.
- **Waste budget (dictated)** — fake ties + all_zero rows carry no routing
  signal; their count is a capped budget we dictate, not an emergent surprise.
- **Diversity coverage** — spread the selection across the cell / corruption /
  corpus-stat strata.

## Hard architectural constraint

**v3 is strictly additive; v2 is frozen.** Never change or touch a v2 artifact
or code path (v2 `labels.parquet`, `catalog.parquet`, `cells.yaml`, composition
code). v3 lives in its own namespace (`catalog_v3.parquet`, `cells_v3.yaml`,
`v3_feasibility/`). v2's 46K rows may be reused later by *joining*, never
mutating. This keeps v2 a reproducible control to A/B against.

## Scale & current state

- Target ~**200K** queries, tuned for the routing task.
- **Phase 0 (done):** `select_v3_prototype.py` — feasibility + shortfall report
  over the existing 46K labels. Verdict: on current data the split caps at
  ~3,847 rows (sparse-binding); the 200K debt is sparse 12× / dense 6.6× /
  hybrid 2.7× under; per-dataset floor is mostly a *labelling* gap.
- **Phase A (done):** per-query corpus stats + PMI; the additive v3 catalog +
  5 v3 cells — the enriched unit is real (49 cells).
- **Phase 3 / next:** Qdrant-in-loop labelling (index + label generated /
  under-labelled queries in the loop, closing the "label-more" floor gaps),
  then generation to 200K against the shortfall order sheet.
