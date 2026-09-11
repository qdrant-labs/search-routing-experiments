# SPEC: Diversified Query Dataset Pipeline

Vocabulary: see CONTEXT.md. Session dates: 2026-07-15 → 2026-07-20 (each
decision carries its own date).

## Problem

Build a diversified query dataset for the Strategy Router (dense/sparse/hybrid)
by composing natural queries harvested from registered IR datasets with
generated/augmented ones — such that the full feature taxonomy
(`query_taxonomy/query-taxonomy.csv` in the sibling package) is well
represented, every row stays answerable against some corpus, and the dataset
can later be labeled empirically. Current focus is extractor quality and
breadth; strategy labeling is explicitly a later stage.

## Decisions

The decisions below keep their original numbers; gaps are removals, not
renumberings. Removed 2026-08-17 as superseded — d1–d20, d22–d34 (extractor
waves and the slice composition, retired by d48/d49), d40, d42–d43, d51–d59
(the generation campaign), d45–d47 (the logistic-regression router, NO-GO in
VERDICT.md). A surviving decision may still cite a removed one; the full list
is at commit 26b9a93.

21. **Dataset acquisition catalog + wave-1 implementation** (2026-07-20).
    The candidate catalog lives in `docs/datasets.md`: card tuples along the
    six DatasetCard dimensions, verified acquisition pointers, and harvest
    hypotheses (priors for decision 7's profiling-proposes/human-ratifies
    loop). Wave 1 = ORCAS, BRIGHT, QUEST, CRUMB, RAR-b math/code pools,
    LIMIT, DBPedia-entity. Implementation plan:
    - New `DatasetName` members: `ORCAS`, `BRIGHT_<split>` (the 3
      taxonomy-relevant splits first: `leetcode`, `aops`,
      `theoremqa_questions`; the other 9 later), `QUEST`, `CRUMB_<task>` ×8,
      `RARB_MATH`, `RARB_CODE`, `LIMIT`, `DBPEDIA_ENTITY`. Parameterized
      classes (MiraclDev pattern) for BRIGHT/CRUMB keep it one class per
      source.
    - irds one-liners (ORCAS, DBPedia, ANTIQUE): subclass
      `IRDatasetsBacked`, set `irds_id`, write the card (~15 lines each).
    - hf fetchers: `load_dataset(repo, config, split=..., streaming=True)`,
      yield `Query(str(id), text)`; field names verified at registration
      (CRUMB and RAR-b schemas unconfirmed).
    - url fetchers (LIMIT, XOR-TyDi): `load_dataset("json",
      data_files=<raw url>, streaming=True)` — reuses the HF machinery;
      cards get `SourceKind.URL`, no new base class.
    - Sample caps (proposals, ratified with the first profile run):
      ORCAS 100K, GooAQ 50K, WebFAQ 50K/language.
    - Every registration ends with `registry.profile()` + harvest-target
      ratification (decision 7); the catalog's hypotheses are the priors.
    LIMIT registers but is excluded from harvest targets (its value is the
    strategy-labeling stage); MEMERAG is not a query source (MIRACL's
    queries, already registered). Open ratifications: MIRACL
    `llm_target`/`non_trivial` card vs the candidate-list coding;
    DBPedia scope G-vs-S.

37. **Golden set: route labels from retrieval outcomes, not from an LLM's
    opinion** (grill-me 2026-07-28, supersedes d35 and d36; amends d30's
    "which strategy wins NDCG" label rule). Two orthogonal tasks are now
    named: *dataset improvement* (more features) and *golden set* (target
    classes). This decision covers the golden set.
    (a) *The objective is `0.7·HitRate@1 + 0.3·NDCG@10`*, with a
    per-dataset `min_relevance` binarizing graded qrels (2 for TREC-DL's
    0–3 scale, where grade 1 is "related but does not answer"; 1 for
    qrels already binary). Bare NDCG@10 was wrong not because it is
    uncomputable on thin qrels but because it has no top-1 primacy, and
    production cares about the top hit.
    (b) *The weights make it lexicographic, not a blend.* While
    hit_weight > ndcg_weight the two score ranges are disjoint (rank-1 hit
    ⇒ ≥0.700, miss ⇒ ≤0.300), so the secondary term can only discriminate
    *within* each group. Per-query values with one relevant doc: rank 1 →
    1.000, rank 2 → 0.189, rank 3 → 0.150, rank 10 → 0.087.
    (c) *NDCG@10 is the tie-breaker because the cheaper candidates go
    blind on data we hold.* MRR@10 is 1.0 for every route with a relevant
    rank-1 doc, so it cannot separate a route that surfaced 1 of 4
    relevant docs from one that surfaced 4 of 4 — the label would fall
    through to tie-break order. Recall@10 collapses to two values when a
    query has a single relevant doc (rank 2 and rank 10 score alike),
    which is the majority of the corpus (msmarco-dev ~1.06 judged/query,
    rarb, most crumb, orcas). Verified: with exactly one relevant doc all
    three variants are strictly decreasing in its rank, so they agree on
    the route — divergence only exists where a query has ≥2 relevant docs.
    (d) *Judgments leave the result row.* `gold_qrel` as a per-row dict
    became a parquet struct with one field per distinct doc_id in the
    file — measured at 30,000 fields for 76 rows, so it does not survive
    composition scale. Replaced by `QrelStore`: long-format
    `(dataset, query_id, doc_id, relevance, source)`. `source` ∈
    {`human`, `click`, `llm`} with conflict priority human > click > llm,
    so scoring a dataset against either lane is a filter, not a second
    pipeline.
    (e) *Per-route results are persisted.* `GoldenRoutingDataset` carries
    `route_scores` and `route_rankings` keyed by strategy name (three
    stable keys — parquet encodes a fixed struct; keying by doc_id would
    not). Any cutoff-≤10 metric, latency margin, or tie rule is therefore
    re-derivable with zero retrieval. This is what makes (a) and (c)
    reversible decisions rather than one-way doors.
    (f) *An LLM cannot be asked which route wins.* Dense-vs-sparse is a
    property of the (query, corpus, index) triple, not the query: for
    nfcorpus's `"DHA"`, sparse wins if the relevant docs say "DHA" and
    dense wins if they say "docosahexaenoic acid", and the model sees the
    same three characters either way. IDF is a corpus statistic. This is
    an information gap, not a capacity gap — a stronger model shares the
    blind spot. The LLM's only legitimate job is
    `(query, doc) → relevant`, a function of its actual inputs.
    (g) *Measured: the production classifier underperforms a constant.*
    On `data/golden_router` vs `data/router_llm` (trec-dl-2022, 76
    queries, NDCG@10): route agreement 43% (33/76, measured) against 82%
    (62/76) for always-picking `dense_only` — the latter *derived* as the
    majority class of the oracle's own labels, not a measured system.
    Regret +0.077 mean, +0.252 p90; the classifier picks `pure_rrf` 61%
    of the time where the oracle wants 14%, which explains hybrid's poor
    top-1. **Does not generalize**: n=76, one dataset, and that corpus is
    `TrecDL2022(30000).materialize()` — 30K docs that are almost all
    judged answers, which inflates dense and starves sparse. Motivates
    re-measuring on a realistic index; proves nothing on its own.
    (h) *The bar is the best constant route, not random.* Constant-dense
    and constant-sparse baselines must appear in every comparison, or a
    router can look respectable on regret while losing to one line of
    code. Only constant-hybrid (`data/pure_rrf`) exists today.
    (i) *Three outcome shapes; two are usable.* All routes tied above
    zero ⇒ equivalent, send to the cheapest (signal for the speed
    requirement). Routes differ ⇒ the quality signal. All routes 0.0 ⇒
    unanswerable, and **no valid label exists** — currently the argmax
    falls through to whichever route is first in the list, fabricating a
    `dense_only` label (1 of 76 on trec-dl; the rate scales with corpus
    realism and with judging orcas only to depth 10). Unanswerable
    queries need an explicit outcome. Ties anywhere must resolve by a
    deliberate rule, noting `dense_only` is likely *not* the cheapest
    route since BM25 needs no query-side transformer pass.
    (j) *The 50K is a candidate pool, not a training set.* Usable yield
    is unknown until retrieval and scoring run, so it is measured on an
    anchor before committing: nfcorpus, 323 queries over 3,633 docs,
    already indexed as collection `nf`.
    (k) *Build labels with zero LLM involvement first.* Existing qrels
    plus ORCAS clicks cover every dataset in the composition, so no
    manufactured qrels are needed to produce a first golden set. The
    judge's real job shrinks to **hole-filling** — correcting the
    documented asymmetry where dense retrievers hit 14–32% qrel holes
    against BM25's ~6%, which biases labels against dense. That is a
    correction to labels we can already compute, and it is self-
    validating: if hole-filling helps, dense gains in the predicted
    direction. Deferred until the raw hole rate per route is measured.
    (l) *Corpus-relative features are now a blocker, not a deferral.*
    Per (f), a query-only router inherits the LLM's exact ceiling. The
    only features available *pre*-retrieval — and therefore compatible
    with the speed requirement — are query-term IDF in the index and
    out-of-vocabulary rate. Score margin and dense/sparse candidate
    overlap require retrieving first, so they can inform a fusion
    decision but never the choice of which retrieval to run.
    — *The LLM judges documents; arithmetic picks the route. Everything
    needed to re-pick it later is on disk.*

38. **msmarco-passage-dev anchor: the composition's largest lane gets
    labelled on a realistic index** (grill-me 2026-07-28, executes d37j/g
    for the 31% lane; nfcorpus anchor measured same day).
    (a) *Queries are the composition's, full stop.* `CellFill().
    build()` → `RouteLabels.rows_for("msmarco-passage-dev")` — 15,678
    rows. Local dev qrels (`~/.ir_datasets/msmarco-passage/dev/qrels`,
    59,273 judgments) cover 7,697 of them (49.1%); those get labelled,
    the other 7,981 stay `unlabelled` in coverage. Source datasets supply
    qrels and passage text for selected query_ids, never queries.
    (b) *Corpus text via ir_datasets*, the same tool that fetched the
    queries/qrels on 07-15: one user-initiated ~1.06GB fetch of the v1
    collection, whose doc_ids are the ones the qrels name. After that,
    fully offline-re-runnable.
    (c) *Corpus recipe: all 8,219 judged-relevant passages force-included
    + uniform-random distractors (fixed seed) to 100K total.* Uniform
    sampling preserves the collection's vocabulary/IDF profile — the
    d37(g) fix: trec-dl's judged-docs-only 30K was near-all answers,
    inflating dense and starving sparse. Full 8.8M rejected for now
    (~day-scale embedding); reversible, the recipe is a parameter.
    (d) *Artifacts follow the existing owners.* `data/msmarco-passage-dev/
    {corpus,qrels}.parquet` in `SnapshotDataset` layout; Qdrant collection
    `msmarco_routes` with the same `dense_base`/`sparse_base` configs
    (bge-small-en-v1.5 384 cosine, Qdrant/bm25); `min_relevance=1`
    (grades are binary). Labels merge into `data/route_labels/
    labels.parquet` via `RouteLabels.label`.
    (e) *Rule parity with the nfcorpus anchor.* The shipped argmax
    labels this run too, so the two datasets differ by exactly one
    variable. The deliberate tie/all_zero rule (d37i) stays a separate
    change, re-derivable from persisted route_scores/route_rankings
    (d37e) with zero retrieval.
    (f) *Notebook: append per-dataset sections* to `route_labels.ipynb`
    following the §4–§7 pattern, closing with coverage over both
    datasets. A dataset loop is premature at 2; revisit at 5+.
    (g) *Why this dataset second:* median 1 relevant/query (vs nfcorpus
    16) is the regime where two different top-10 lists cannot both be
    right — measured on nfcorpus that dense-vs-sparse top-10 Jaccard is
    0.184 yet 66% of rows sit within 0.06 of a tie, so rich judging, not
    retrieval agreement, was eating the signal.

39. **Qrels acquisition: every human-judged lane, snapshot-first**
    (grill-me 2026-07-29, d38 follow-on; scales d37j's anchor to the
    composition). Scope: the 18 QQ-grounded lanes, ~18.3K selected rows.
    Motivation: the labelled distribution (84% dense over decisive rows)
    is an artifact of *which* lanes are labelled — two natural-language
    lanes where the dense encoder is at home. The sparse/hybrid signal
    lives in the unlabelled technical and entity lanes.
    (a) *Registry bypass, settled.* Doc-side acquisition lives in
    per-lane `RetrievalDataset` classes in
    `hybrid_search_rrf_dataset/retrieval.py`; the registry stays
    queries-only — its own documented doctrine. Proof the registry was
    never on the labeling path: zero `dataset_registry` imports in the
    labeling package, and 8,020 rows labelled while the cache stayed
    `[query_id, text]`. The TODOS "lift queries-only" item is re-scoped
    to what actually needs it — corpus-relative features (d37l) — and
    per-lane Qdrant indexes may serve query-term IDF before any refill.
    (b) *Corpus-pending snapshots.* Pass 1 writes
    `data/<lane>/{queries,qrels}.parquet` for all 18 lanes; `corpus.
    parquet` arrives in pass 2. `SnapshotDataset` init tolerates the
    pending state, `corpus()` fails loud. `coverage()` gains a
    `qrels_ready` state: unlabelled → qrels_ready → labelled. Network
    hit once per lane ever.
    (c) *Declarative lane table.* `LANES` in
    `hybrid_search_rrf_dataset` maps composition key → (source class,
    min_relevance, quirks — BRIGHT's `excluded_ids` honored). The
    notebook iterates the table; amends d38(f): the per-lane-sections
    pattern was right at 2 lanes, the loop is earned at 18.
    nfcorpus/msmarco sections stay as shipped; done lanes sit in the
    table and skip via `label()`'s existing idempotency.
    (d) *min_relevance defaults:* 1 for the binary lanes (rarb, bright,
    crumb, quest, limit, miracl); dbpedia-entity 1 with the §9
    zero-retrieval sensitivity cell as the escape hatch; trec-dl 2 per
    d37(a) (parked anyway).
    (e) *One threshold corpus rule.* Corpus ≤ 100K docs ⇒ index in
    full; > 100K ⇒ d38(c) recipe verbatim (judged force-included +
    uniform-random distractors to 100K, seed=0). No per-lane hand
    decisions; the table records measured size and which branch fired.
    Expected: rarb/bright/crumb full; quest, dbpedia-entity,
    miracl-en-dev capped. Embedding budget ≈ 500–800K docs total,
    cached. **Amended 2026-07-29 (i)**: the rule presumes relevant ≪ cap,
    and crumb-code-retrieval breaks it (108,782 judged-relevant of a
    232,444 corpus — 47% answer-dense intrinsically). `Lane.corpus_cap`
    is the deliberate per-lane override for exactly this case; set to
    250K there, so the lane indexes its full real corpus. A global
    raise was rejected: it would drag five ~1M-passage crumb corpora
    to the new cap for ~15h of embedding nobody asked for.
    **Amended 2026-07-29 (ii): flat 100K default replaced by the 20/80
    recipe, and the recipe is computed, not hand-written** —
    `CorpusRecipe(answer_share=0.2, floor=10_000, ceiling=100_000)` in
    retrieval.py; `target = clamp(relevant / answer_share, floor,
    ceiling)` evaluated from each lane's own qrels at materialize time.
    `LANES` carries only pinned exceptions with reasons
    (`Lane.corpus_target`): crumb-code 120K (its 108,782 relevant — a
    median of 23 relevant docs per query, the benchmark's design —
    exceed the ceiling; floor-plus-pad chosen over full-232K for the
    embedding budget, the hard confusables being in the forced set
    either way), limit 50K (the recipe would compute the floor and
    delete its stress-test design), rarb-code kept at its
    already-embedded 100K. Rationale
    (user-argued, measurement-backed): the label is an argmax over
    three routes on the same index, and the corpus-room diagnostic
    showed 9.3× corpus growth flips only 13.6% of labels while
    `all_zero` only grows — distractor mass buys difficulty, not route
    signal. The three recipe parameters are the tunable surface; floor
    keeps every corpus above the strategies' fetch depth (trivial
    ranking otherwise), ceiling stops answer-heavy lanes ballooning
    (clinical computes to 198K unbounded). Wave-1 embedding ~25h →
    ~11h. Caveat carried: lanes differ in corpus size, so cross-lane
    margin comparisons pick up a size term; per-lane labels stay
    internally valid.
    (f) *Pass-2 order: sparse signal first.* Wave 1, the ≤100K
    technical lanes — rarb-code, crumb-code-retrieval, bright-leetcode/
    aops/theoremqa, crumb-theorem/legal/clinical/stack-exchange/paper,
    rarb-math, crumb-set-operation, crumb-tip-of-the-tongue, limit
    (~16K rows). Wave 2, capped lanes by entity signal: quest →
    dbpedia-entity → miracl-en-dev. Row-count-first ordering rejected:
    it answers the sparse question last.
    (g) *Parked, with reopen triggers.* ORCAS click lane (15,744 rows;
    own grill — needs the 18.8M click pairs and a trust model for
    clicks). LLM judgment source for never-judged rows (msmarco's
    7,981 + pass-1 findings; trigger: d37k hole rate measured).
    trec-dl-2022 rebuild (16 of 79 selected rows judged; hours of v2
    scanning; trigger: hole-filling reopens it).
    (h) *Standing rules carried:* collections named `<lane>_routes`;
    shared `./.embedding_cache`; seed=0; missing qrels ⇒ `unlabelled`,
    never query substitution; argmax rule parity until the tie-rule
    decision lands.
    — *The registry catalogs queries; lanes own their judgments. The
    distribution question is answered when the technical lanes land.*

41. **Label form: the score vector is the record; the route column is a
    derived serving decision** (grill-me 2026-07-29; resolves d37i's
    tie-break + all_zero halves and d38's label-form deferral. Measured
    over the 15,413 rows on disk mid-wave-1: 73% of `route` values were
    decided by Python list position — 3,769 exact top-two ties inside
    routes_differ, 5,688 all_tied, 1,745 all_zero — not by retrieval).
    (a) *Canonical label = the three per-route objective scores*,
    already stored on every row. A tie is honest by construction — two
    equal numbers assert no winner. Training leans regression over the
    vector; any classifier target is derived from it by (b)'s rule,
    never stored as a separate truth.
    (b) *Route derivation — quality first, cost second:* `route` = the
    cheapest route among those achieving the maximal score; null when
    the max is 0 (all_zero: no route worked ⇒ no label — ends the 1,745
    fabricated `dense_only`). Cost never overrides quality: sparse at
    0.0 cannot take a row whose tied-best is {dense 1.0, rrf 1.0}. A
    tied row's route is the operationally correct decision, not a
    tie-break hack — when quality is equal, serve the cheapest — so the
    column encodes production's actual job.
    (c) *Cost order `sparse_only < dense_only < pure_rrf`.* rrf costlier
    than each component is structural (it runs both plus fusion) and
    settles 3,744 of the 3,769 pair ties assumption-free; sparse <
    dense (no query-side transformer pass) is the single assumption,
    pending the latency benchmark (TODOS), load-bearing only for
    all_tied rows + 25 dense+sparse ties, reversible by re-derivation
    in seconds.
    (d) *Decisive redefined parameter-free:* winner hit rank-1 AND
    runner-up missed rank-1 — under the lexicographic objective exactly
    margin ≥ 0.4. Replaces the hand 0.06 band, which was unknowingly
    approximating it (1,673 vs 1,668 on the rows on disk; the 5 lost
    rows are both-hit tail differences, not route separations).
    (e) *Ties are exact* (`outcome_shape`'s 1e-9 tolerance). Near-ties
    stay routes_differ with a thin margin; margin and winner-set are
    read-time derivations, never stored columns. Resolves d33/R2's "set
    tie margin ε": ε = 0.
    (f) *One rule, one place:* `derive_route` + the cost order live
    where `StrategyName` lives (fusion.py), importable by golden.py and
    labels.py without cycles. `GoldenRoutingBuilder.strategy_name` = the
    serving choice (non-null; all-zero degenerates to cheapest overall
    — it names the ranking the row carries, not a label);
    labels.parquet `route` = the label (null on all_zero). The
    list-order `max()` dies at the source.
    (g) *Migration is a re-derivation, not a re-run:* one pandas pass
    over labels.parquet after wave 1 lands rewrites `route` from the
    stored scores; notebook readouts switch to (d)'s decisive. Reading
    rule: quality-dominance headlines are read over decisive rows only
    — the raw route column now mixes quality winners with cost policy
    on tied rows (sparse inherits the all_tied mass by design).
    — *The scores are the measurement; the route is a decision computed
    from them. Nothing in the golden set is a coin flip anymore.*

44. **Corpus-conditioned routing: the headroom readout is recorded, the
    router gains collection eyes, and two pilots gate every scale-out
    claim** (grill-me 2026-07-30; informed by the 10-thread labeling-
    strategy research in `docs/research/route-label-sourcing.md` and by
    arXiv:2504.01101's negative results; resolves d37(l)'s blocker with
    a concrete feature list).
    (a) *The headroom readout is a recorded fact with mandatory
    caveats.* Measured 2026-07-30 over the 24,338 labelled rows
    (16 lanes): always-dense 0.481 mean objective; best constant PER
    collection 0.511 (+6.3%); per-query oracle 0.557 (+9.0% further,
    +15.8% total). Decisive rows 2,510 (10.3%); their winner split:
    dense 1,858 / sparse 507 / rrf 145 — equal class thirds are
    structurally impossible under the top-1-primacy objective; the skew
    is the finding. Per-lane ceiling runs 0.3% (limit — 43% decisive
    but one-sided, so the constant already captures it) to 28.7%
    (crumb-theorem); technical/entity lanes carry 20–29%, msmarco 6.6%.
    Caveats travel with the numbers, always: oracle is a ceiling, not
    an achievement (published QPP-driven selective processing *achieved*
    ≤~4%, arXiv:2504.01101, whose Fig. 4 single-predictor selection
    "seldom outperform[s] the individual system"); the pooled figure
    depends on an arbitrary lane mix (msmarco alone is 32% of rows and
    drags it toward its 6.6%); the composition optimizes feature
    diversity (d32), never routing signal — **this dataset is not an
    optimal routing dataset and the readout must say so wherever it is
    shown**; qrel holes unmeasured (d37k); stack-pinned to bge-small
    (R2 pilot queued). Lands as a permanent route_labels.ipynb section
    that prints the caveats beside the table (closes d39's "re-read the
    distribution" item), re-runnable as lanes land.
    (b) *Six collection statistics as a side artifact; labeling
    untouched.* Per (query, lane): avg-IDF and max-IDF of the query's
    terms in the lane corpus, OOV share (query terms absent from the
    collection vocabulary), collection size N, avgdl, query-vocabulary
    overlap share. One offline counting pass over each lane's existing
    `corpus.parquet` (tokenizer declared and pinned in the artifact);
    written to `data/route_labels/collection_features.parquet` keyed
    (dataset, query_id). labels.parquet stays frozen — paid labels are
    never edited; the stats are router features computed after
    labeling, so golden-set creation gains zero extra cost (the only
    standing obligation, corpus.parquet on disk per lane, is already
    the snapshot layout). Mechanism lineage: resource selection
    (CORI/ReDDE/Taily; TREC FedWeb 2013–14) — collection statistics
    transfer to unseen collections, while query-only predictors do not
    ("with collections as the main factor and rankers next",
    arXiv:2504.01101, ANOVA-backed; see
    docs/research/federated-search-resource-selection.md).
    (c) *Transfer pilot: two protocols, and the gap between them is the
    measurement.* One simple classifier (the five signals + span
    features), evaluated under (i) a random 20% mask within every lane
    — the existing-deployment question, near-duplicate-aware (the
    ~5.5% cos>0.95 pairs must not straddle the split) — and (ii) a
    hide-one-dataset rotation, each of the 16 lanes taking one turn
    fully unseen during training — the new-customer question. Each
    protocol runs with and without (b)'s features, raw and
    per-collection-normalized. Metrics: share of held-out headroom
    captured against the d37(h) best-constant bar; the (i)−(ii) gap
    reported as the corpus-dependence measurement — (b) succeeds iff
    it shrinks that gap. Side test, nearly free: can (b)'s six numbers
    alone predict each lane's best constant (16 predictions — the
    +6.3% tier, a collection-level product needing no per-query
    router).
    (d) *List-preference judge spike — promoted from deferred to now.*
    ~500 stratified rows from three contrasting lanes (nfcorpus /
    crumb-legal-qa / rarb-math); the judge sees the query plus each
    route's stored top-10 (route_rankings, d37e — zero new retrieval,
    docs read from the lane corpus parquets); strongest current model;
    every comparison asked twice with list order swapped, ties allowed;
    agreement vs the empirical route read over decisive rows.
    Thresholds: ≥80% agreement → the scalable-labeling path opens
    (hole-filling d37k, then real-query labeling); 60–80% → judge-panel
    design; <60% → that number goes into the CTO conversation as-is.
    d37(f) is not violated: the judge sees retrieved lists — corpus
    evidence — never the bare query; it stays calibrated against the
    empirical spine, never trusted raw.
    (e) *Named-deferred with reopen triggers.* Per-deployment
    auto-benchmark (SEARA pattern: generate style-diversified queries
    from a customer corpus via taxonomy-generators, label empirically
    on their own index; d40's admission rules already fit): reopens
    when (c) reads positive AND a per-customer consumer exists.
    Interleaving on page-search (production click truth, aggregate
    first): reopens with the log acquisition already in TODOS. Evidence
    map for both: docs/research/route-label-sourcing.md.
    — *The labels stay measurements; the router learns to read the
    collection it serves; every scale-out claim now has a number to
    beat.*

48. **Composition redesign: archetype cells replace span floors; the split
    becomes a train/control boundary** (2026-08-04; executes d45(h5) ahead of
    d47's ablation — the diagnosis no longer needs it). Cell definitions:
    `src/composition/cells.json`. Generation brief:
    `docs/composition-cells-prompt.md`.
    (a) *Diagnosis: route is a lane property, not a query property.* Sparse-win
    rate spans 0.8%–100% across the 16 labelled lanes, and query-side signals
    that look predictive pooled go flat within a lane. Most populated archetype
    cells under the d32 fill drew the bulk of their rows from one corpus. This,
    not feature coverage, is what caps the router, and it explains the
    within-lane vs holdout-lane gap.
    (b) *A recipe change alone is worthless.* The d32 fill already selected
    every feature-bearing row available in the labelable lanes — the
    `exhausted` order-sheet lines restated. No different recipe has unexploited
    supply to pick. Rejects "reopen recipe values" as a standalone fix.
    (c) *`ArchetypeCell` replaces the 1-D span floor.* ANDed `AxisBand`s over
    `catalog_axes` columns, plus `any_of` so one cell can span an identifier
    family (the router cannot learn a rule from 33 examples of one format, but
    can learn the token shape across several).
    (d) *Cell definitions carry no numbers derived from current holdings.*
    Supply, lane spread, labelling cost and fill tier are COMPUTED at fill time
    against whatever catalog exists then; freezing them in the spec is
    stale-by-construction and imports today's bias into tomorrow's fill.
    `predicts` is a prior from retrieval first principles, tested by labelling,
    never used to allocate.
    (e) *Never narrow a target from current labels.* Asymmetry: observing a
    route in a cell PROVES reachability, while not observing one proves
    nothing — absence of evidence in a lane-biased sample. Narrowing is also
    self-fulfilling, since a quota that asks only for dense never draws
    candidates that could show sparse. Every cell quotas dense AND sparse;
    `n_per_route` is a single Recipe value, not a per-cell judgement.
    `pure_rrf` is never quotaed — it wins on near-ties, which the decisive
    filter removes by construction, so its signal comes from the d41a
    score-vector target.
    (f) *Lane diversity is a share, not a count.* `min_lanes` is satisfiable
    while a cell remains single-corpus in substance; `max_lane_share` is
    checkable against real supply. Computed per cell at fill time as the
    tightest cap that cell's lanes can satisfy. Cells that cannot reach the
    target share are the acquisition requirement, stated in numbers.
    (g) *Slices A+C merge; B+D become the control group.* A and C were two
    mechanisms for one job (a cell predicate with zero identifiers plus a
    length band IS a C stratum). B and D are the only rows whose distribution
    we did not choose, so they become the never-trained evaluation slice —
    spending them as training data throws away the only unbiased measurement
    we own (WEAKNESSES #14). D's mechanism dies with it: three champions at
    ≤50% each is what the ≤20% allocator forbids. Per-archetype eval runs on
    held-out CELL rows, since rare archetypes are by definition rare in an
    unbiased sample; aggregate realism eval runs on the control group.
    (h) *Cell set provenance: three LLM runs, scored not read.* Ranked against
    the catalog on hypothesis breadth, lane concentration and redundancy rather
    than on how convincing the rationales read. All three runs skipped the same
    thin-supply features — the region the d33 floors also under-served —
    covered by hand-written cells grouped by retrieval mechanism. Caveat
    carried: the brief fed the runs statistics from our own labels, so the
    priors are contaminated to that extent; a clean re-run needs those sections
    cut.
    (i) *Gates before the fill.* Acquisition (the ≤20% cap makes T=100,000
    unreachable from current holdings) and the d47 CorpusIndex side test, which
    decides whether lane diversity is capped by row share or by corpus-stat
    band. Both govern which rows get labelled, so both precede the spend.
    — *The old fill asked "does feature X appear enough?" The cells ask "does
    it appear enough, in enough corpora, with enough outcome variety?" Only the
    first was answerable before labelling, which is why the fill is iterative.*

49. **Datasets become supply, not structure: the box model** (grill-me
    2026-08-04; supersedes d29's per-dataset quota framing and d33d's
    champion/dark-forest mechanism; amends d39g's ORCAS parking).
    (a) *Datasets are undifferentiated boxes; cells are the only objective.*
    Dataset identity stops being a selection axis. The fill searches every box
    for queries satisfying a cell predicate and fills that cell's quota; which
    box a row came from matters only to `max_lane_share`. Kills the "which
    dataset should we acquire" judgement call — the answer is all of them that
    can produce a label.
    (b) *Size is an output, not a target.* No row-count goal. T is whatever
    the filled quotas plus the dark forest come to. Rejects hunting for large
    datasets: under a per-dataset cap, supply depth past the cap buys nothing
    and only lane COUNT moves the ceiling, so selecting for size is selecting
    for the wrong thing.
    (c) *The global ≤20% per-dataset cap is dropped.* `max_lane_share` inside
    each cell is the whole constraint — a diversified global mix is fully
    compatible with individual cells being single-corpus, which is the
    confound, so the cap belongs where it means something. Global shares are
    reported, never targeted. Also removes the cell-vs-ledger conflict that
    would have forced the deferred ILP escalation.
    (d) *Fill order puts the cheap decision first.* Cell membership is
    computable from query features alone, so: load queries from every box →
    assign cell candidates across all boxes → materialise and embed ONLY the
    corpora that won quota slots → label → check route quotas → top up. The
    expensive corpus decision falls out of the fill instead of being a guess
    about a dataset nobody has opened.
    (e) *Box pool: everything registered plus Wave 2, English only.* Wave 1 is
    already registered but not exhausted — `BrightSplit` and `RarbPool` are
    parameterized, so the 9 unregistered BRIGHT splits and the RAR-b
    commonsense pools (TempReason targets the temporal cell) are an enum
    member and a line each. Those are the cheapest lanes available (landed
    2026-08-04: 12 BRIGHT splits registered, 30 datasets / 29 lanes).
    Admission is by **answer key, not by grounding card** — corrected
    2026-08-04, the first draft of this clause wrongly excluded all QC boxes.
    Two admissible kinds: QQ (source qrels) and QC-with-passage-answers,
    which is d40(a)'s `doc_grounded` path — the answer passage IS the
    grounding doc, msmarco's own regime. GooAQ is the case that forced the
    correction: measured 62.7% passage-answer coverage over a 100K sample,
    and its Google-autocomplete register (short, telegraphic, unit-bearing)
    is exactly what the msmarco-dominated short-question cells lack.
    Inadmissible: QO (no corpus, so nothing to retrieve) and
    QC-without-answers (no key at all). Wave 2 therefore admits FreshStack,
    ANTIQUE, LoTTE, WebFAQ en, ScIRGen-Geo en, CLERC (pending availability),
    GooAQ, and nq_open if it ships a corpus — its card says QC while its
    description says "short answers only", so verify before counting it.
    XOR-TyDi excluded — cross-lingual is its point and the extractors are
    English-only (`en_core_web_sm`; d14 rejects the weaker `xx` model).
    Query-wellformedness excluded twice over: QO, and its only stated
    consumer was the dropped corruption group. hotchpotch-simulated is not a
    box but keeps a job: generation calibration against ORCAS.
    (f) *ORCAS unparked as its own lane* (amends d39g). `msmarco-document`
    corpus, `QrelSource.CLICK`. It is the only source that breaks the
    head-of-traffic cells' single-corpus concentration, because it is a
    genuinely different corpus at comparable scale — real Bing queries versus
    crowdsourced questions — and it supplies four otherwise-empty cells.
    Accepted cost: positive-only click qrels, ~1 judged doc per query, the
    known-noisy regime of WEAKNESSES #10. `deferred` as a label lane is
    retired; lanes are named by qrel source.
    (g) *Existing labels are reused on overlap.* Labels key on
    (dataset, query_id) and record what happened when retrieval ran; the old
    bias lived in which queries were selected, not in the labels. Fresh
    selection, free labels on the intersection.
    (h) *`predicate − 1` relaxation: relax where the feature exists, mint
    where it does not.* Measured over the thin cells: only two have a
    near-miss to nudge (the feature is common, just not at that length band)
    and nine have no candidate at any length in any box. So StatRewrite
    serves the two, `InjectOperator` the nine, `OperatorSyntaxRewrite` the
    boolean cell. Injection is therefore the dominant augmentation path, not
    the fallback — which puts d40e on the critical path for roughly a third
    of the cell spec.
    (i) *The augmentation stack is reused, not rebuilt.* It is keyed on a
    floor NAME and is indifferent to what a floor is, so cells emit
    shortfalls with `floor = cell.name` and `operator_for` maps per (h).
    `AugmentationCampaign.pilot_n` already implements pilot-sized staging for
    gated floors, so cells short of natural supply need no new tiering
    mechanism. d40e coherence pilot runs in parallel with box loading — it
    needs no new data, and a failure is cheapest to learn before the corpora
    are paid for.
    (j) *Dark forest = leftovers.* Random draws from queries that entered no
    cell, at ~20% of the cell rows. Strictly better than d33d's three-champion
    mechanism: it is genuinely unchosen rather than chosen-to-look-unchosen,
    and it costs nothing to select. The 20% is provisional — the defensible
    size depends on how often the router and the best constant disagree per
    row, which is only measurable once these rows are labelled, so it gets
    resized next pass.
    (k) *The d32 selection artifact is retained, not overwritten.* Every
    number in PLAN.md and WEAKNESSES.md references it.
    — *The old design made datasets structural — champions, per-dataset
    quotas, a dark forest drawn from named sources. Making them anonymous
    supply removes a whole layer of accounting and lets the objective be
    stated once, in the cells.*

50. **Cells are a-priori archetypes: predicate repair now, mechanism-driven
    expansion next, generator deferred** (grill-me 2026-08-05). The audit
    (`src/selection_audit.ipynb`) asked whether the 32 cells cover enough and
    whether more single-route cells were needed; the answer inverted the
    question.
    (a) *A cell's validity is a real dense/sparse divergence mechanism plus
    expressibility as bands over measurable features — never natural supply in
    the acquired lanes.* The 41 datasets are supply, not ground truth (d49).
    Thin cells (uuid 24 catalog rows, ip 51, email 35) are generation targets,
    not weak cells; low supply must never rank, prune, or gate a cell. This
    retires the "supply-as-validity" reasoning the audit's feasibility counts
    invited.
    (b) *Predicate repair happens at the bank, with a re-extraction.* The audit
    found the sparse-leaning cells matching false positives, and the counts are
    post-resolution catalog columns, so the fix is in `query_taxonomy`, not
    cells.json: **UUID** keeps its 32–64 hex digest branch but rejects
    low-entropy / single-repeated-char runs (`aaaa…` is a padding artifact, not
    a digest) — a regex negative-lookahead if edify exposes backreferences,
    else a bank post-filter plus an `OVERRIDES` generator entry; **DateTime**
    drops the bare-epoch branch entirely (any 10-digit int in 1.5–1.9e9 is a
    phone/ID false positive), ISO 8601 only. Rebuild `catalog.parquet` (380K
    rows); the generator↔bank round-trip test stays green (generators derive
    from bank patterns, so a regex tightening self-heals). IP-vs-version is left
    irreducible — a 4-part version is a valid IPv4 and RIGID IP wins the range,
    so `version_string` undercounts; both are two-route cells and labelling
    tests them. Email is left as-is: the strict pattern's flagged rows are real
    incidental addresses, not FPs.
    (c) *Two cells.json guards ride along, no re-extraction:* `uri_in_query`
    gains `length_words below 15` (stop claiming prose that merely mentions a
    link); `opaque_token_any_domain` drops `http_status_code` from its `any_of`
    (the "277 V" false positive).
    (d) *Cell expansion is produced the way the current set was — a separate,
    human-run LLM task that writes a static `cells.json`, NOT an automated
    pipeline pass — with a revised prompt.* The 32 are shallow (single-feature
    or 2–3-band); the gap is cross-group INTERACTION cells and statistical
    permutations (conjunctions across identifiers × stats × markers × logical —
    e.g. rare-entity × high-stopword × short). The lever is the generation
    brief `docs/composition-cells-prompt.md`, adapted to prior shortcomings:
    **cut the `MEASURED FACTS` section** (it fed our 16-lane label statistics as
    facts to "respect" — the d48h `predicts`-contamination) and **cut the
    `SUPPLY` section** (per-feature catalog counts — the supply-as-validity error
    d50a retires); **drop `BUDGET`/`n_per_route`** (not a cell property, computed
    at fill time). Keep the exact-column vocabulary (the measurability
    constraint — invent no columns), the bad-cell rules, and the
    pooled-vs-within-corpus confound restated as a PRINCIPLE, not a table of
    numbers. Add an explicit ask for cross-group conjunctions and statistical
    permutations, and for a stated retrieval mechanism + `predicts` prior per
    cell. Argue from retrieval mechanics and entity×statistic relationships
    (LLM knowledge), never from what the acquired corpora contain. Archetypes
    needing a property nothing measures are logged as a taxonomy-extractor
    backlog, not built now. Resolves the d48h contamination deferral. *(Brief
    rewritten 2026-08-05; a stale output schema — `lo/hi`, `hypothesis`,
    `n_per_route`, `min_lanes`, `supply` — was also corrected to the shipped
    `ArchetypeCell` fields `at_least/below`, `any_of`, `predicts`, `source`, so
    the generated file loads. Running an LLM against it is the open step.)*
    (e) *`predicts` stays a tested prior; selection and pruning are post-label
    on measured divergence, never pre-label on prior or supply.* The loop: LLM
    proposes rich cells → fill (natural draw + augmentation + constructed docs)
    → label → keep the cells that measurably diverge (dense/sparse top-10
    Jaccard as a pre-label screen; measured route-split after labels), merge or
    drop the rest.
    (f) *50/50 sparse/dense is an eval-time weighting, not a selection target.*
    Real decisive traffic is ~74% dense in the current mix; a balanced
    population would be unrepresentative and teach over-prediction of sparse.
    Training needs a per-class FLOOR — enough sparse decisive rows to learn the
    boundary — met by augmentation/generation, consistent with the
    Representative doctrine (balance applied at eval as a weighting).
    (g) *Cell-conditioned generation is the deferred spine — its own session,
    recorded here for a parallel effort.* The whole augmentation stack is
    parent-based (mutates an existing selection row); nothing generates a query
    from a cell spec. The capability the vision needs: given a cell's multi-band
    predicate, produce a coherent query hitting every band and — for a
    zero-natural-supply cell — the constructed document that answers it, from
    LLM knowledge, no parent. It leans on the constructed-docs / synthetic lane,
    which moves from edge-case to core under this design.
    — *The audit's real finding was not a coverage gap but a category error:
    judging cells by what the acquired corpora happen to supply. Fixing the
    false-positive predicates makes the sparse claims honest; expanding by
    mechanism (bounded by what the extractors can measure) and validating by
    measured divergence is how the cell set earns a defensible route mix without
    ever baking in the prior.*

60. **Ties are judgment-resolution artifacts, upgraded by an acceptability
    view, not by re-weighting the objective** (grill-me 2026-08-07).
    (a) *Diagnosis, measured on the cached lane oracles.* Of 15,037
    `all_tied` rows, **zero** have identical top-10 lists across the three
    routes; 96.1% share only the top-1 doc and differ in their (unjudged)
    tails; 70% sit in three median-one-judged-doc lanes (webfaq 60% tie
    rate, gooaq 45%, orcas 28%). A tie means the judgments ran out of
    resolution, never that the routes are equivalent. Objective
    re-weighting was separately measured dead (§9, all 42 lanes pooled):
    0.5/0.5 flips zero real labels vs shipped, bare NDCG@10 ~92 of 46K —
    the label is not an artifact of the 0.7/0.3 choice. min_relevance+1
    is degenerate (39 of 42 lanes have binary qrels; gold empties).
    (b) *Acceptability is a derived view, never stored columns.* Per route:
    `ok = score >= oracle − tolerance`, `oracle = max` of the row's three
    scores — computed by a view class over `labels.parquet`'s score vector,
    tolerance as its parameter. CONTEXT.md already rules `route` "a view
    over the scores"; materializing ok_* would create the twin-structure
    consistency bug CLAUDE.md forbids and bake one tolerance into the
    artifact. `tolerance=0` reproduces today's labels exactly — nothing is
    destroyed. (Sanity-check, compressed: the component is one small class
    computing three booleans from three stored floats; the rejected
    alternatives — new parquet columns, a sibling parquet per tolerance —
    both add an artifact to keep consistent. KEEP the view.)
    (c) *Canonical tolerance 0.3 = hit parity, derived not hand-picked.*
    0.3 is the objective's own `ndcg_weight`: the widest gap two routes
    can show while sharing the same top-1 outcome (a hit/miss difference
    forces ≥ 0.4 — same derivation style as d41's 0.4 decisive margin).
    The measured gap distribution confirms the band is real: runner-up
    gaps have median 0.022, p75 0.095, then jump to 0.811 at p90 —
    nothing lives between 0.3 and 0.4, and serve flips saturate at 6,827
    by tol 0.2. Accepted trade-off: tail quality alone (NDCG 1.0 vs 0.4,
    both rank-1 hits) never disqualifies a route — Top-1 parity is what
    the production router optimizes.
    (d) *all_zero stays null (d41 upheld), for a bias reason, not
    conservatism.* The view could express `[0,0,0]` ("no route
    acceptable") — argmax never could — but qrel holes are asymmetric
    (dense 14–32% vs BM25 ~6% on older pools), so a chunk of the 8,181
    all-zeros are fake and disproportionately fake against dense;
    training all-negatives on them injects exactly that bias. Revisit
    after (f)'s option A shrinks the fake-zero population.
    (e) *The training target is three binary acceptability heads.* One
    head per route on ok_dense/ok_rrf/ok_sparse; at inference, serve the
    cheapest route whose P(ok) clears a threshold. This keeps [1,0,1]
    distinct from [1,1,1], turns the 15,018 tied rows into sparse-positive
    training signal (7,789 genuine sparse wins + ties ⇒ the starving
    class is fed), and leaves cost policy in the inference rule, tunable
    without relabeling. The derived `serve` column is the evaluation
    oracle readout only — including for the production hard-classifier
    baseline.
    (f) *Follow-up sequence, decided: A next, C conditional, D last.*
    A = LLM-judge the differing tails (PPI-rectifier spine per
    docs/research/route-label-sourcing.md): ~14.4K tied queries ×
    ~15–20 unique unjudged tail docs, the honest tie-breaker, and the
    unlock for revisiting (d). C = harden webfaq/gooaq/msmarco corpora
    with adversarial distractors — only if A shows the ties are real
    (tails genuinely irrelevant ⇒ the corpus is too easy). D = augment
    tied parents into harder children — waits for A/C evidence on which
    perturbations break ties.
    — *The finding that pays for the whole grill: zero of fifteen
    thousand ties are irreducible. The information to break every one of
    them already sits in the cached rankings; only judgments are missing.
    The acceptability view is how the dataset stays useful while they
    are.*

61. **Admitted augmented rows reach evaluation by read-time supplement, not
    by mutating a lane's own snapshot** (grill-me 2026-08-10; closes the
    "labels.py merge of augmentation qrels" item open since d40's grill).
    (a) *Diagnosis: two files, one silent gap.* `cell_selection.parquet`
    records WHICH `query_id`s the composition wants scored; `QuerySubset`
    (retrieval/base.py) narrows a lane's OWN `queries()`/`qrels()` down to
    those ids but can only subtract, never add — a query_id absent from the
    lane's persisted `queries.parquet` silently disappears from evaluation,
    no error. Confirmed by grep: `CellFill.admit()` never touches any lane's
    `queries.parquet`. An admitted augmented row would have valid qrels
    (`AugmentationQrels`, d43d/d59-era `backfill()`) but no query text
    anywhere retrieval reads from.
    (b) *The qrels half of this was already designed for, never wired up.*
    `QrelStore`/`QrelSource` (hybrid_search_rrf_dataset/qrels.py) already
    merge judgments from multiple sources at READ time — `concat()` plus a
    declared trust precedence (`HUMAN > CONSTRUCTED > CLICK > LLM`) — and
    `QrelSource.CONSTRUCTED`'s own docstring names Inject rows explicitly.
    Nothing calls `QrelStore.from_dataset`-equivalent construction against
    `AugmentationQrels`'s data today; the mechanism exists, the caller
    doesn't.
    (c) *Query text follows the identical pattern: read-time, additive,
    never a snapshot mutation.* No corpus reindexing or re-embedding is
    ever needed — confirmed by reading `fusion.py`/`indexer.py`: dense and
    sparse both embed the query string LIVE inside `rank()`, nothing is
    precomputed or cached per query, and augmentation never adds documents,
    only queries pointing at documents already indexed. Given that, physically
    appending into each lane's `queries.parquet` would be the ONLY reason
    to ever touch it, and doing so risks silently losing augmented rows on
    the next `materialize_corpora.py` rebuild of that lane. Resolution:
    `QuerySupplement`, mirroring `QuerySubset`'s exact shape (wraps a
    `source: RetrievalDataset`, same constructor pattern) but ADDS rows to
    `.queries()`/`.qrels()` instead of narrowing them. Named "supplement"
    specifically over "overlay" (CONTEXT.md, d61): an overlay reads as
    covering/replacing what's underneath; a supplement only adds alongside
    it, which is the whole point — the lane's own snapshot stays exactly
    reproducible from scratch.
    (d) *Two generations of augmented row need different rules, so the
    inclusion criterion is parametric, not hard-coded.* `floor` distinguishes
    them by construction: a bare label (`marker:greeting`, `id:medical`, ...)
    predates the d51 cell rebuild; a `CELLS_BY_NAME` member postdates it.
    `generation: Literal["floor_based", "cell_based"] = "cell_based"` (named
    for what each IS, not a version number). `"floor_based"` includes every
    such pool row unconditionally — admission never applied to them (they
    predate `CellFill`; they are already gate-free, `credit_gate="none"`, so
    there was never a gate to clear either) — retrofitting an admission
    check onto them would invent a rule that never governed their creation.
    `"cell_based"` includes only rows whose `query_id` is already in
    `cell_selection.parquet` — the credit-gate/admission pipeline (d42h)
    stays the single source of truth for "counted," and a raw, unreviewed
    `pool.parquet` row (today, most of it) must not become silently
    evaluable.
    (e) *One shared filter feeds both halves, so text and qrels can never
    silently disagree about which rows are in.* A single function resolves
    `(pool, cell_selection, generation) -> filtered pool rows`; `QuerySupplement`
    reads `query_id`/`query` off it, the `QrelStore` builder reads
    `query_id`/`doc_id`/`relevance`/`source` off the matching `AugmentationQrels`
    rows for the same id set. Verified as a fact, not assumed: every
    `home_lane` value in the current pool matches a real dataset-registry
    name, so it is trustworthy as the lane key both pieces key off.
    (f) *Scope: both halves, one decision.* Query text and qrels are two
    views of the same filtered row set — building them separately now risks
    them drifting apart on which rows count as "in" later.
    — *The lane's own files are the ground truth for what it shipped;
    augmentation's contribution is additive and provable from
    `pool.parquet`/`qrels.parquet` alone. Nothing about "did this query get
    evaluated" should ever require asking "did someone remember to append it
    somewhere."*

62. **Opaque tokens defeat two measurements at once; both repairs are stated as
    rules and enforced by tests, never as named instances** (grill-me
    2026-08-10; extends d56's `NUM` removal and d50(c)'s cell guards, each of
    which repaired an instance and did not survive).
    (a) *Diagnosis: two independent defects on one path.* The archetype probe
    `a3f5d8b9e12c4d56789abcdef0123456` routes `dense_only`. Two inputs are
    wrong and either alone is sufficient. FEATURE: `natural_language_share`
    reads 1.0 — spaCy tags the unseen token `AUX`, a closed class, so a hex
    digest scores as pure grammatical glue, above real prose at 0.4–0.5.
    MEMBERSHIP: `bare_concept_token` claims it, the cell whose prior is
    `dense_only`, so the training data teaches the same thing the feature
    does. Repairing one alone leaves the probe failing.
    (b) *The tag is a guess on any token the tagger has not seen, and the guess
    is not systematic.* `deadbeefcafe1234` tags `NOUN` and is harmless; the
    probe's digest tags `AUX` and counts as glue. d56 diagnosed the `NUM` case
    as "not tagger noise, a definition mismatch" — true there, since UD really
    does file numerals closed-class. It does not cover `AUX`: UD files no hex
    digest as an auxiliary. The root cause is wider than d56 named, which is
    why removing one tag did not end it.
    (c) *The rule: a closed-class token must be word-shaped.* A closed class is
    closed, and no member of English's contains a digit — so a digit-bearing
    token is never a function word, whatever tag it carries. One condition at
    the counting site, stated about token shape rather than about `AUX`, so a
    sibling tag on a future unseen token is already covered. Alternatives
    measured and refused: excluding `AUX` (the whack-a-mole d56 already lost
    once); `token.is_alpha` (drops `'s` and `n't`, real function words out of
    contractions); `token.is_oov` (true for every token under
    `en_core_web_sm`, so it separates nothing).
    (d) *Cost is small in aggregate and decisive where it counts.* 18 of 4,000
    labelled queries change `nl_share` (0.45%), none by more than 0.15 — the
    same shape as d56's own 24.8%-inflated / 2.0%-band-crossing split. Only a
    query containing a digit can change value, so the rebuild filters to
    digit-bearing rows, a provable superset, instead of the full 380K.
    (e) *Scope is the natural-language bank alone, on evidence.* `MorphologyBank`
    reads 0.000 on opaque input. `SyntacticDepthBank` genuinely hallucinates —
    a bare UUID parses to `nesting_depth` 3.0, above a real question at 2.0 —
    but every cell banding a parser scalar already carries a
    `natural_language_share` floor, so the NL signal IS the guard for the
    parser banks and repairing it restores the gate. That convention holds in
    all four such cells today and nothing enforces it; (g) does.
    (f) *`bare_concept_token`'s predicate contradicts its own `looks_like`, and
    the repair must cover identifiers as a class.* The prose says "No digits,
    no acronyms, no code, nothing verbatim-rare — opaque jargon belongs to the
    identifier cells"; the predicate says only `length_words < 3` and
    `number < 1`, and cells.yaml's header rules the predicate the sole matcher.
    So it claims the digest, `ERR_CONNECTION_RESET` and `HTTP 502` alike. The
    trap to avoid: the cell DOES band an identifier absence, so a test asking
    "does this cell forbid an identifier?" passes while the bug stands — it
    forbids 1 of 54. The invariant is about coverage, which is why the class
    must be expressible as one column.
    (g) *Both rules are enforced by tests in `tests/test_cells.py`, because SPEC
    prose demonstrably cannot enforce them.* d50(c) guarded `uri_in_query` and
    `opaque_token_any_domain`; d50(d) regenerated the cell set, both cells
    stopped existing, and neither guard is among today's 44. Two tests, beside
    the 14 already there: a short cell (a `length_words` ceiling ≤ 10) that
    demands no identifier presence must band identifier absence as a class; and
    a cell banding any parser scalar must band a `natural_language_share`
    floor. The trigger reads length and identifier demand, never `predicts` —
    d48(d) makes that a falsifiable prior, and a prior must no more drive an
    invariant than it drives allocation.
    (h) *The identifier aggregate is derived, never stored.* One column summing
    the structured-identifier span counts, so the class is one band instead of
    54 and the invariant is expressible at all. Storing it would create a sum
    that can disagree with its parts after any bank change — the twin structure
    CLAUDE.md forbids — and would not even suffice: three producers build
    catalog-shaped rows, and `mini_catalog` builds them from `QueryFeatures`
    for generated children without reading the parquet at all. The derivation
    lives with the existing catalog-column convention and is applied at every
    producer; nothing is re-extracted.
    (i) *Sequencing: the cell side now, the bank side behind a green build.*
    (f)–(h) need no re-extraction and can land immediately; labels key on
    `(dataset, query_id)` and are reused on overlap (d49g), so a membership
    change costs labelling only for newly selected rows. (c)–(d) land in the
    nested `src/query-taxonomy` repo, whose round-trip suite is red (92
    failures at `6008f40`) — a change made against a red build cannot be shown
    to have broken nothing.
    (j) *The probes are the acceptance test, not the aggregate.* Both repairs
    are judged on `src/hybrid_search_rrf_dataset/probes.py` and
    `route_experiments.ipynb` §7, where the affected rows are visible. At 0.45%
    of queries and 3 of 44 cells, neither repair is expected to move a headline
    mean, and quoting one as evidence either way would be reading noise.
    (k) *Corpus indexing becomes incremental, because every repair path from
    here assumes it* (amendment, same session). `_index()` re-uploaded an
    ENTIRE lane whenever the collection held fewer points than the corpus —
    119,976 re-embeds to add one document to `crumb-code-retrieval`, O(n)
    embedding work per O(1) documents added. Harmless while every lane is a
    frozen snapshot, which is why it had never fired; load-bearing the moment
    d50(g)'s cell-conditioned generation writes its first constructed document,
    since that lane then grows every round. `BaseIndexer.missing()` diffs
    `item_id` — a deterministic `uuid5` of `doc_id`, so the building block was
    already there — against the collection and uploads only the difference; the
    point count stays as the cheap "did this corpus grow?" trigger. Accepted
    limitation, recorded rather than left to be discovered: identity is the
    point id alone, so edited text under an unchanged `doc_id` is not
    re-embedded — exactly the behaviour of the count check it replaces.
    — *Both prior repairs were correct, and both were lost: one to a sibling
    tag, one to a regeneration. What makes this decision different is not a
    better patch but that the rule outlives the artifact it was found on — a
    test fails loudly where a decision paragraph waits to be read.*

63. **The encoder router: frozen input channels, one latent, feed-forward
    privileged branches** (grill-me 2026-08-11; supersedes d46's encoder
    line and d47's LUPI-option-B item; evidence base:
    `docs/research/qpp-retrieval-routing.md`,
    `docs/research/lupi-privileged-information.md`,
    `docs/research/tooling-mlp-router.md` — a three-report literature pass
    standing in for arch-validator).
    (a) *Input = frozen bge-small embedding ⊕ hashed char-3–5-gram SVD
    (~128 dims, fit on training queries) — complete transforms, nothing
    curated.* bge-small supersedes d46's "NOT bge-small-en": a frozen
    encoder's weights never see labels, and the stack coupling lives in
    the labels themselves, so hiding the embedding would not decouple it.
    The d46 worry stays falsifiable, not vetoed: arm 7 trains the identical
    architecture on multilingual-e5-small; if bge wins only
    random_within_lane and loses holdout routes_differ, d46 was right and
    e5 ships. The SVD block is the complete lexical channel — every
    substring counts, so identifiers, digits, and casing survive with zero
    per-format judgment; completeness, not curation, is what makes an
    input unbiased. Evidence: lexical inputs beating frozen embeddings on
    route classes (RAGRouter-Bench), XGBoost-on-frozen-embedding beating
    fine-tuned DeBERTa at moderate scale (LTRR). Fine-tuning the encoder
    is arm 6, run once as the published ceiling (the BERT-QPP/CIKM'21
    regime starts ~500K labels; we have ~40K).
    (b) *One MLP encoder → latent z; two supervised branches whose outputs
    FEED the route layers (hallucination wiring, Hoffman 2016) — not
    dropped heads.* Cell branch ĉ: Linear(z→44) sigmoids, BCE with
    per-cell pos_weight, graded against per-row evaluation of EVERY cell
    predicate (multi-hot, recomputed offline at build time) — never the
    fill's stored single assignment, which bakes quota and tie-break
    bookkeeping into the geometry; overlapping multi-hot targets force the
    latent to encode shared archetype factors instead of 44 islands.
    Corpus branch â: z-scored MSE over three named blocks — per-lane
    corpus stats (corpus.parquet scan + taxonomy extraction over ~5K
    sampled docs per lane; retires d48i's CorpusIndex wish), per-query
    gold-doc stats (taxonomy over the row's qrel docs at the lane's
    min_relevance, plus query↔gold lexical overlap — per-query targets are
    what break the ~15-lane lane-ID degeneracy), and fold-local
    route-outcome stats (ok-rate per route + decisive share, computed from
    training rows only; named outcome stats because they are a property of
    corpus × selection, never of the corpus alone). Route layers consume
    concat(z, ĉ, â) at train AND serve: the model imputes at inference the
    privileged knowledge it cannot see, and the skip connection on z lets
    training down-weight unreliable estimates. The branch losses exist
    only at training — without them ĉ and â are anonymous hidden units;
    with them they are meaningful, inspectable estimates.
    (c) *Route output = the three d60 acceptability heads under the
    existing cheapest-acceptable serve rule*, so LR and encoder router are
    compared under identical serving.
    (d) *Training frame: every labelled row, masked route loss.* all_zero
    rows contribute zero route gradient (d60d holds) but still supervise
    both branches; all_tied rows supervise all five outputs; near-duplicate
    clusters (cos>0.95) never straddle a split.
    (e) *Seven arms, one eval harness.* (1) the design; (2) taxonomy
    features as extra input, corpus branch only — do features earn input
    status; (3) no branches — does branch supervision help at all;
    (4) shuffled branch targets — information or regularization, the
    TMLR-2025 mandatory control; (5) LightGBM ×3 on arm-2 inputs — is the
    MLP the right learner at this scale; (6) fine-tuned bge + 3 heads,
    once — the ceiling; (7) e5-small control — settles (a) empirically.
    Eval: leave-one-lane-out CV over every lane with the spread reported
    (the binding small number is ~15 lanes, not 40K rows), routes_differ
    as the headline slice (every published query-side predictor is weakest
    exactly where dense and sparse disagree), baselines const-dense /
    LR+priority / auto-fusion / production hard classifier, the 7
    archetype probes as the standing smoke test. Branch losses z-scored,
    λ annealed down (branch targets are ground truth from step 0, unlike a
    distillation teacher), gradient cosine-gated (Du et al. 2018) so a
    branch cannot hurt the route loss by construction, λ tuned on the
    route validation metric only.
    (f) *No feature column ever supervises the model.* The per-feature
    recoverability question ("can the embedding see identifiers?") is an
    offline linear probe on frozen inputs and latent — a diagnostic with
    zero gradient. This is where the curation-bias objection closes: the
    only human-designed structures that touch the weights are
    dataset-native — cells, gold docs, outcomes — each already policed by
    its own falsification loop.
    (g) *Sequencing: prototype on dataset v2 now; reportable numbers ride
    the v3 dataset build.* The catalog is stale in the exact columns cell
    predicates read (d56e/d62 re-extraction owed, blocked on the taxonomy
    repo's 92 red round-trip tests), so v2 cell targets are wrong for
    digit-bearing queries. Rather than a bug-fix rerun, the extractor
    repairs land inside the v3 build together with more data and better
    augmentation. v2 numbers are shakedown; v3 numbers are the arm
    comparison of record.
    (h) *Code shape: own package `src/encoder_router/` (the composition/
    precedent — user's call over the single-module recommendation),
    experiments surfaced through a notebook the user runs; deps: torch
    (direct pin), lightgbm.* Plain PyTorch with a hand loop — every
    tabular framework surveyed makes the branches harder, not easier.
    sanity-check 2026-08-11: KEEP — rungs 1–3 (production classifier, LR,
    AcceptabilityRouter) are measured below the ceiling this exists to
    lift; the GBDT rung is embedded as arm 5; revisit if arm 5 matches
    arm 1 on holdout routes_differ (then ship the trees).
    — *The through-line of the grill: every "which features?" question
    dissolved into "which dataset-native structure already owns this?" —
    cells over span columns, predicates over assignments, measured
    outcomes over hand priors. The model is supervised by the dataset's
    own artifacts, and the dataset was built to be exactly that.*

64. **Ceiling levers ride one re-measurement** (grill-me 2026-08-12; the
    "remaining levers" triage after the playground exposed the harness).
    (a) *Nothing is adopted until the arm sweep re-runs on the repaired
    harness.* This session's fixes invalidate every v3 panel number: early
    stopping restored epoch-0 weights on every fold (validation used
    unweighted BCE, which RISES as the pos_weighted training objective
    converges — the two losses disagreed about what "better" means), and
    serving/threshold-tuning carried a cost discount now removed (thresholds
    maximize raw captured score; serve = most probable head among those
    clearing thresholds, rrf hedge when none fires; cost never picks between
    heads). Re-run the 10-lane panel on a fresh results path, plus a
    learning curve (25/50/100% of fit rows, fixed val split) — hours of
    compute, zero new design, and every lever below reads its go/no-go off
    this readout.
    (b) *The serve-time constraint stays hard: the router sees the raw query
    string only.* No extractor, no corpus, no spaCy at inference — "we
    generalize features, we do not extract them": feature knowledge enters
    through privileged branches that teach the model to estimate at serve
    time what it cannot compute there. Bundled static data (a frequency
    table) is admissible; a runtime dependency is not. CONTEXT.md updated.
    (c) *Two new arms instead of the d63 proxy trigger.* `zipf_channel`:
    design inputs ⊕ query-local rarity scalars from a background-frequency
    table (wordfreq dep; min/mean/max token Zipf, share below a rarity
    cutoff, share absent from the table — the hex digest maxes the last).
    `feature_branch`: taxonomy features as a third privileged branch's
    TARGETS (the admissible form of arm 2, whose serve-time inputs violate
    (b)). Supersedes d63's deferred "rarity input channel if arm 2 > arm 1
    on sparse wins" — for the same compute the sweep measures both designs
    directly. Adoption rule: beats design on differ_agreement, checked
    specifically on sparse-win rows.
    (d) *Labels: the cheap repairs land BEFORE the re-run; the expensive one
    waits for it.* Now: raise beir-nfcorpus min_relevance (the §1b audit:
    95.3% of its qrels are grade 1, weak positives counted as full
    successes) and relabel that lane; fix the two Option A blockers
    (GoldenRoutingBuilder round-trip contradiction, stale nfcorpus oracle
    cache) — they are correctness debts regardless. Option A itself
    (LLM-judge tied-row tails, PPI spine) is spend, gated on the readout:
    if the new arms move differ_agreement, inputs were binding and A waits;
    if every arm stays flat under honest training, label noise is the prime
    suspect and A jumps the queue. Ordering is load-bearing: the relabel
    changes one lane's labels, so after-the-sweep would mix label regimes.
    (e) *Composition gated on the learning curve; "more data" means decisive
    rows in thin archetypes, never more of the same.* Flat by 50→100% ⇒ the
    lever closes this cycle (the v3 build continues on its own d63g gate).
    Still sloping at 100% ⇒ the buy order is d50(g) cell-conditioned
    generation — which means finally unblocking distractor borrowing — not
    more fat-lane natural rows, which mostly add ties. Bundled proxy, free
    with the re-run batch: a TIE_WEIGHT sweep (0 / 0.25 / 1.0) — if
    excluding ties helps validation, that is the same
    decisive-rows-matter-most hypothesis confirmed before any generation
    spend.
    — *The through-line: every lever already had a standing decision or a
    designed next step; what was missing was a trustworthy measurement to
    arbitrate between them. One re-run buys arbitration for all three.*

65. **Query authorship is a card fact, never inferred** (grill-me 2026-08-24).
    The v3 showcase reported 71,966 rows as "natural (real user query)" when
    52,343 of them are ScIRGen-Geo, whose own catalog entry calls it
    synthetic-but-filtered LLM generation, and 723 are LIMIT, constructed from
    a template. Nothing on `DatasetCard` recorded authorship, and `llm_target`
    does not: it describes the retrieval task's orientation and is `True` for
    CRUMB, whose queries no model wrote as far as anyone has checked.
    `DatasetCard.query_provenance: QueryProvenance` now carries it —
    `HUMAN | LLM | TEMPLATE | UNKNOWN`, **with no default**, so a new
    registration cannot pass silently as human-written. `UNKNOWN` is a
    ratification state in the sense of decision 7 (profiling proposes, a human
    ratifies), not a fallback: the eight CRUMB lanes hold it until someone
    reads the upstream card, and any consumer that groups by provenance shows
    them as unratified rather than folding them into `HUMAN`. Two consequences
    worth stating, because they are the reason the field earns its place:
    the published half of v3's supply is ~74% machine-written, so v3's claim
    was never "our queries are human" but "our minted queries are grounded in
    a real corpus document and their route labels were earned by retrieval";
    and a boolean would have been wrong, since LIMIT has no author at all.

71. **Per-lane card generator (`CardGenerator`) validated on 6 negative-bearing
    lanes; the binding constraint on further recall is definitional, not
    technical (2026-09-07).** `scripts/card_transfer.py` runs paired A/B
    (arm A: universal INSTRUCTION only; arm B: universal + `G(lane)` card) on
    crumb-clinical-trial, wands, dbpedia-entity, freshstack-{langchain,laravel},
    miracl-en-dev — the six referee lanes with both human positives AND
    negatives, so precision and recall are simultaneously measurable. On the
    two task-family twins for deploy lanes (dbpedia ~ quest, crumb-clinical ~
    crumb-legal-qa) G delivered **Δrecall +0.180 and +0.110 with zero
    card-caused false positives** — matching hand-card effectiveness (+13.0%
    mean on the 5 deploy lanes per d69). Aggregate across all 6 test lanes was
    only +5.3%, dragged down by 3 lanes where cards were structurally inert.

    Three-agent triangulation (fable, opus×2) on why the 3 lanes were inert:
    **~57% of the residual false-negative gap is definitional** (partial-grade
    matches the strict standard correctly rejects; loosening trades precision
    blind), plus ~13% judge literalism on numeric equivalence (miracl), plus
    ~5% lanes where the graded-relevance boundary is invisible to any
    metadata bundle (wands, freshstack-laravel gold is nugget-based). Only
    ~25–35% of the remaining gap on the twin lanes is bundle-addressable —
    and even a hand-tuned card cannot exceed what a strict, gate-safe rule
    can encode. Model upgrades are the wrong lever: G's cards already fire
    correctly per their input; the ceiling is what a strict card CAN say, not
    how well it's written.

    Bundle-content fix APPLIED (`card_generator._lane_metadata`): stratified
    query sampling across short/long × set-cue/no-cue buckets (dbpedia's
    "list of X" queries now surface), and up to 3 grade-0 negatives per query
    with a fallback that keeps trying candidates when the first doc_ids don't
    resolve to loaded corpus text (a silent zero-snippet bug the first pass
    hit on dbpedia). Post-fix per-lane negative snippet counts: wands 15,
    freshstack-{laravel,langchain} 15/13, crumb-clinical 11, dbpedia 1,
    miracl 0 (last two data-bound — grade-0 doc_ids point outside the loaded
    corpus). Expected additional lift on twin lanes: +5–8pp; the definitional
    residue remains unaddressed and is CORRECT-per-standard, not a defect.

    STANDING: the generator mechanism is safe on all 6 lanes (Δfpr ≤ +0.01,
    max 1 fp on 100 negatives); useful where a card is warranted; and its
    average lift across a mixed lane population is modest by construction
    because most lanes don't need cards. Deploy G on deploy lanes with this
    caveat: the aggregate lift claim is misleading — quote per-lane twin
    numbers instead. `wands`/`freshstack` recall gap is a strict-vs-topical
    definitional gap; do not "fix" it by loosening cards without changing
    what min_relevance selects. Artifacts:
    `src/data/relevance_judge/card_transfer/{predictions_{baseline,carded},summary}.parquet`
    and `generated_cards.parquet`.

72. **Dataset release shape — licence-conditional partition** (2026-09-10).
    The project ships deliverables beside the router. The first is the route-label
    dataset, released from `src/data/rungs/100k-v2/labeling/labels.parquet`
    (91,093 rows, 46 lanes). The release shape proposed in `docs/datasets.md`
    "Licensing and publication standing" is RATIFIED, with two amendments.

    What ships, partitioned by publication standing rather than concatenated —
    share-alike text in the same file as CC-BY material would relicense our own
    labels, which is the binding reason to partition:
    - **pointer artifact**, all shipped lanes — `(dataset, query_id)` plus our
      route measurements, our annotations under CC-BY-4.0, a per-lane
      attribution table, and a loader that rebuilds text upstream;
    - **`queries_permissive.parquet`** — text for the CC-BY / apache lanes,
      plus `wands` (MIT) from amendment 1;
    - **`queries_sharealike.parquet`** — the 9 SA lanes, marked CC-BY-SA-4.0;
    - **pointer-only** — the non-commercial and unlicensed lanes.

    *Amendment 1 — lanes outside the verified 42-lane matrix.* The artifact holds
    46 lanes; four were never verified. `wands` (479 rows, MIT) ships text;
    `finder` (5,703, CC-BY-NC-4.0) and `techqa` (621, content review pending) are
    pointer-only; `beir-touche-2020` (49) is EXCLUDED — its `ids` standing is
    itself unresolved and 49 rows do not justify a licence review.

    *Amendment 2 — minted rows inherit their parent lane's standing.* The matrix
    covered only sourced rows. Of 16,676 minted rows (15,256 synthetic, 1,177
    augmented, 243 doc_grounded), 15,327 derive from `quest` (apache-2.0) and
    ship with text; ~961 from restrictive parents stay pointer-only. Deliberately
    stricter than a paraphrase sharing no parent text requires — do not loosen
    without a per-operator analysis of surviving parent text, which has not been
    done.

    *Non-commercial lanes ship as pointers* — see `docs/adr/0003`. 12 lanes /
    22,346 rows (24.5%) carry non-commercial terms, and `docs/datasets.md` warned
    those terms reach our own use, not only redistribution. Resolved in favour of
    publication: a free HuggingFace research dataset, for a routing capability
    that is not a premium Qdrant feature, is non-commercial research use, and the
    pointer shape is the TREC-qrels pattern BEIR and MTEB already follow over
    these corpora. Dropping them would have cost 24.5% of the rows and biased
    what remained — the NC set holds nearly all the real user query traffic,
    leaving ~51% LLM-authored or minted-here. `antique` (222 rows) is the one
    exclusion: the Yahoo Webscope DUA forbids reposting, a prohibition rather
    than an absence.

    Net: **44 lanes** (46 − `beir-touche-2020` − `antique`). Publishing is
    one-way; a check with whoever owns OSS/legal precedes upload but blocks
    neither spec nor build. Card must disclose RAR-b's unlicensed upstream
    repackaging and the LLM-authored share.

73. **Demo v3: live graded answer keys via micro-pooling; budget is the only
    spend cap** (grill-me 2026-09-11, after three-way validation: Codex
    implementation lens, forked-session coherence lens, Opus product lens; gap
    analysis against the production goal "zero queries/qrels in, working eval
    system out"). Plan of record:
    `~/.claude/plans/cheeky-dreaming-alpaca.md`; the demo design doc gets a
    dated v2 amendment section.

    (a) *Welcome step:* the presenter sets a session budget — chips $0.50/$2/$5,
    default $2 — and it is the ONLY spend cap; `max_llm_calls` is removed. Time
    is guarded by deadlines (20s generate, 15s retrieve, 25s judge).

    (b) *Pooled judging:* after retrieval, the union of each strategy's top-10
    (metric validity over latency) is judged in ONE batched `/qrels/expand`
    call, auto-fired concurrently with the results render, with a Skip.
    Verdicts are namespaced `dataset=f"{collection}-demo"` — never the research
    lane name, which would pollute the shared verdict bank. `/qrels/expand`
    gains per-pair `verdicts` (aggregates cannot grade a pool).

    (c) *Grades:* presenter's grounding click → source doc 2/0 (grade 2 is
    human-only); judge positive → 1; judge negative renders "checked, no
    support" and counts as UNJUDGED (recall 0.308 — absence of evidence).
    Human–judge disagreement on the source: human grade stands, the judge's
    note renders with the recall caveat — never silent, never blocking.

    (d) *Scoring:* per-strategy Hit@1 / NDCG@10 / O (RouterObjective — the
    definition of record; grade ≥1 is a hit, empty qrels scores 0) rendered
    UNDER the three result columns; coverage as "N of 10 graded" (knowledge =
    human-judged + judge-positive only; the rest count as zero — a floor, not
    a score); divergence badge Δ = max(O) − min(O).

    (e) *Cross-process budget = conservative hold:* deduct a fixed
    `judge_allowance_usd` (0.05) before the judge call, settle to the actual
    response spend, FORFEIT the hold on a lost response — overspend and
    undercount impossible without receipt machinery; lock around session-budget
    mutations. Judge failure degrades to source-only scoring under the LIVE
    banner; a replay is never paired with live retrieval.

    (f) *Comparability (the one-way door):* every scored block carries its
    `retrieval_config` (collection, model ids, slot names, fetch depth,
    fusion) and `rerun()` APPENDS runs — identity cannot be retrofitted onto
    overwritten records. Named `retrieval_config`, not "fingerprint" (taken by
    the catalog heatmap view). Suite UI shows latest run + an ×N chip.

    (g) *Repair issues a new query_id* — text changed means cache identity
    changed (`/qrels/expand` keys on (dataset, query_id, doc_id)).

    (h) *Export:* `queries.jsonl` (minted text, ours) + `qrels.tsv`
    (doc_id, grade, provenance) — corpus text never leaves.

    (i) *Destination:* phase 4 = BYO collection (connect form, multi-session,
    per-collection judge referee sample — the 0.987/0.308 operating point was
    measured on OUR lanes and is inherited on faith until revalidated);
    phase 5 = production eval system: cold-start SEED benchmark minted by
    targeting CELLS directly (the /examine order sheet measures deficits of an
    existing query set — cold start has none; mintable ceiling today ~11 of 62
    cells), CLI batch runner over the importable runner with an explicit
    human-gate policy, config-comparison harness reusing the twice-calibration
    statistics, "directional, not a winner" reporting at N≈30–100, seed/real
    segmentation with the measured mint bias. Positioning: "first eval
    bootstrap for vector search configuration" — integration novelty backed by
    the S≈R transfer result (1.12), never claimed as new methodology.

## Deferred questions

- Register/box definitions for eval-time weighting + page-search log
  acquisition (d30; boxes wait for a real log).
- Judgment-shaped MODEL features (word-order sensitivity, syntactic depth,
  corruption degree...) — GLiNER2 classification head is the default
  candidate, but behavioral/perturbation designs may fit better; own
  session.
- Multilingual MODEL tier: GLiNER v1 `gliner_multi` vs future mDeBERTa
  GLiNER2 — revisit when non-English profiling matters.
- Corpus-relative features (IDF profile, vocabulary mismatch, ambiguity,
  specificity, answerability): requires explicitly reopening the registry's
  queries-only decision. **Promoted from deferral to blocker by d37(l)**;
  **resolved by d44(b) 2026-07-30** — six collection statistics computed
  from lane corpus parquets, registry untouched (consistent with d39a).
  The richer candidates (vocabulary mismatch, ambiguity, specificity)
  reopen only if the d44(c) transfer pilot's captured headroom stalls.
- msmarco corpus scale-up past 100K if margins look corpus-limited —
  recipe is a parameter (d38c), embedding cache amortizes the retry.
- StatRewrite entries beyond (length, up) — reopen when a band goes
  hungry; a "telegram queries feel underrepresented" instinct is a
  recipe/band question first, demand second (d42d).
- Augmenter LLM model + cost envelope; retry-budget size — set at d42
  implementation, informed by the Decorate pilot batches.
- Hungry-floor rung assignment (d42f) — pending the supply-scan readout
  (augmentation_supply.ipynb, three cells left to run).
- Realism overrides over the d34b defaults: which features need them is
  discovered empirically from seeded round-trip samples, not decided up
  front.
- Model backstop for CODE_FRAGMENT/MATH_EXPRESSION recall (symbol-light
  formal content: "x squared plus y squared", prose pseudo-code) — layered
  bank; needs a code/math detection model choice (d20).
- Attested search-syntax extensions to OPERATOR_SYNTAX (quoted phrases,
  minus-exclusion, `site:`) — attested in query logs but precision-dangerous;
  own decision (d20).
- Rung 2 under d51 (d43's inversion: any lane doc carrying the surface, not the
  parent's own gold doc). A key minted against a doc that carries the surface
  but does not answer the parent's need is weaker than rung 1 — decide when
  rung 1 runs dry, not before.
- `AugmentationCampaign.pilot_n` staging under cell demand (d51). Credit gates
  are per-operator, so one cell served by Inject stages a pilot while the same
  cell served by Decorate does not — unexamined in the d51 grill.
- Whether `operator:` is removed from cells.yaml or kept as a human override of
  d51(c)'s derivation. Two sources of truth is what went stale; a veto path for
  a mechanically-valid-but-semantically-wrong mint has no home without it.
- Inference fallback when no acceptability head clears its threshold (d60e) —
  a serving policy (cheapest? abstain-to-rrf?), decided at router-training
  time, never encoded into labels.
- `[0,0,0]` for all_zero rows (d60d) — reopens only after option A's tail
  judging shrinks the fake-zero population enough to measure the dense-hole
  bias instead of assuming it.
- `GoldenRoutingBuilder` parquet round-trip inconsistency: ~147 of 46K rows
  have `route_rankings` that contradict their stored `route_scores` (traced
  live on gooaq 139935 — score implies the gold doc in top-10, rankings lack
  it). The §9 noise floor; harmless to d60's view (reads labels.parquet, not
  oracles) but must be fixed before option A judges tails from those rankings.
- Stale `beir-nfcorpus_oracle` cache: 323 upstream queries vs the 12 the
  composition selects — rebuild or delete before any oracle-pooled readout is
  quoted as exact.
- Enumerated function-word lexicon replacing the POS test entirely (d62c).
  A closed class has finite membership, so membership could be looked up
  rather than inferred — immune to every tagger guess, alpha or not. Costs a
  hand-maintained vocabulary and redefines the signal from POS-based to
  lexicon-based. Reopen on evidence, not taste: a pure-alpha out-of-vocabulary
  token observed landing in a closed class (0 of 13 sampled).
- Stale `cell` values on already-labelled rows after a predicate change (d62f).
  `RouteLabels.CARRIED` copies `cell` onto every label at `label()` time, so
  rows labelled under the old `bare_concept_token` predicate keep a membership
  the cell no longer claims — 217 of 46,856 measured. Harmless to the scores,
  which are measured facts and must NOT be deleted when a row leaves a cell:
  three retrieval runs bought them, leaving a cell does not make a measurement
  wrong, and a later predicate may claim the row back. No refresh is needed
  today and none should be built: the canonical per-cell readout
  (`scripts/cell_divergence.py`) reads `cell` off `cell_selection.parquet` and
  merges rankings on `(dataset, query_id)`, exactly as `label()`'s own comment
  says — so rebuilding the selection makes every joined readout correct and the
  carried column is a stale copy nothing consults. Reopen only if something
  starts reading `labels["cell"]` directly.
- PFD teacher-student wiring (teacher sees query+corpus, student distills
  its logits) — the literature's strongest-evidenced privileged-information
  alternative (Taobao, +5% online), deliberately not a v1 arm. Reopen if
  arm 1 fails to beat arm 3 (d63b).
- Prototype/contrastive cell shaping (learned archetype anchors in latent
  space instead of the linear multi-hot branch). Reopen if the cell branch
  flatlines — per-cell AP ≈ 0 despite pos_weight (d63b).
- Sparse-leg/Zipf rarity stats as a third input channel — RESOLVED by
  d64(c) 2026-08-12: the conditional trigger is replaced by a direct
  `zipf_channel` arm in the re-run sweep; adoption reads off its own
  measured readout, not the arm-2 proxy.
- If BOTH d64(c) arms (`zipf_channel`, `feature_branch`) beat design:
  ship one, or both — a combined arm needs its own run to attribute the
  gain before it ships.
- Whether the label-side serve oracle (d60e cheapest-acceptable at
  tolerance 0.3) should also drop cost, now that router serving is
  quality-only (d64a) — differ_agreement currently compares a cost-free
  router against a cost-aware oracle, so a residual disagreement band is
  structural, not model error.
- d50(g) distractor borrowing — opened only if the d64(a) learning curve
  still slopes at 100%; flat curve keeps it closed this cycle.
- Multilingual serving: bge-small is English-only; arm 7 previews the
  encoder swap, but real multilingual routing also needs multilingual
  probes and a decision on ngram-SVD script coverage (d63a).
- Gold-doc aggregation for multi-qrel rows (v1 default: mean over docs at
  the lane's min_relevance) and the ~5K-docs-per-lane sampling size —
  implementation defaults; revisit only if per-lane variance is large
  (d63b).

66. **v4 composition validation: dials pinned against Layer 2 evidence, no
    code-implementer handoff** (grill-me 2026-08-26). Resolves the eight
    open dials of the v4 plan (`~/.claude/plans/i-feel-like-we-scalable-
    phoenix.md` §9) using Layer 2 measured evidence
    (`notebooks/v4_substrate_replay.ipynb`) against the shipped Rung B
    skeleton (`src/composition/composer_v4.py`, `tests/test_compose_v4.py`,
    13 passing property tests, 1 skipped for Rung A). No prior SPEC decision
    superseded — each plan-local dial reinterpreted to remain compatible with
    d48(e), d49(b), d49(c), d50(f). The plan itself explicitly rejects
    code-implementer handoff; this entry records the dial resolution against
    measured evidence, not a build-order.
    (a) *Acceptance = paired Layer 3 pilot, §6 is a regression bar not a
    verdict.* Lane-equal aggregation at n ≥ 100 per lane (per
    `VERDICT.md:97`), three arms: constant-router null, matched-random draw,
    v4 draw. Aggregate = mean of per-lane deltas. The §6 shortcut gate stays
    as a build-to-build regression check between v4 iterations. The plan
    itself retracted its "worse than random" claim after realizing §6 was
    underpowered as a verdict.
    (b) *N* is a label-spend ceiling, not a target (d49(b) preserved).*
    Composition stops at the tighter of `{all cells filled, N* labels
    spent}`. Working ceiling N* = 200,000; size |D| remains an output,
    reported via `class_shortfall` and `waste_shortfall` on
    `CompositionReport`.
    (c) *κ is a global lane-share safety cap at 0.20 (d49(c) preserved).*
    Must always be ≥ max(cell.max_lane_share); per-cell caps carry the load,
    κ backstops.
    (d) *π is a composition share, deliberately narrower than d50(f).*
    Midpoint {dense: .45, sparse: .45, hybrid: .10} allocated by
    `_largest_remainder`. The router training layer applies d50(f)'s
    per-class floor separately — carve-out, not supersession.
    (e) *ρ = 0.5 for large-|S_a| axes; flat f_s = 25 for axes with |S_a|
    < 10.* Layer 2 measured: at ρ = 0.5, only 12 of 63 cells reach f_s =
    1,587 → 81% generation debt. Consistent with d50(a): thin cells are
    generation targets. The small-|S_a| fix is already in the skeleton.
    (f) *θ₀ = 0.7, Δ = 0.1 (Rung A novelty, pre-registered).* Layer 2: 95%
    stratum survival at θ = 0.7, Δ relaxation almost never fires. Rung B
    skeleton does not use these — deferred activation until Rung A
    `order()` ships.
    (g) *π is a band, ±5pp uniformly, on the direct `route_class_any`.*
    dense ∈ [40, 50], sparse ∈ [40, 50], hybrid ∈ [7, 13]. Band-pass status
    reported in `CompositionReport`; the composer never sacrifices floors
    to hit an exact share. Consistent with d50(f) — balance is corrected
    at eval-time weighting.
    (h) *Rung A class allocation via per-candidate propensity, not lane-
    yield.* P̂(c|q) via character n-gram Jaccard kNN over the labelled pool
    (k = 20, weighted vote, `max_sim < 0.15` = class-unknown, excluded from
    E[n_c(B)] used for the (g) band check). Lane-yield ŷ_c(l) retires from
    class allocation; lane budgets remain supply-driven. R0-compliant: the
    candidate's own outcome is never referenced, only neighbors' frozen
    labels — a finer-grained aggregate prior than lane-yield, same shape.
    (i) *U_lo = 0.15, U_hi = 0.40 on U(D) = (1/|D|) Σ clip(m/m*, 0, 1).*
    Symmetric protection: catches all-zero (U → 0) and cherry-pick (measured
    U = .455 for high-margin policy) with equal ~6pp margin; realistic
    policies (.217–.341, SD .048) sit safely in-band. U is a gate, not a
    maximand — the composer never optimizes it.
    (j) *δ = 0.10, 25 of 39 lanes clear AUC ≥ 0.60, held out by lane.* If
    the `information_gain_proxy` (G, N, Π → 1[m ≥ m*]) fails the gate, kill
    the term; Rung A falls back to §2.6 lexicographic order (already the
    default, so failure is not catastrophic). Pre-registering δ prevents
    base-rate hacking after the pilot lands.
    (k) *Stability test on the Rung B skeleton: perturb ρ ∈ {0.3, 0.4}
    (one-sided; 0.5 is the (e) ceiling, no symmetric upper perturbation
    exists) and κ ∈ {0.15, 0.25} around the baseline (ρ=0.5, κ=0.20); pass
    if mean Jaccard(D_baseline, D_perturbed) ≥ 0.80 AND min ≥ 0.70 across
    the four perturbations.* Runs on an n=600 synthetic pool, not the n=240
    Layer 1 fixture: at n=240 the κ=0.25 cascade drops Jaccard to 0.558 as
    a small-pool artifact (cap changes shift a large fraction of a tiny
    row set), whereas at n≥600 the numbers converge (κ=0.25 → 0.845 stable
    through n=2400). θ perturbation deferred until Rung A ships. Lives in
    `tests/test_stability.py`.
    — *The plan is not a SPEC decision; it explicitly rejected
    code-implementer handoff. Rung A `order()` implementation, catalog_v3
    unlabelled-query extension (plan B0), Rung A propensity index, band
    reporting on `CompositionReport`, stability test wiring, and the Layer
    3 pilot are tracked in TODOS. Nothing here authorizes a build without
    a further grill-me on the Rung A scope.*

67. **Tie-breaking judge work-list: all tail docs, one function, framed as
    qrels depth not router gain** (grill-me 2026-09-03). Replaces
    `JudgeQueue.sub1_pairs()` with a single work-list method covering every
    tied row. Supersedes nothing; implements the work half of d60 Option A
    (TODOS "Option A (next)"), whose PPI-rectifier half stays deferred and
    whose list-preference framing is dead under arch5k §10.

    *Why the scope was wrong.* `above_gold()` documents itself as "the only
    docs whose relevance can break a sub-1.0 tie" — true for the HitRate@1
    lever, false for the objective as a whole. A perfect (1.0/1.0/1.0) tie
    breaks through the `0.3·NDCG@10` term when any tail doc is judged
    relevant, verified on the real objective (exact tie -> margin 0.0505).
    `sub1_pairs()` filtered `res["sub1"]`, so all 1,832 perfect ties were
    structurally excluded: the 85-vs-1,917 gap was a code artifact, not a
    mechanism limit. Measured on the arch5k draw's 1,595 perfect ties,
    **0 are unbreakable** (matches TODOS' earlier "0 of 15,037 have
    identical top-10s").

    (a) **Candidate set = every route's top-k, minus gold, minus docs at the
    identical rank in all routes.** 15.9 docs/row (25,439 for the draw).
    Docs in all routes at *differing* ranks discriminate too (verified,
    margin 0.0368), so the narrower "union − intersection" set (14.3/row)
    drops ~12% of valid breakers for no reason. Same-rank-everywhere docs
    are provably inert: they add an identical DCG and IDCG increment to
    every route, so a tie stays a tie — one rule, valid for sub-1.0 and
    perfect ties alike.

    (b) **One function, not two.** `sub1_pairs()` deleted; the new method
    serves all 1,917 residual tied rows. Two work-list builders would be two
    candidate-set definitions to keep consistent (a design bug per
    CLAUDE.md), and `sub1_pairs()` was under-serving even its own 85 rows by
    offering only above-gold docs when tail docs break sub-1.0 ties too.
    `above_gold()` itself is retained — the pilot notebook uses it for the
    l2 above-gold diagnosis and three tests cover it — with its overclaiming
    docstring corrected. Breaking callers to update:
    `src/scripts/run_relevance_judge.py:95` and the pilot notebook's
    `queue.sub1_pairs()` cell.

    (c) **Static ranked work-list, no adaptive early-stop.** Return all
    candidates as one inspectable frame ordered by rank-spread (widest
    route disagreement first), stable-tiebroken by doc_id for determinism.
    At $0.00038/judgment the draw's 1,917 tied rows yield ~30,300 judgments
    = **~$11.50** (measured: 15.8 pairs/row over a 200-row build; the 25,439
    figure quoted during the grill covered only the 1,595 perfect ties, not
    all tied rows). That EXCEEDS the config's `max_spend_usd = $10`, so a
    full pass either raises the guard or shards via `--limit` — the guard
    truncating is safe, not a crash. Early-stop would save ~half and cost a
    redesign (judge<->scorer interleaving couples the queue to `PilotScorer`
    and destroys the auditable work-list). Truncation comes
    free from `--limit` and the spend guard, and banked judgments make
    partial runs resumable. Full v2-100K would be 393,540 judgments = $148,
    at the bottom of d60 Option A's own $140–700 estimate.

    (d) **Framed as dataset supply (qrels depth), never as router gain.**
    A perfect tie keeps HitRate@1 = 1 for every route, so only the
    `0.3·NDCG` term can move and the maximum reachable margin is 0.3 —
    below `RouterObjective.decisive_margin` (`hit_weight − ndcg_weight` =
    0.4). Measured over 360 sampled rows: every row breaks, best achievable
    margin is **0.1161** (median = p90 = max), and **0% reach 0.4**. So
    deep-judging moves rows all_tied -> low_margin and adds **zero** rows to
    the router's decisive training set (`router.py:304`). Recording this in
    the method docstring and here, because the two thresholds disagree:
    `residual.regime()` calls 0.116 `decisive_strong` (its bar is a
    hand-typed 0.1) while the router rejects it (bar 0.4) — a ledger read as
    router gain would be wrong. Consistent with
    `docs/research/relevance-judge-recovery.md` pre-registering the judge as
    supply, not router improvement.

    (e) **Binary relevance retained; graded deferred to a binary ladder.**
    Judged atoms stay `relevance ∈ {0,1}`. Binary is sufficient — with a
    second relevant doc, binary gold already separates NDCG@10 1.0000 vs
    0.6131 — and it is what the 0.955-precision gate was measured on, so
    grading would invalidate the validation and need a new per-grade gate.
    LLMs are also more reliable on binary than on multi-point scales. If
    depth measurement later shows grades matter, the upgrade is a ladder of
    binary questions (rung 1 "relevant at all?", rung 2 "fully answers?"),
    whose first rung is the question already asked — so existing atoms stay
    valid and nothing is re-judged.

    (f) **No objective change now; the dependency runs the other way.**
    99.6% of perfect ties carry exactly one relevant doc, at rank 1 in
    100% of routes. Under that condition HitRate@1, NDCG@10, MRR, Recall@10
    and MAP all equal exactly 1.0, so **no weighting of this metric family
    can separate the routes** — which is why previously-tried objectives
    moved all_tied nothing. Depth is the only lever, so the judge is the
    precondition for evaluating an objective, not the reverse. The
    "tier-2 rank-win certificate" (separating a rank-1 flip from a
    ranking-quality win, so margins in (0, 0.3] become expressible) is
    deliberately NOT specified here: its threshold must be derived from
    post-depth measurement, per the computed-not-assumed rule. Tracked in
    TODOS.

    *Not authorized here: the tier-2 certificate, any objective reweighting,
    graded atoms, the PPI rectifier, and scaling beyond the draw. Each needs
    the step-2 depth measurement first.*

68. **Judge run hardening + the tie-depth pilot result: ties break, none reach
    decisive** (session 2026-09-03). Implements d67 and reports what the first
    paid run bought. Supersedes no decision; d67(d)'s prediction is now measured
    rather than argued.

    (a) **Measured payoff — the mechanism works, and buys only depth.**
    5,133 atoms over 354 rows of the arch5k residual. Rescoring the v2 rankings
    under `human ∪ judged` gold:

        before \ after   all_tied  low_margin  decisive_strong
        all_tied              189          71                0
        low_margin              1          88                5

    **71 of 260 tied rows (27.3%) broke** — ties are NOT encoder-invariant, and
    the NDCG lever is real. But **0 of those 71 reached `decisive_strong`**; all
    landed in `low_margin`, exactly as d67(d) predicted from the 0.3-vs-0.4
    ceiling. The 5 decisive rows came from `low_margin`, never from a tie.
    `qrels_hole_rate_rows` = 35.6% — a third of judged rows had a relevant doc
    missing from qrels. Published `tie_conversion_rate` (0.209) UNDERSTATES it:
    its denominator is all 354 rescored rows, including the 94 never tied.

    (a2) **The true-l2 run closes the stack caveat and splits the tie mass in
    two.** Re-retrieving l2 rankings live (no v2 proxy) over the sub-1.0 tie
    population — all 85 of them — gives 80 tied before, **18 broke (22.5%), of
    which 2 reached `decisive_strong`** and 16 `low_margin`. That is not a
    contradiction of (a); the two runs cover complementary halves, and the
    mechanism predicts both:

    | population | share of `all_tied` | tied | broke | -> decisive | -> low_margin |
    |---|---|---|---|---|---|
    | perfect (score = 1.0) | 1,832 (95.6%) | 260 (v2) | 71 (27.3%) | **0** | 71 |
    | sub-1.0 (score < 1.0) | 85 (4.4%) | 80 (l2) | 18 (22.5%) | **2** | 16 |

    A perfect tie has gold at rank 1 in EVERY route, so `HitRate@1` is already
    maxed and only the `0.3*NDCG` term can move — capped below the 0.4 bar, hence
    0. A sub-1.0 tie does not, so judging an ABOVE-GOLD doc relevant flips
    `HitRate@1` and the margin can clear 0.4 — hence 2. So d67(d)'s ceiling holds
    exactly where it was argued (perfect ties) and the `above_gold` lever d67(b)
    kept is what produces the only decisive rows.

    **Router yield is still negligible: 2 decisive rows out of 1,917 tied rows
    (0.1%)**, or ~26 rows if scaled to v2-100K's 24,751 ties. The program remains
    **qrels depth / dataset supply and nothing else**; any ledger reading these
    conversions as router gain is wrong.

    (b) **Sparse discovers more missing gold than dense** — discovered-relevance
    rate 6.3% sparse / 6.5% rrf / 5.0% dense. A small counter-signal to arch5k
    finding 4's 87%-dense skew, in the direction the os_distill upgrade
    ([[project-leg2-sparse-bakeoff]]) would push further. Not a router claim.

    (c) **Reasoning suppression is the whole latency story; the prompt paid for
    the precision.** luna is a reasoning model. Sent only as nested
    `extra_body={"reasoning":{"effort":...}}` it reasoned anyway at ~60s/call;
    adding the top-level `reasoning_effort` param drops it to **~1.2s (50x)**.
    Precision then moved 0.955 (full reasoning) -> **0.905** (suppressed,
    verdict-first prompt) -> **0.932** (`ASKED:` evidence line before the
    verdict) -> **0.988** (instruction rewritten from a measured failure brief,
    `docs/research/judge-prompt-brief.md`), at `anchor_recall` 0.7225 over the
    0.6 floor. So the accuracy lost to suppression was recovered by prompt
    design, not by paying 50x latency — and the winning version beats full
    reasoning. Verdict-FIRST is specifically wrong for a non-reasoning judge:
    it commits before examining, then rationalises.

    (d) **Pricing was 2x over-budgeted.** Engine rates corrected to OpenRouter's
    published card for `openai/gpt-5.6-luna`: **$0.20/M in, $1.20/M out** (were
    0.40/1.60 placeholders the config itself flagged as provisional). The full
    tie run re-costs from $13.10 to **~$7.00**, inside the existing $10 guard, so
    d67(c)'s "raise the guard or shard" question is moot. A `:batch` variant
    halves it again and was REJECTED: at 19 min and $7 it cannot repay an
    async submit/poll code path.

    (e) **Four paid-path defects, each with a regression test.**
    - `judge_pairs` and `judge_rows` both flushed their verdict buffer AFTER the
      loop, so a `BudgetExceeded` arriving through `windowed_map`'s
      `future.result()` discarded up to 99 (resp. 49) already-PAID verdicts.
      Both wrapped in `try/finally`.
    - `judge_one` caught only `TRANSIENT_PROVIDER_ERRORS`, so any other provider
      error abandoned every remaining pair — observed killing a run after 2 of
      3,600. Now catches broadly (re-raising `BudgetExceeded` alone, which MUST
      stop the run) and counts by exception name.
    - A reply cut off at the token cap loses its `VERDICT:` line, parses to
      nothing, and is discarded: **249 paid pairs (5.2%) lost in one run**. The
      cap is now a config knob (128 -> 256) AND `finish_reason == "length"`
      triggers ONE retry at double the cap; non-truncated garbage is not retried
      (it would only pay twice). Re-running recovered all 249 with 0 unreadable.
    - Swallowed errors were invisible, which is why a 60s/call provider looked
      like a hung run and a format regression looked like a provider blip.
      `RelevanceJudge.dropped` / `.dropped_detail` / `.unreadable` now count by
      cause, keep the first message per type, and sample raw unparseable replies;
      both run paths print them. `judge_rows` also stopped discarding its `Spend`
      and now reports cost live on the bar and as `spend_usd` in the report.

    (f) **The validation gate is a TRANSFER estimate, and says so.** The 12
    negative-bearing referee lanes and the 6 lanes the judge is applied to are
    **disjoint** — the deploy lanes ship positive-only qrels, so precision is
    structurally unmeasurable there (a judged-relevant unjudged doc is
    indistinguishable from the qrels hole the judge exists to fill). `score()`
    now emits `precision_is_transfer_estimate`, `referee_lanes`,
    `deploy_lanes_refereed` / `_unrefereed`, and a random-corpus pseudo-negative
    false-positive rate measured ON the deploy lanes — kept strictly OUT of every
    gate metric (a `pseudo` flag), because random docs are far easier to reject
    than human-judged near-misses and pooling them would inflate precision
    exactly where it is least earned. This bounds gross over-calling only; the
    near-miss boundary, where a relevance judge actually fails, stays unmeasured
    on the deployment population.

    *Known-unfixed: neither litellm's per-call `timeout=` kwarg nor the
    module-level `litellm.request_timeout` bounds a call on this OpenRouter path
    (a `timeout=30` call was measured completing at 60.9s; a probe with the
    global set to 90 ran past 120s). There is currently NO working per-call
    ceiling, so one stalled request parks a worker indefinitely. At ~1.2s typical
    this is low-impact but unresolved — tracked in TODOS.*

69. **Per-lane task context for the judge: recall +46% at no precision cost from
    the cards** (session 2026-09-05). The judge applied one question-answering
    definition of "relevant" to lanes whose task is not question-answering.
    Reading the deploy-lane false negatives showed five DIFFERENT failures, not
    one recall problem — which is why no single lever had presented itself.

    (a) **The diagnosis, per lane.** Recall against human gold on the six lanes
    the judge is applied to ranged 0.055–0.980. The failures decompose:

    | lane | recall | what was actually wrong |
    |---|---|---|
    | quest | 0.165 | queries are boolean set expressions; the judge read OR as AND |
    | finder | 0.105 | terse analyst shorthand; gold SUPPLIES figures, judge demanded the analysis |
    | clerc | 0.265 | query is a legal passage TRUNCATED at the citation point — no question is asked |
    | scirgen-geo-en | 0.185 | query GENERATED FROM the gold record; gold is "derived-from", not "answers" |
    | crumb-legal-qa | 0.055 | gold is wrong (attorney-fees query paired with a violations-against-elderly statute) |
    | rarb-math | 0.980 | works; its 4 misses reject solutions with wrong arithmetic |

    The quest case was a comprehension bug, not strictness: "2004 Italian novels
    or about secret societies or books by Eco" -> "It is an Eco book, but lacks
    the other requested attributes" -> REJECTED. 56 of 167 false negatives were
    explicitly disjunctive queries rejected for satisfying only one alternative.

    (b) **Root cause was in the universal contract, not the lanes.** `ASKED: the
    ONE thing the query needs` plus `MISSING: whatever THE QUERY needs that the
    document never states` made a disjunctive query structurally unanswerable —
    a document satisfying one of "A or B or C" put A and B in MISSING, and the
    mechanical rule "MISSING names anything -> no" rejected it. No per-lane card
    can override arithmetic. Fixed by (i) `ASKED` naming the single alternative
    THIS document could satisfy, and (ii) rebinding `MISSING` to ASKED rather
    than to the whole query ("an alternative ASKED did not name is not missing").

    (c) **`relevance_judge/lane_context.py`** — a per-lane card of five fields
    (task / queries / gold / judging / benchmark). Rendered into the SYSTEM turn
    after the universal rules, so a card NARROWS what counts as relevant for a
    collection and never loosens the standard of evidence. `benchmark` is stored
    for humans and deliberately NOT sent: it never changes a verdict and would
    cost tokens on every call. The card is hashed into `prompt_hash`, or two
    lanes' atoms would record identical provenance. `crumb-legal-qa` is
    DELIBERATELY uncarded — no card can be written without rationalising labels
    that look simply wrong, and a missing card falls back to the universal
    instruction, which is the safe default.

    (d) **Measured on a full n=3,600 sample, both stages carded.**

    | lane | before | after | change |
    |---|---|---|---|
    | clerc | 0.265 | **0.590** | +0.325 |
    | quest | 0.165 | **0.465** | +0.300 |
    | finder | 0.105 | 0.205 | +0.100 |
    | scirgen-geo-en | 0.185 | 0.245 | +0.060 |
    | rarb-math | 0.980 | 0.980 | 0.000 |
    | crumb-legal-qa (uncarded) | 0.055 | 0.075 | +0.020 |
    | **deploy pooled** | **0.292** | **0.427** | **+0.135 (+46% rel)** |

    Precision 0.9882 -> 0.9683 (8 false positives, gate 0.95). **All 8 are in
    `freshstack-*` lanes and NONE in a carded lane** — carded lanes are
    positive-only and structurally cannot produce a false positive, so the cards
    are provably free on precision and the ~0.02 cost is attributable solely to
    the universal (b) edit. The uncarded `crumb-legal-qa` barely moving is the
    control: it confirms the gains came from the cards, not run-to-run drift.

    (e) **Both stages must send the same cards.** `judge_rows` (the gate) and
    `judge_pairs` (which writes gold) call one `judge_one`, and both pass the
    row's `dataset`. Had only the banking stage been carded, the gate would
    certify an instrument that never runs — the same structural error as the
    disjoint referee/deploy lanes, self-inflicted. Tested per-row: five lanes in
    one batch produce five distinct system prompts.

    (f) **Transport faults are retried, not discarded.** One run lost 2,115 of
    3,600 pairs to `OpenrouterException - [Errno 61] Connection refused` and
    reported a MEANINGLESS precision of 1.0 — the drops fell hardest on
    `freshstack` (10% sampled), the lanes carrying 83% of all false positives,
    so the hard cases were removed by the failure mode. A refused connection
    reaches no model, costs nothing, and is pure lost work: `connect_retries=3`
    with linear backoff now recovers it. Request-level rejections are NOT
    retried (they would fail identically). Matched on the message, because
    litellm wraps the socket error and erases its type. `llm_workers` 32 -> 12,
    and litellm's per-error stderr banner is suppressed — at 32 workers it
    buried the run's own output while telling us nothing.

    *Caveat that travels: precision is still measured only on the 12 uncarded
    referee lanes, so the cards' effect on precision is unmeasured, not proven
    zero — the argument is structural (positive-only lanes cannot produce a false
    positive), not empirical.*
