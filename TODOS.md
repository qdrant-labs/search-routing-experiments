# TODOS

One section per SPEC decision. Sections whose decision was removed from
SPEC.md on 2026-08-17 went with it — the generation campaign, the
logistic-regression router, and the extractor-repair programme (that code
now lives in the query-taxonomy sibling repo). They are at commit 26b9a93.

## Ceiling levers — one re-measurement arbitrates all three (2026-08-12, SPEC decision 64)

Design closed via grill-me. Context: this session's harness repairs
(val loss now uses the training objective — the old unweighted-BCE val
metric RISES as pos_weighted training converges, so every fold restored
epoch-0 weights; cost dropped from thresholds and serving) invalidate every
number in `arm_results_panel_v3.parquet`. `src/router_playground.ipynb` is
the new per-run instrument.

Before the re-run (ordering is load-bearing — the relabel changes labels):

- [x] RESOLVED (verified 2026-08-20): `lanes.py` has `beir-nfcorpus` at
      `min_relevance=2` (`6b8bf4b`).
- [x] RESOLVED (verified 2026-08-20): `GoldenRoutingBuilder` round-trip
      contradiction — checked all 138,426 route-scores across every
      `*_oracle` cache under current code (re-score `route_rankings` via
      `RouterObjective`, compare to stored `route_scores`): **0 mismatches**.
      Fixed by `5d05ec1` (2026-08-19, predates this TODO's next re-read),
      which scores the persisted `ordered()` list instead of the raw
      ranking dict. Unblocks Option A regardless of the LLM-judge spend
      decision.
- [x] RESOLVED (verified 2026-08-20): `beir-nfcorpus_oracle` cache is 12
      rows, matching the current selection (not the stale 323).
- [ ] Implement the two new arms (d64c): `zipf_channel` (wordfreq dep;
      query-local rarity scalars as input block) and `feature_branch`
      (taxonomy features as third privileged-branch TARGETS — the
      serve-safe form of arm 2).

The re-run batch (readout arbitrates every lever):

- [ ] 10-lane panel sweep, all arms incl. the two new, fresh results path.
- [ ] Learning curve: 25/50/100% of fit rows, fixed val split
      (playground). Flat by 50→100% ⇒ composition lever closed this cycle;
      still sloping ⇒ buy decisive rows in thin archetypes (d50g), never
      fat-lane mass.
- [ ] TIE_WEIGHT sweep 0/0.25/1.0 (playground) — free decisive-rows proxy.

Decision rules on the readout:

- [ ] Input arms: adopt whichever beats design on differ_agreement,
      checked on sparse-win rows; both win ⇒ combined arm gets its own
      attribution run first (deferred question).
- [ ] Option A (LLM-judge tied tails, PPI spine): arms move ⇒ waits;
      all arms flat under honest training ⇒ labels are prime suspect,
      A jumps the queue.
- [ ] Serve-time constraint stays HARD (d64b): raw query string only —
      generalize features via branches, never extract at inference.

## Encoder router (2026-08-11, SPEC decision 63)

Design closed via grill-me + compressed sanity-check (KEEP; flips if
LightGBM matches the design arm on holdout routes_differ — then ship the
trees). Evidence reports in docs/research/. Prototype against dataset v2
now; the arm comparison of record waits for the v3 dataset build.

- [ ] `src/encoder_router/` package (own package, composition/ precedent):
      training-table builder (bge embeddings, char-3–5-gram SVD fit on
      train queries, predicate multi-hot targets, â target blocks), model
      (encoder → z; cell branch ĉ = 44 sigmoids, BCE + per-cell pos_weight;
      corpus branch â = z-scored MSE; route heads on concat(z, ĉ, â) —
      feed-forward wiring, d63b), hand training loop (λ annealed down,
      Du-et-al cosine gate), LOLO-CV eval harness reused by every arm.
- [ ] New offline artifacts for â (no LLM spend): per-lane corpus profile
      (corpus.parquet scan + ~5K-doc sampled taxonomy extraction — retires
      d48i's CorpusIndex), per-query gold-doc profile + query↔gold lexical
      overlap, fold-local route-outcome rates (training rows only).
- [ ] `src/encoder_experiments.ipynb`: arms 1–7 as user-run cells, per-cell
      AP monitoring for ĉ, offline linear probe (per-feature
      recoverability — diagnostic, zero gradient, d63f).
- [ ] Deps: torch (direct pin), lightgbm (arm 5), via poetry.
- [ ] GATE for reportable numbers (d63g): the v3 dataset build — taxonomy
      round-trip fix → d62 word-shape guard → re-extraction → predicate
      re-eval, bundled with more data and better augmentation. Every v2
      run is shakedown only and must be labelled as such.
- [ ] Deferred (mirrored in SPEC): PFD teacher-student if arm 1 ≤ arm 3;
      prototype/contrastive shaping if ĉ flatlines; multilingual serving;
      gold-doc aggregation + sampling-size defaults. (The rarity-channel
      trigger is superseded by d64c's direct `zipf_channel` arm.)

## Augmented rows reach evaluation via read-time supplement (2026-08-10, SPEC decision 61)

- [x] DONE 2026-08-10: `QuerySupplement` (retrieval/base.py), `_augmented_rows`
      shared filter + `generation` param wired into `RouteLabels.label()`
      (labels.py). Tests in tests/test_labels.py. Closes the "labels.py merge
      of augmentation qrels" item open since d40's grill.
- [x] `src/scripts/label_routes.py` — terminal sweep (index + label every
      lane, cheapest corpus first), ports notebooks/route_labels.ipynb §16-17.
- [x] `provenance` column on every labelled row (2026-08-10): threaded
      natively through `RetrievalDataset.provenance()` → `QueryContext` →
      `FusionRow`, not bolted on in `label()` — so `BaselineDataset`/
      `HybridRoutingDataset` get it too, for free.
- [x] FIXED same day: `_augmented_rows`'s `floor_based` branch never checked
      `credit_gate`, so 240 gated rows (the exact ones behind the deferred
      audit above) leaked into `labels.parquet` unreviewed for
      beir-nfcorpus/crumb-legal-qa. Caught by eyeballing real provenance
      values post-label, not by test — tests/test_labels.py now has a
      regression case. Both lanes re-labelled clean; the other 18 lanes
      from the interrupted floor_based sweep were never touched by the bug
      (they hadn't run yet).
- [x] FOUND AND FIXED later same day: the SAME leak had already reached 8
      more lanes (bright-theoremqa-questions, crumb-tip-of-the-tongue,
      crumb-theorem-retrieval, bright-aops, crumb-paper-retrieval,
      rarb-math, crumb-set-operation-entity-retrieval, quest) — 66 gated
      rows, from a run against the code before the fix above landed. Purged
      by query_id (gated in the pool AND absent from `cell_selection.parquet`
      -> never legitimately admitted). Separately, the naive
      `provenance.fillna("natural")` backfill (done to close the "how do we
      know" gap) wrongly overwrote 580 real augmented rows that had NaN
      provenance for the same reason — corrected from the pool's own
      provenance column, the actual source of truth. Lesson: a blanket
      fillna over a column two independent pipelines write into is exactly
      how a real augmented row silently becomes indistinguishable from a
      natural one — the next such backfill needs to match on `query_id`
      against `pool.parquet`, never against nulls alone.
      Final clean state: 46,856 rows, 714 augmented (all decorate/floor-based
      — zero cell-based rows have been labelled yet, `--generation
      cell_based` hasn't been run to completion).
- [x] `label()` made incremental (2026-08-10): skips per query_id already on
      disk instead of per whole dataset — `--force` still means "redo
      everything," the default path now only scores query_ids missing from
      `labels.parquet` and appends, so adding a handful of new augmented
      rows to an already-labelled lane no longer re-ranks the whole lane.
      `RouteLabelSweep.run()`'s own dataset-level pre-skip removed (it would
      have bypassed this and required `--force` for no reason). Verified
      live: re-running an already-labelled lane is now a ~7s no-op instead
      of a full re-rank.

## Cell-based gated rows admitted WITHOUT the audit (2026-08-10, deliberate)

User decision: don't wait for the audit tool below — admit the 377 gated
cell-based rows now, validate later. `CellFill().admit()` was called against
an in-memory copy of the pool with `credit_gate` forced to `none` for rows
where `floor.isin(CELLS_BY_NAME)` — `pool.parquet` on disk was NOT touched,
so it still honestly records these as unaudited (`coherence_gate` /
`declaration_audit`). 308 of 377 were actually admitted (69 hit ordinary
per-cell shortfall/lane-share caps, same as any admission — nothing to do
with the bypass). `cell_selection.parquet`: 57,653 → 57,961 rows, across 15
lanes (beir-nfcorpus, crumb-code-retrieval, crumb-legal-qa, rarb-code,
crumb-clinical-trial, rarb-math, crumb-set-operation-entity-retrieval,
crumb-tip-of-the-tongue, bright-aops, trec-dl-2022, crumb-paper-retrieval,
antique, bright-theoremqa-questions, crumb-stack-exchange, scirgen-geo-en).

- [ ] These 308 rows carry real routing labels once `label_routes.py
      --generation cell_based` runs, but their underlying meaning-
      preservation/coherence claim has never been human-checked. If the
      audit tool below eventually rejects a declaration or a floor, its
      rows already in `labels.parquet` need to be pulled back out — this is
      the concrete cleanup debt the skip creates.
- [ ] Re-running this same override is needed for any FUTURE campaign batch
      that stages more cell-based gated rows — nothing here is a permanent
      pipeline change, it's a one-off decision to redo each time until the
      audit tool ships.

## Acceptability view over route labels (2026-08-07, SPEC decision 60)

Design closed via grill-me. Ties are judgment-resolution artifacts (0 of
15,037 have identical top-10s); the upgrade is a derived acceptability view
(`ok = score >= oracle − 0.3`, hit parity), three binary training heads,
all_zero stays null. Vocabulary in CONTEXT.md (Acceptability label, Serve
decision); full rationale in SPEC d60.

- [x] Implement the view class — DONE 2026-08-07: `AcceptabilityLabels`
      (labels.py) + `RouteLabels.acceptability()`; default tolerance derived
      as `RouterObjective().ndcg_weight`, never typed. Must-pass test
      (`tolerance=0` reproduces stored `route`) + per-shape cases in
      tests/test_labels.py.
- [x] Heads router + notebook comparison — DONE 2026-08-07:
      `AcceptabilityRouter` (router.py, 3 binaries on ok_*, serves cheapest
      clearing threshold; no-head-fires fallback = most probable, the SPEC
      deferred policy) + route_experiments.ipynb §6 (readout table vs serve
      oracle / argmax router / constants; smoke-verified end to end).
      Threshold tuning and the auto-fusion baseline row still to run/judge
      in the notebook — user-driven.
- [ ] **Option A (next)**: LLM-judge the differing tails of tied rows,
      PPI-rectifier spine (docs/research/route-label-sourcing.md) —
      ~14.4K queries × ~15–20 unjudged tail docs, sampled not exhaustive.
      Both hygiene blockers below are RESOLVED — this is now blocked only
      on the spend decision (~$140–700), not on code.
- [x] RESOLVED (verified 2026-08-20, see the ceiling-levers section above):
      0/138,426 round-trip mismatches under current code.
- [x] RESOLVED (verified 2026-08-20): `beir-nfcorpus_oracle` is 12 rows.
- [ ] **Option C (conditional)**: harden webfaq/gooaq/msmarco corpora with
      adversarial distractors — only if A finds the ties real.
- [ ] **Option D (last)**: augment tied parents into harder children —
      after A/C show which perturbations break ties.

## From cell-quality grill (2026-08-05, SPEC decision 50)

This session, in order:

- [ ] Predicate repair (SPEC d50b), at the bank + a re-extraction:
      UUID bank keeps the 32–64 hex digest branch but rejects
      single-repeated-char / low-entropy runs (regex negative-lookahead if
      edify exposes backreferences; else bank post-filter + an `OVERRIDES`
      generator entry). DateTime bank drops the bare-epoch branch (ISO 8601
      only). Rebuild `catalog.parquet` (380K rows); generator↔bank round-trip
      test must stay green. IP-vs-version left irreducible; email left as-is.
- [x] Predicate repair DONE 2026-08-05 (SPEC d50b): UUID gained a
      `(?!(.)\1*\b)` negative-lookahead (edify exposes backrefs; no OVERRIDES
      needed — sampler skips `assert_not`); DateTime bare-epoch branch deleted.
      Submodule bank tests 366 pass. Catalog rebuilt via
      `feature_table.py --force` → 440,534 rows × 75 cols.
- [ ] FOLLOW-UP (makes d50b + d50c effective): re-run the fill against the
      rebuilt catalog — `CellFill(...).build(force=True)` — cell_selection.parquet
      and every selection_audit number are stale (built from the old 380K
      catalog). Then re-run `src/selection_audit.ipynb`.
- [ ] FOLLOW-UP: the round-trip gate `tests/test_taxonomy_generators.py` is RED
      independent of d50b (92 failed / 3 passed on clean baseline). Cause is the
      key-dialect skew already logged under the taxonomy-generators grill:
      `BANKS_BY_FEATURE` keys on bare `str(bank.name)`, `generator.feature`
      returns `"{group}:{name}"` → KeyError. The gate is meaningless until fixed;
      d50b's banks were verified by 50-surface self-heal sampling instead.
- [ ] Two cells.json guards (SPEC d50c), no re-extraction: `uri_in_query`
      + `length_words below 15`; `opaque_token_any_domain` drops
      `http_status_code` from its `any_of`.
- [x] Jaccard coverage audit script (SPEC d50e) — DONE 2026-08-05,
      `src/scripts/cell_divergence.py` (self-check + ruff clean). Globs
      `data/route_labels/*_oracle/rows.parquet`, joins cell_selection, computes
      per-cell dense/sparse top-10 Jaccard, sorts most-divergent first.
- [ ] BLOCKER for d50e to be meaningful: persist `route_rankings` during
      labeling. `RouteLabels.label()` (labels.py:136) builds GoldenRoutingBuilder
      rows that contain the per-route top-10 doc lists but writes only the score
      columns, discarding the rankings — so only `beir-nfcorpus_oracle` has them
      (11 of 27,145 cell rows). Fix: save the golden rows per dataset
      (`GoldenRoutingBuilder(...).save(out_dir/f"{key}_oracle")`, as the lone
      nfcorpus file was made), then re-run labeling. Until then the divergence
      screen scores ~0% of cells.
- [x] Improve the cell-generation brief (SPEC d50d) — DONE 2026-08-05.
      `docs/composition-cells-prompt.md` rewritten: cut `MEASURED FACTS` (d48h
      label-stat contamination) and `SUPPLY` (supply-as-validity error d50a);
      dropped `BUDGET`/`n_per_route`; restated the pooled-vs-within-corpus
      confound as a principle; added the cross-group INTERACTION + statistical-
      permutation emphasis; fixed the STALE output schema (was `lo/hi`,
      `hypothesis`, `n_per_route`, `min_lanes`, `supply` — now `at_least/below`,
      `any_of`, `predicts`, `source`, matching `ArchetypeCell`, which would have
      failed validation). Subsumes the d48h deferral.
- [ ] Run the revised brief through an LLM (separate task) → new static
      `cells.json`; validate it loads via `composition.cells._load`, then
      re-run `src/selection_audit.ipynb` against the new set.

Deferred:

- [ ] Cell-conditioned generation — the spine, own session (SPEC d50g).
      Given a cell's multi-band predicate, generate a coherent query hitting
      every band and, for zero-supply cells, the constructed document that
      answers it — from LLM knowledge, no parent. New capability; the current
      augmentation stack is entirely parent-based. Leans on the constructed-docs
      / synthetic lane (promote from edge-case to core).
- [ ] Taxonomy-extractor backlog (SPEC d50d): archetypes the proposal pass
      wants but no current feature measures (entity specificity, compound-noun-
      with-common-parts, …). New banks; only after the conjunction space of
      existing features is exhausted.
- [ ] Post-label divergence pruner (SPEC d50e): after labels exist, keep the
      cells whose measured route-split diverges, merge/drop the rest. `predicts`
      is never an allocator.

## From golden-set grill (2026-07-28, SPEC decision 37)

Next actions, in order — (1) and (2) block everything else:

- [ ] Lift the registry's queries-only restriction — **re-scoped by
      d39(a)**: labeling never needed it (18 lanes acquire doc-side via
      retrieval.py snapshot classes; 8,020 rows labelled with the cache
      untouched). Remaining customers: corpus-relative features (d37l,
      possibly servable from per-lane Qdrant indexes instead) and ORCAS
      clicks (parked click lane). Refill only when one of those fires.
- [ ] Pool extraction: run the three routes over a query list and emit
      `(dataset, query_id, route, rank, doc_id, score)`. No qrels needed,
      so it does not touch `FusionBuilder`. `_iter_queries`' skip-on-no-
      qrels guard is correct and stays — removing it would fabricate a
      `dense_only` label per unlabelable query.
- [x] Unanswerable-query outcome, remaining half (d37i) — RESOLVED by
      d41 (2026-07-29): all_zero rows carry `route = null` in
      labels.parquet; the shape column already shipped.
- [x] Deliberate tie-break rule, replacing incidental list order —
      RESOLVED by d41 (2026-07-29): route = cheapest among the
      tied-best (sparse < dense < rrf); quality first, cost only
      between exact ties. Implementation tracked in the d41 section.
- [ ] Retrieval latency benchmark for the three routes. Speed is one of
      three stated requirements and is entirely unmeasured;
      `data/engines.csv` benchmarks taxonomy extractors, not retrieval.
      Now also validates d41(c)'s assumed sparse < dense cost order
      (rrf > both components is structural); a flip re-derives the
      route column in seconds, touching only all_tied + dense+sparse
      tie rows.
- [ ] Per-dataset corpus sizing. Indexing cost is uncorrelated with row
      yield: trec-dl needs 138M docs for 79 rows, miracl ~33M for 530,
      dbpedia ~4.6M for 400, while msmarco is 8.8M for 15,678.
      `TrecDL2022(30000).materialize()` (judged docs only) is the existing
      pattern, but a corpus of near-answers inflates dense — d37(g).
      **msmarco settled by d38(c)**: judged-relevant + uniform-random
      distractors to 100K; remaining datasets still open.
- [x] Pre-retrieval corpus-relative features (d37l) — RESOLVED by
      d44(b) 2026-07-30: six collection statistics (avg/max query-term
      IDF, OOV share, N, avgdl, vocabulary overlap) from lane corpus
      parquets; implementation tracked in the d44 section. Post-
      retrieval ones (score margin, candidate overlap) still cannot
      inform which retrieval to run.
- [ ] Hole-filling decision (d37k) — deferred until the raw per-route
      hole rate is measured. Self-validating: if it helps, dense gains.
- [ ] Tests for `objective.py`, `qrels.py`, `golden.py` — no coverage
      exists. Highest value: `RouterObjective.score` at the
      `min_relevance` boundary, `QrelStore.lookup` conflict resolution,
      the `build_or_load` objective-mismatch guard.

## From msmarco-anchor grill (2026-07-28, SPEC decision 38)

- [x] The run itself — DONE 2026-07-29: 7,697 rows labelled into
      `labels.parquet` (52% routes_differ, 2% all_zero; 927 decisive,
      84% dense). Finding: 53% of routes_differ rows are top-two ties
      whose label is argmax list order — feeds the tie-rule and
      label-form decisions.
- [x] Label form: argmax one-hot vs per-route score vector — RESOLVED
      by d41 (2026-07-29): score vector canonical (regression lean),
      `route` a derived serving decision, decisive parameter-free
      (margin ≥ 0.4 ≡ winner hit@1 ∧ runner-up missed).
- [ ] msmarco corpus scale-up past 100K if margins look corpus-limited;
      recipe is a parameter, embedding cache amortizes the retry.
- [ ] ORCAS click-lane labeling (other 31% of composition): clicks are
      `source='click'` relevance (d37d) — own decision, own session.

## From corpus-conditioned-routing grill (2026-07-30, SPEC decision 44)

- [x] Headroom readout as a permanent route_labels.ipynb section
      (d44a) — DONE 2026-07-30: §18 (markdown caveats + code cell) over
      three new RouteLabels methods (`headroom_decomposition`,
      `headroom`, `decisive_winners`); decisive margin now derived from
      the objective's weights (`Objective.decisive_margin`, inf for
      non-lexicographic configs), never hand-typed. Verified against
      disk: 0.481 / 0.511 (+6.3%) / 0.557 (+9.0%); split
      1,858/145/507. Tests: tests/test_labels.py (4 green).
- [x] `collection_features.parquet` builder (d44b) — SUPERSEDED by
      d47 (2026-08-03): compute machinery moves to
      `query_taxonomy/corpus_relative/` (`CorpusRelativeBank` family),
      parent repo owns the `CorpusIndex` build + writer. Tracked in
      d47 passes 2 + 3.
- [x] Transfer pilot (d44c) — SUPERSEDED by d47 (2026-08-03): the
      6-config ablation with per-collection z-score is d47 pass 5,
      gated on the 16-row side test (d47 pass 4). Raw scale dropped
      as SPEC d47(g) rationale (magnitudes not comparable across
      corpora).
- [ ] List-preference judge spike (d44d): ~500 stratified rows from
      nfcorpus / crumb-legal-qa / rarb-math; query + stored top-10 per
      route (route_rankings + corpus parquets, zero retrieval);
      strongest current model, order-swapped double ask, ties allowed;
      agreement vs empirical routes on decisive rows; thresholds
      ≥80% open the scalable-labeling path / 60–80% judge panel /
      <60% the number goes to the CTO conversation.
- [ ] Deferred with triggers (d44e): SEARA-style per-deployment
      auto-benchmark (trigger: d44c positive + a per-customer
      consumer); interleaving on page-search (trigger: the log
      acquisition below in the composition-fill section).

## From label-form grill (2026-07-29, SPEC decision 41)

- [ ] Implement (code-implementer): `derive_route` + cost order in
      fusion.py next to `StrategyName` (one place, no import cycles);
      `GoldenRoutingBuilder` drops the list-order `max()`; labels.py
      writes `route` via the rule with null on all_zero; decisive
      readouts switch to margin ≥ 0.4.
- [ ] Migration after wave 1 completes: one pandas re-derivation pass
      over labels.parquet (route recomputed from the stored score
      columns — no retrieval); §15/§17 readout cells re-read under the
      new rule (quality headlines over decisive rows only).

## From qrels-acquisition grill (2026-07-29, SPEC decision 39)

- [ ] Pass 1 (code-implementer): per-lane RetrievalDataset classes for
      the 5 HF families (RAR-b, BRIGHT w/ excluded_ids, crumb, quest,
      limit) + dbpedia-entity/miracl via ir_datasets; `LANES` table;
      SnapshotDataset corpus-pending relaxation; `coverage()` gains
      `qrels_ready`. Output: `data/<lane>/{queries,qrels}.parquet` × 18
      + per-lane qrels_ready counts in the notebook.
- [ ] Pass 2 wave 1: index + label the ≤100K technical lanes
      (rarb-code, crumb-code, bright-*, crumb-theorem/legal/clinical/
      stack-exchange/paper, rarb-math, crumb-set-operation,
      crumb-tip-of-the-tongue, limit) — threshold rule d39(e), argmax
      parity d39(h).
- [ ] Pass 2 wave 2: quest → dbpedia-entity → miracl-en-dev, capped
      100K per d38(c) recipe.
- [x] Re-read the route distribution + margins over all landed lanes —
      DONE 2026-07-30 as the d44(a) headroom readout: global constant
      0.481 / per-collection constant 0.511 / oracle 0.557 (+15.8%
      ceiling); decisive winner split dense 1,858 / sparse 507 /
      rrf 145. Permanent notebook cell tracked in the d44 section.
- [ ] Re-price wave order if pass 1 shows a lane's qrels_ready is thin
      (msmarco precedent: declared QQ ≠ per-query coverage).

## Opaque-token repairs: word-shape guard + identifier-class band (2026-08-10, SPEC decision 62)

Design closed via grill-me. Two independent defects put a hex digest on the
dense route: spaCy tags unseen tokens into closed classes (`nl_share` 1.0 on a
32-char digest), and `bare_concept_token` bands only `number` out of 54
identifier columns. Both prior repairs of this class (d56, d50c) were recorded
against named artifacts and did not survive. Acceptance is the probe table, not
an aggregate.

Cell side — no re-extraction, lands first:

- [x] Derived identifier-span aggregate column, summed from the
      `structured_identifiers.*` columns, applied at both catalog-shaped
      producers (`CellFill._catalog`, `mini_catalog`). Derived, never
      stored (d62h).
- [x] `bare_concept_token`, `keyword_telegram_short`,
      `short_grammatical_question` band identifier absence as a class; the
      now-subsumed `number` band goes. `bare_concept_token` also gains a
      `sentence_markers.acronym` guard (`HTTP 502`).
- [x] `test_a_short_concept_cell_admits_no_identifier` — a cell with a
      `length_words` ceiling ≤ 10 that demands no identifier presence must band
      the aggregate. Trigger reads length + identifier demand, never `predicts`.
- [x] `test_parser_scalars_require_a_natural_language_floor` — any cell banding
      `nesting_depth` / `statement_count` / `widest_list_size` must band an
      `nl_share` floor. Passes on all four such cells today; the point is that
      the next regeneration cannot drop it.
- [x] `test_band_columns_exist_in_the_catalog` reads through the derivation, not
      the raw parquet, or the new column fails it.
- [ ] Re-run `CellFill().build(force=True)` — writes `cell_selection.parquet`,
      NOT the retired d32 `selection.parquet`, and `build()` returns the
      cached file unless forced. Labels are reused on
      overlap (d49g), so only backfill rows cost retrieval.
- [x] Carried `cell` staleness on stored labels — NOTHING TO DO, verified
      2026-08-10. 217 of 46,856 rows (0.46%) carry a `cell` the predicate no
      longer claims, but `scripts/cell_divergence.py` (the d50e readout) reads
      `cell` off `cell_selection.parquet` and merges on `(dataset, query_id)`;
      nothing reads `labels["cell"]`. Rebuilding the selection is the fix. Do
      not add a refresh method for a column no analysis path consults.
- [x] Incremental corpus indexing (d62k) — DONE 2026-08-10.
      `BaseIndexer.missing()` diffs point ids and uploads only the difference;
      `_index()` keeps the count as the cheap grew?-trigger. Was: one new
      document re-embedded the whole lane (119,976 docs for `crumb-code-
      retrieval`). Never fired while lanes are frozen snapshots; fires every
      round once d50(g) generation writes constructed documents.
      `tests/test_indexer.py` pins the second-pass-uploads-nothing case.
- [x] Corpus re-embedding for d62 — NONE NEEDED, verified 2026-08-10. `_index()` guards
      on `client.count(collection) < len(corpus)` and `ensure_collection()`
      no-ops when the collection exists; d62 changed no corpus. All 42 selected
      lanes are already labelled, so already indexed. Re-check after the rebuild
      with the selected-minus-labelled lane diff: non-empty means a lane needs
      full corpus indexing.

Bank side — BLOCKED on `src/query-taxonomy` going green (92 round-trip failures
at `6008f40`):

- [ ] Word-shape guard at the closed-class counting site (`metrics/pos.py`): a
      digit-bearing token is never a function word. Comment states the rule, not
      `AUX`.
- [ ] Rebuild the catalog for digit-bearing queries only — a provable superset
      of the 0.45% that can change.
- [ ] Re-check the probe table; `nl_share` on the digest must read 0.0.

Deferred by the grill — do not start (full text in SPEC "Deferred questions"):

- Enumerated function-word lexicon replacing the POS test (d62c). Reopens on
  evidence only: a pure-alpha out-of-vocabulary token observed landing in a
  closed class. 0 of 13 sampled.
- Stale `cell` values on rows labelled under the old predicate (d62f). Scores
  are unaffected; any per-cell readout over existing labels reads the old
  partition until a rebuild.

## Query provenance on the card (2026-08-24, SPEC decision 65)

Field shipped, notebook consumes it, two ratifications evidenced
(`scirgen-geo-en` → LLM, `limit` → TEMPLATE, both from `docs/datasets.md`).

- [ ] Ratify the eight CRUMB lanes. They ship `UNKNOWN`, so any provenance
      readout shows ~8,855 v3 rows as unratified. The `jfkback/crumb` card
      documents the tasks, not who wrote the queries; `llm_target=True` on
      that card is the task-orientation flag and says nothing. Offline read of
      the upstream card only — never a smoke fetch.
- [ ] `quest` is HUMAN on the strength of "3,357 natural queries"
      (`docs/datasets.md:60`), but the same entry mentions augmented splits.
      Confirm the augmented rows are not in our snapshot, or split the card.

Surfaced by the showcase rewrite, not caused by it:

- [ ] `min_zipf` / `mean_pmi` are NaN on ~698 rows (1.5%). NaN fails every
      band silently, so the rarity and PMI strata under-count by that much.
- [ ] Corruption is computed twice and the two disagree on ~3% of rows: the
      catalog columns include the TOKENIZER engine, `attach_strata`'s
      re-extraction does not. The showcase reads the catalog's derived total;
      which pass is authoritative is undecided, and a corruption FLOOR must
      not be enforced before it is.
- [ ] `LaneOrder.build` returned a column-less frame when nothing was owed —
      an empty boolean LIST selects columns, not rows. Fixed at the filter
      (one ranked index instead of a mask) with a zero-residual case in
      `tests/test_lane_carrier.py`. Left here because the same shape may exist
      elsewhere: grep for `frame[[` over a comprehension.
