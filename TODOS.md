# TODOS

Gates / next actions:

- [ ] POS domain-shift diagnostic (d14 guardrail): one-off
      natural_language_share ↔ stopword-ratio correlation over the cached
      datasets — natural to run with the first full profile.
- [ ] `src/scripts/benchmark_engines.py` argparse guard: ANY invocation
      (even `--help`) runs the full benchmark and appends rows to
      engines.csv — bit us 2026-07-21 (two stray rows scrubbed by hand).

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
- [ ] Unanswerable-query outcome, remaining half (d37i): the `shape`
      column ships (`labels.py` `outcome_shape`; nfcorpus anchor: 77/323
      all_zero), but the `route` column still fabricates a label for
      all_zero rows via tie-break list order.
- [ ] Deliberate tie-break rule, replacing incidental list order. Note
      `dense_only` is likely not the cheapest route — BM25 needs no
      query-side transformer pass.
- [ ] Retrieval latency benchmark for the three routes. Speed is one of
      three stated requirements and is entirely unmeasured;
      `data/engines.csv` benchmarks taxonomy extractors, not retrieval.
- [ ] Per-dataset corpus sizing. Indexing cost is uncorrelated with row
      yield: trec-dl needs 138M docs for 79 rows, miracl ~33M for 530,
      dbpedia ~4.6M for 400, while msmarco is 8.8M for 15,678.
      `TrecDL2022(30000).materialize()` (judged docs only) is the existing
      pattern, but a corpus of near-answers inflates dense — d37(g).
      **msmarco settled by d38(c)**: judged-relevant + uniform-random
      distractors to 100K; remaining datasets still open.
- [ ] Pre-retrieval corpus-relative features (d37l): query-term IDF in
      the index, out-of-vocabulary rate. Post-retrieval ones (score
      margin, dense/sparse candidate overlap) cannot inform which
      retrieval to run.
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
- [ ] Label form: argmax one-hot vs per-route score vector as training
      target — decide after msmarco margins land (d38g).
- [ ] msmarco corpus scale-up past 100K if margins look corpus-limited;
      recipe is a parameter, embedding cache amortizes the retry.
- [ ] ORCAS click-lane labeling (other 31% of composition): clicks are
      `source='click'` relevance (d37d) — own decision, own session.

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
- [ ] Re-read the route distribution + margins over all landed lanes —
      the "where to go next" readout this plan exists for; feeds the
      label-form decision (d38 deferred).
- [ ] Re-price wave order if pass 1 shows a lane's qrels_ready is thin
      (msmarco precedent: declared QQ ≠ per-query coverage).

## R1 LLM-judge pilot (SPEC d35 + d36) — CLOSED 2026-07-28, no action

Superseded by SPEC d37; full record in `docs/adr/0001` (kappa numbers, root
cause, and the do-not-reuse warning for the cached judgments).
`data/r1_pilot/` is the measurement audit trail.

## From taxonomy-generators grill (2026-07-23, SPEC decision 34)
- [ ] Realism-override audit: eyeball seeded samples per feature once
      defaults ship; write overrides where gibberish hurts (d34b).
      Named cases so far: `uri` (bank pattern is deliberately loose, so
      samples are valid-but-garbage scheme://host strings — reported
      2026-07-27); `datetime` (samples bare epoch seconds, "since
      1523759459") and `ip_address` (leading-zero octets, "0.4.251.255")
      — both observed in the enricher prototype run 2026-07-27.
- [ ] Enrichment grounding layer (doc consistency + provenance of
      augmented rows) — own grill-me session (d34a).
- [ ] Corruption operators' home (taxonomy-generators vs parent lane) —
      decide together with the R5 dark-matter work (d34d).
- [ ] Orchestrator-LLM batch lane over the d34e tool registry — when the
      order sheet demands volume. First prototype shipped 2026-07-27:
      `src/check_generator.ipynb` Enricher (litellm tool loop over the
      trio + instructor structured parse). Known d2 gap to fix before
      production: the final text is never re-measured locally —
      `features_used` is the LLM's self-report, and the instructor pass
      can reword after the last in-loop verify. End the lane with a
      local `verify()` on the returned text; raise or retry on FAIL.

## From feature-review of extractor-model generalization, 2026-07-16
- [ ] Watch: edify ignore_case() flag scope if phrase_alternation is ever
      composed with case-sensitive parts inside one define().

## From stage-1 close-out grill (2026-07-17, SPEC decision 17)
- [ ] PMI (avg pointwise mutual information) — blocked on a design question:
      which background co-occurrence source? (wordfreq is unigram-only;
      using the query corpus itself makes it corpus-relative)
- [ ] Char-level typos — blocked on: which dictionary for OOV/edit-distance?
- [ ] Corruption degree (clean/light/heavy) — derived view over corruption
      features once more of them exist (SPEC decision 8)

## From multilingual grill (2026-07-17, SPEC decisions 18-19)
- [ ] lingua-py dependency (scoped to the MIRACL-14 via from_languages, not
      all 75 — memory guardrail) + LanguageSetBank / CodeSwitchingBank in a
      new semantical/ package (SemanticFeature enum already in taxonomy.py).
- [ ] `languages` parameter plumbing: resolve/extract signature,
      supported_languages ClassVar on GeneralBank, extractor skip logic +
      init warning via spacy.util.get_installed_models().
- [ ] Language-keyed spaCy caches: _pipeline(lang), _doc(text, lang); pinned
      per-language model table; loud failure with download command.
- [ ] English-gated identifier audit: flag banks with English keyword gates
      (error/HTTP, Sprint, odds of...) and script-locked shapes (\b, [A-Z])
      for CJK/Cyrillic behavior.
- [ ] UD-based marker layer: negation via Polarity=Neg morphology as the
      multilingual replacement for the English word list — prototype and
      measure vs the regex bank on English first.
- [ ] Per-language gate: rerun the smoke-eval harness on MIRACL topics per
      language before any engine ships in that language.
- [ ] LLMBank: reserved last-resort engine — deferred by design (d18);
      LLM's active lane is teacher (silver labels) + judgment features.

## From profile-at-scale grill (2026-07-22, SPEC decision 31)
- [ ] ORCAS-tail harvest: named top source for short × feature-rich cells
      (~500K natural feature-bearing rows) — the d33 order sheet is the
      demand signal; wire harvest as the fill's supply escalation.

## From composition-fill grill (2026-07-22, SPEC decision 33)
- [ ] `tests/test_composition.py` — the package (`src/composition/`),
      `src/scripts/compose_target.py`, and the first 50K fill shipped
      2026-07-22; tests are the missing piece of the ratified plan.
- [ ] Selection data-quality chase. Finding (2026-07-27,
      `composition_audit` integrity): 13 empty/whitespace-only query
      texts (chase upstream in the source caches — likely a cache bug,
      distinct from dedup) and 339 exact-duplicate texts (the exact-dup
      half of the near-duplicate gate below — step 1 upstream dedup
      catches these for free). Also (2026-07-29, d38 labeling run):
      `checkable`/`label_lane` are assigned per-dataset, not per-query —
      all 15,678 msmarco-passage-dev rows carry `checkable=True`, but
      only 7,697 have a published judgment (MS MARCO judged 55% of dev;
      the fill drew judgment-blind and slightly anti-correlated, 49.1%
      realized). The per-query `checkable` feature-table column below is
      the fix.
- [ ] Provisional recipe values to revisit after the first fill's order
      sheet + lane split: ambiguity discounts (MODERATE 0.75,
      AMBIGUOUS 0.5), 1,000-weight floor size / slack margin.
- [ ] Minimum natural share — becomes binding when the generation lane
      starts filling order-sheet items.
- [ ] Near-duplicate gate. Finding (2026-07-27, `composition_embeddings`
      Act 3): the 50K carries ~5.5% near-duplicate pairs at cos > 0.95,
      concentrated in template-heavy sources (crumb-legal-qa, rarb-code)
      + a cross-dataset tail (crumb-code ↔ rarb-code paraphrases,
      quest ↔ crumb-set-operation). The composition doctrine assumed
      "distinct rows" without operationalizing it — feature-based fills
      can't see paraphrases, so if a source ships them we import them.
      Design order (cheap → correct):
      1. **Upstream dataset dedup** — cluster within each source's
         cache, keep one representative per near-duplicate cluster;
         catches within-dataset templating (the dominant contributor).
      2. **Post-composition dedup pass** — pairwise scan on the shipped
         Qdrant collection, drop the loser of each cos > θ pair (dropped
         from the slice with lower floor contribution), floors-respecting
         mini-fill on the vacated slots. Adds ~1-2 min to a build.
      3. **In-fill similarity gate** — `WeakestFirstFill` rejects
         candidates whose cos to any already-selected exceeds θ.
         Requires the full catalog pre-embedded (~50 min one-time) and
         an in-memory nearest-neighbor index during the fill.
      Threshold θ = 0.95 is the current observation cutoff; revisit once
      the post-pass runs and its precision/recall tradeoff shows up in
      the labeling stage's per-cell disagreement rates.
- [ ] Information-weighted labeling budget. Finding (2026-07-27,
      `composition_embeddings` Act 4): dense-vs-sparse top-10 Jaccard
      per slice — C median 0.00 (mean 0.06), A median 0.05 (mean 0.09),
      B 0.09, D mean 0.15 with a real right-tail at Jaccard = 1.0. Rows
      carry information proportional to (1 − Jaccard): C queries are
      worth ~15× the labeling effort D queries are, per row. If
      labeling per query is expensive (retrieval + judgment), equal
      per-slice budgets waste ~half the D effort on rows whose labels
      are already predictable. Design: mirror the disagreement gradient,
      not the composition gradient — e.g. per-slice labeling budget ∝
      slice_rows × (1 − mean_slice_jaccard). Feeds into R2's label
      schema (label a query "confident" when its Jaccard is high
      and both arms agree; label uncertainty needs explicit rate on
      C-shaped rows). Caveat: some of C's zero-Jaccard is retrievers
      disagreeing about *what the query even asks*, not about which
      candidate is more relevant — those rows carry noisy signal,
      not richer signal. Distinguish via label-agreement rate in the
      hole-filling measurement (d37k) before scaling the weighting.
- [ ] Pilot A/B: quota-sequential vs weakest-first fill over the same
      feature table — fill-ratio profiles at several budget cuts +
      selected-set overlap.
- [ ] `checkable` boolean column in the feature table (≥1 judged doc, or
      generated provenance).
- [ ] Floor precision target — one number (±points per cell → n via
      1/√n); decide together with the recipe values.
- [ ] R2 label schema: pin (dense_model, sparse_model, fusion, k, depth)
      tuple in the label artifact; set tie margin ε; two-dense-model
      kappa pilot on ~500 rows (low-kappa cells = stack artifacts).
- [ ] R3 corpus covariate: store corpus id with every label; measure
      per-cell cross-corpus score variance in the pilot — high variance
      un-defers corpus-relative features (vocabulary mismatch first).
- [ ] R4 design effect: cluster-robust SEs (by source dataset) on pilot
      cells; multiply floors by measured deff.
- [ ] R5 dark-matter generation: programmatic corruption operators
      post-generation + rejection sampling against target signatures —
      never prompt the LLM to "be messy" (hotchpotch evidence).
- [ ] Chase Qdrant page-search query logs — the one obtainable real
      workload histogram for eval weights.
- [ ] Register/box definitions for eval weighting — deferred until a
      real log can inform the boxes.

## From first-profile results grill (2026-07-20, SPEC decisions 22-25)
- [ ] Per-snapshot profile artifact — second profile view scoped to the
      grounded snapshot (for future per-query NDCG correlation, i.e. the
      "labeling view" that d23 explicitly deferred). Unblocked: engine
      benchmark passed 2026-07-21 (src/data/benchmarks/engines.csv).
