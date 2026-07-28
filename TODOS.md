# TODOS

Gates / next actions:

- [ ] POS domain-shift diagnostic (d14 guardrail): one-off
      natural_language_share ↔ stopword-ratio correlation over the cached
      datasets — natural to run with the first full profile.
- [ ] `src/scripts/benchmark_engines.py` argparse guard: ANY invocation
      (even `--help`) runs the full benchmark and appends rows to
      engines.csv — bit us 2026-07-21 (two stray rows scrubbed by hand).

Deferred by grill-me session 2026-07-15 (see SPEC.md):

- [ ] Corpus-relative features: explicitly decide whether to lift the
      registry's queries-only restriction.
- [ ] Strategy labeling stage: empirical dense/sparse/hybrid labels via NDCG
      in src/hybrid_search_rrf_dataset.
- [ ] ILP escalation path if greedy quota-fill conflicts (solver,
      formulation).
- [ ] Demo (b) infra: corpus indexing + local Qdrant for the
      strategy-disagreement measurement.
- [ ] Open card ratifications (SPEC d21): MIRACL llm_target/non_trivial
      vs candidate-list coding; DBPedia scope G-vs-S.

## From golden-set grill (2026-07-28, SPEC decision 37)

Next actions, in order — (1) and (2) block everything else:

- [ ] Lift the registry's queries-only restriction to retain doc_ids and
      ORCAS clicks. Cache is `[query_id, text]` for all 21 datasets, so
      the 18.8M ORCAS click pairs are dropped on ingest. Needs a cache
      refill (user-initiated).
- [ ] Pool extraction: run the three routes over a query list and emit
      `(dataset, query_id, route, rank, doc_id, score)`. No qrels needed,
      so it does not touch `FusionBuilder`. `_iter_queries`' skip-on-no-
      qrels guard is correct and stays — removing it would fabricate a
      `dense_only` label per unlabelable query.
- [ ] Anchor yield measurement on nfcorpus (323 queries, 3,633 docs,
      already indexed as collection `nf`): the three-way split of
      d37(i) — all-tied / routes-differ / all-zero. Decides whether 50K
      queries yield enough trainable rows.
- [ ] Explicit outcome for unanswerable queries (d37i). Today the argmax
      falls through to list order and fabricates `dense_only` (1 of 76 on
      trec-dl; grows with corpus realism).
- [ ] Deliberate tie-break rule, replacing incidental list order. Note
      `dense_only` is likely not the cheapest route — BM25 needs no
      query-side transformer pass.
- [ ] Constant-dense and constant-sparse baselines (d37h). Two
      `BaselineBuilder` calls; without them a router can lose to one line
      of code and still look fine on regret.
- [ ] `QrelSource.CLICK` + conflict priority human > click > llm.
- [ ] Retrieval latency benchmark for the three routes. Speed is one of
      three stated requirements and is entirely unmeasured;
      `data/engines.csv` benchmarks taxonomy extractors, not retrieval.
- [ ] Per-dataset corpus sizing. Indexing cost is uncorrelated with row
      yield: trec-dl needs 138M docs for 79 rows, miracl ~33M for 530,
      dbpedia ~4.6M for 400, while msmarco is 8.8M for 15,678.
      `TrecDL2022(30000).materialize()` (judged docs only) is the existing
      pattern, but a corpus of near-answers inflates dense — d37(g).
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

## R1 LLM-judge pilot (SPEC d35 + d36) — CLOSED 2026-07-28, no action

Superseded by SPEC d37. `judge.py` and `r1_judge_calibration.ipynb` are
deleted; `data/r1_pilot/` is kept as the measurement audit trail. Recorded
here so the result is not re-derived from scratch:

- 4-level kappa 0.120–0.139, bootstrap CI upper bound 0.181 — a 0.6 gate is
  out of reach at any sample size, so the verdict is definitive rather than
  underpowered.
- Few-shot moved it by ±0.02 with fully overlapping CIs (Haiku 0.348 →
  0.328, Sonnet 0.320 → 0.340 at grade≥2). Exemplar quality was never the
  binding constraint, so the real-exemplar ablation is dropped, not deferred.
- Sonnet ≈ Haiku throughout: 3× the cost bought nothing.
- Root cause: the design measured per-document grade agreement where the
  pipeline consumes a per-query route decision. Whether route agreement
  survives moderate per-document noise is untested — that is d37's question.
- Do not reuse those cached judgments to evaluate a *binary* judge. They came
  from a 4-level prompt over pairs sampled from existing qrels; the 155-vs-8
  false-negative skew is an artifact of collapsing that rubric at grade≥2,
  not a property of a judge asked the binary question directly.

## From taxonomy-generators grill (2026-07-23, SPEC decision 34)
- [x] Implement `src/taxonomy_generators/` per d34: SurfaceGenerator ABC,
      PatternGenerator defaults over FEATURE_BANKS (rstr, guard-stripped
      patterns), override hook, tool trio registry, MCP extra, round-trip
      test (`tests/test_taxonomy_generators.py`). Absorbed the former
      "MCP wrapper around verify()" item (d34e).
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

## From logical-group expansion grill (2026-07-20, SPEC decision 20)
- [ ] Model backstop for CODE_FRAGMENT/MATH_EXPRESSION recall (symbol-light
      formal content: "x squared plus y squared", prose pseudo-code) —
      layered bank; needs a code/math detection model choice.
- [ ] Attested search-syntax extensions to OPERATOR_SYNTAX (quoted phrases,
      minus-exclusion, `site:`) — attested in query logs but
      precision-dangerous; own decision.

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
      catches these for free).
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
      R1 hole pilot before scaling the weighting.
- [ ] Pilot A/B: quota-sequential vs weakest-first fill over the same
      feature table — fill-ratio profiles at several budget cuts +
      selected-set overlap.
- [ ] `checkable` boolean column in the feature table (≥1 judged doc, or
      generated provenance).
- [ ] Floor precision target — one number (±points per cell → n via
      1/√n); decide together with the recipe values.
- [ ] R1 hole pilot — SUPERSEDED 2026-07-27 by d35: reframed as an
      LLM-judge calibration pilot because the deferred lane already
      forces LLM-as-judge deployment. Implementation item is now in
      the composition-fill section below (d35 R1-implementation).
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
