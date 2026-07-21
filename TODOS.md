# TODOS

Gates / next actions (2026-07-15, GLiNER2 sidestep — SPEC.md decision 13):

- [ ] POS domain-shift diagnostic (d14 guardrail): one-off
      natural_language_share ↔ stopword-ratio correlation over the cached
      datasets — natural to run with the first full profile.

Deferred by grill-me session 2026-07-15 (see SPEC.md):

- [ ] Recipe values: total size, per-feature quotas, within-feature strata
      quotas, minimum natural share, harvest-target N, per-quota
      per-dataset source cap (d29 default ≤50%).
- [ ] MODEL-tier features: pick the model + testing story for
      non-deterministic extractors (arch-validator session).
- [ ] Corpus-relative features: explicitly decide whether to lift the
      registry's queries-only restriction.
- [ ] Strategy labeling stage: empirical dense/sparse/hybrid labels via NDCG
      in src/hybrid_search_rrf_dataset.
- [ ] ILP escalation path if greedy quota-fill conflicts (solver,
      formulation).
- [ ] Register next datasets — full catalog with card tuples, harvest
      hypotheses, and backend sketches now in docs/datasets.md (2026-07-20).
      Wave 1: ORCAS, BRIGHT, QUEST, CRUMB, RAR-b math/code pools, LIMIT,
      DBPedia-entity. Open ratifications: MIRACL llm_target/non_trivial
      card vs candidate-list coding; DBPedia scope G-vs-S.
- [ ] MCP wrapper around verify() for interactive generation.
- [ ] Demo (b) infra: corpus indexing + local Qdrant for the
      strategy-disagreement measurement.

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

## From presentation & composition grill (2026-07-21, SPEC decisions 27-29)
- [ ] `CorpusReport` in query_taxonomy `reporting.py` (.text() human
      rewrite + chart-ready rollup data, stdlib only); delete
      `CorpusFeatures.summary()`/`__str__`; point profile_datasets.py's
      `.summary.txt` at the new text.
- [ ] Parent-repo chart helpers (matplotlib): two-ring domain donut (d27),
      fingerprint heatmap + spider comparison (d28); wire into demo.ipynb.
- [x] (done 2026-07-21) Feature-table extraction pass → parquet (d29):
      `src/feature_table.py` — per-dataset parquet + concatenated catalog
      in src/data/feature_table/, columns `<group>.<type>` span counts +
      `<bank>.<stat>` scalars. `checkable` is card-level (grounding == QQ)
      until the qrels/queries-only decision lands, then goes per-query.
      Greedy quota-fill with capped harvest-priority reads it — fill
      itself stays blocked on recipe values.

## From composition doctrine grill (2026-07-21, SPEC decision 30)
- [ ] Pilot A/B: quota-sequential vs weakest-first fill over the same
      feature table — fill-ratio profiles at several budget cuts +
      selected-set overlap.
- [ ] `checkable` boolean column in the feature table (≥1 judged doc, or
      generated provenance).
- [ ] Floor precision target — one number (±points per cell → n via
      1/√n); decide together with the recipe values.
- [ ] R1 hole pilot (a day, runs first): one cell, ~100 rows, all three
      strategies, per-strategy Hole@10 — decides whether post-hoc
      LLM-as-judge (MEMERAG-calibrated) is a hard labeling prerequisite.
      Quantitative gate replaces the qualitative low-trust flag: hole-gap
      over threshold blocks the cell's labels.
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
- [x] (done 2026-07-21) Full-scale run of `src/benchmark_engines.py` at
      N=100K on both engines (regex / spacy; gliner dropped 2026-07-21).
      Results in `src/data/benchmarks/engines.csv`: regex ~11.4K qps
      (p50 0.08ms, RSS flat ~69MB); spacy ~2.3K qps (p50 0.41ms, p99
      0.89ms, RSS plateaus at ~365MB after 10K — no growth at 100K).
      Both linear in N; no degradation → unblocks the item below.
- [ ] Per-snapshot profile artifact — second profile view scoped to the
      grounded snapshot (for future per-query NDCG correlation, i.e. the
      "labeling view" that d23 explicitly deferred). Blocked on the stress
      test above.
