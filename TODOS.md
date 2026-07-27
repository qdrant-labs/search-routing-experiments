# TODOS

Gates / next actions:

- [ ] POS domain-shift diagnostic (d14 guardrail): one-off
      natural_language_share ↔ stopword-ratio correlation over the cached
      datasets — natural to run with the first full profile.

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

## From taxonomy-generators grill (2026-07-23, SPEC decision 34)
- [ ] Implement `src/taxonomy_generators/` per d34: SurfaceGenerator ABC,
      PatternGenerator defaults over FEATURE_BANKS (rstr, guard-stripped
      patterns), override hook, tool trio registry, MCP extra, round-trip
      test — code-implementer picks this up. Absorbs the former "MCP
      wrapper around verify()" item (d34e).
- [ ] Realism-override audit: eyeball seeded samples per feature once
      defaults ship; write overrides where gibberish hurts (d34b).
      Named cases so far: `uri` (bank pattern is deliberately loose, so
      samples are valid-but-garbage scheme://host strings — reported
      2026-07-27).
- [ ] Enrichment grounding layer (doc consistency + provenance of
      augmented rows) — own grill-me session (d34a).
- [ ] Corruption operators' home (taxonomy-generators vs parent lane) —
      decide together with the R5 dark-matter work (d34d).
- [ ] Orchestrator-LLM batch lane over the d34e tool registry — when the
      order sheet demands volume.

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
- [ ] Per-snapshot profile artifact — second profile view scoped to the
      grounded snapshot (for future per-query NDCG correlation, i.e. the
      "labeling view" that d23 explicitly deferred). Unblocked: engine
      benchmark passed 2026-07-21 (src/data/benchmarks/engines.csv).
