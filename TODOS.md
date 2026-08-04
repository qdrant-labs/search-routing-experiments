# TODOS

## From router-improvement work (SPEC decision 47)

Current state (2026-08-04): setup pass shipped (F2 + F1'); router beats
constant on both protocols (0.751 / 0.611). d47(b) inference-time corpus
stats retired — deployment target unknown at ship time. See PLAN.md for
the current-state summary.

Done:

- [x] Pass 1 — Setup pass. **F2 landed**: three derived columns in
      `FeatureSpace.transform` (`identifier_density`, `avg_word_length`,
      `short_id_query`); `length.length_words` dropped. **F1' landed**:
      `tune_thresholds` filters input to `routes_differ` rows only.
      **F1 tried and reverted** as documented negative result. Numbers
      after F2 + F1': random_within_lane 0.751 (+0.057 headroom),
      holdout_lane 0.611 (+0.017). First positive headroom on either
      protocol. Auto-fusion added to the ablation table via
      `AutoFusionRouter` + `LLMScoreClient` (`RouterExperiment.run(
      autofusion=True)`); cache at
      `data/route_labels/autofusion_cache.parquet`.

Retired:

- [x] d47(b) — corpus stats as router inference features — RETIRED
      2026-08-04. Deployment target unknown; `query → route` surface is
      a hard constraint. Corpus stats stay in the offline labelling /
      diagnostic toolkit. `CorpusRelativeBank` design preserved in
      SPEC d47(b) for offline use if needed. Passes 2–5 (build corpus
      index / side test / six-config ablation) all fall under this
      retirement.

Open (priority order):

- [ ] Per-archetype eval. Group held-out decisive rows by feature
      signature (`has_uri`, `has_uuid`, `is_short`, `is_math`,
      `is_natural_language`). Report LR router vs auto-fusion per group.
      Surfaces coverage gaps in the composition. Cheap; belongs in
      `route_baseline.ipynb` as a new section.
- [ ] Score-vector regression target (SPEC d41a canonical path). Replace
      the two argmax-hard-label binaries with a multi-output regression
      on `(score_dense, score_pure_rrf, score_sparse)`. Same input, same
      serving API, richer training signal. Derive `route` via existing
      cost-order rule + margin-based hedge to `pure_rrf`.
- [ ] LightGBM v2 (SPEC d45c). Query-only ceiling test with a learner
      that captures interactions. Behind the same `predict(query) →
      StrategyName` API. If it beats the LR meaningfully, LR is a
      distillation target; if not, LR sits at the query-only ceiling.
- [ ] LUPI prototype (PLAN.md option B — auxiliary corpus
      reconstruction). Small MLP encoder on query features, two heads
      (one predicts corpus features as auxiliary loss during training,
      dropped at inference; one predicts route). First real test of
      whether privileged corpus features at training lift the query-only
      ceiling. Options A (dropout on corpus features) and C (teacher-
      student distillation) documented as fallback / heavier variants.
- [ ] Composition + augmentation loops (SPEC d45h5, d45h6). Driven by
      the per-archetype failure map. Fill the coverage holes named in
      PLAN.md issues #1–3 and #7. Not gated by modeling work; each
      iteration compounds.

Deferred:

- [ ] F3 (relax decisive-only training) — F1' subsumed the tuner half of
      this concern. The training-set half (loosen the decisive filter to
      routes_differ) still SPEC-touching; the F3 experimental flag on
      `StrategyRouter.fit` stays in the code for future testing but the
      default keeps decisive-only per d45(a).

Gates / next actions:

- [ ] POS domain-shift diagnostic (d14 guardrail): one-off
      natural_language_share ↔ stopword-ratio correlation over the cached
      datasets — natural to run with the first full profile.
- [ ] `src/scripts/benchmark_engines.py` argparse guard: ANY invocation
      (even `--help`) runs the full benchmark and appends rows to
      engines.csv — bit us 2026-07-21 (two stray rows scrubbed by hand).

## From router-baseline grill (2026-07-30, SPEC decision 45) — human twin PLAN.md

Two experiments on existing data, in parallel, before collecting anything:

- [ ] Classifier baseline (d45a/b/c + d46): decisive-row dataset (2,510;
      dense 1,858 / sparse 507 / rrf 145), two one-vs-rest logistic
      binaries (dense, sparse), serving rule both-below-threshold ⇒
      pure_rrf, both-fire ⇒ higher prob. Train on decisive, tune
      thresholds by CV on train (incl. all_tied/thin) so ambiguous
      queries abstain to rrf. rrf never a trained class.
      Build spec in d46: `router.py` StrategyRouter, three-config
      ablation (57 engineered / e5-small embedding PCA~50 / both),
      encoder = intfloat/multilingual-e5-small via sentence-transformers
      (NOT bge-small-en — label-coupling bias; needs `query: ` prefix),
      add scikit-learn + sentence-transformers.
- [ ] Stable API (d45d): `predict(query, *, collection_stats=None) ->
      StrategyName`; carry `collection_stats` from day one though v1
      ignores it (d44b grafts in with no break). logistic→LightGBM is
      an implementation swap behind this surface.
- [ ] Validation (d45e): (i) random 20% within-lane mask,
      near-duplicate-aware (~5.5% cos>0.95 pairs must not straddle);
      (ii) hold-one-lane-out = **rarb-math** (all 3 classes, dense 334 /
      sparse 191 / rrf 61). Transfer estimate trains on the other 15;
      shipped model retrains on all 16 (rarb-math holds 38% of sparse).
- [ ] Six-column eval (d45f) over decisive rows: constant-dense /
      -sparse / -rrf / production classifier / our router / oracle.
      Bar = best constant, not production alone.
- [ ] Judge spike IN PARALLEL (d45g): list-preference over stored
      top-10s (route_rankings, zero retrieval), ~500 rows nfcorpus /
      crumb-legal-qa / rarb-math, vs the ~24K empirical spine;
      ≥80% opens scale-labeling, 60–80% panel, <60% → CTO conversation.
- [ ] v2 upgrade (d45c): LightGBM behind the same API once v1's ceiling
      is measured; NN ruled out for the tabular feature profile.
- [ ] Gated follow-ons, plans not builds until the experiments report
      (d45h): (5) composition redesign to (query, corpus) targets;
      (6) augmentation = Inject-only for sparse, behind the orcas
      realism baseline, only if harvest leaves sparse starved.

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

## From generated-rows grill (2026-07-29, SPEC decision 40)

- [ ] Coherence-gate pilot (d40e): LLM meaning-gate for doc-consistent
      injections, validated against a human-audited sample (d34b audit
      pattern) before any injected row enters the golden set.
      d37(f)-compatible: need-identity is a function of the two query
      texts — no information gap.
- [ ] Generation-lane creation contract (d40c), when the lane opens:
      (provenance, home_lane, grounding_doc_id, parent_query_id) required
      at row birth; qrels minted mechanically with source='constructed';
      undeclared operators ⇒ feature-stock.
- [ ] QrelStore.source: add 'constructed' to the enum + priority
      human > constructed > click > llm (d40b) — schema touch, do
      together with the first constructed rows. Scheduled: d43 pass 4
      (lands with data/augmentation/qrels.parquet; inherit-path copies
      keep source='human' + inherited_from, d43d).
- [ ] Multi-doc grounding for logical-structure features (d40g hard
      case): set-operation/conditional queries whose answer is a doc
      *set*, QUEST-style construction from category structure. Deferred —
      harvest quest (928) and crumb-set-op (423) natural rows first;
      generate only for cells they leave empty.
- [ ] Generation-batch quality gauge (d40h): wire the per-batch
      `all_zero` rate into the generation lane's verification loop once
      the lane opens — construction failures surface as label-less rows,
      no embedding metric anywhere in the gate.

## From full-loop grill (2026-07-30, SPEC decision 43)

- [x] Pass 2 — DONE 2026-07-30: composition/admission.py MiniFill,
      WeakestFirstFill top_up=False mode, min_natural_share=0.85,
      generated_from column, sheet re-emit. Validated in tmp: 14/29
      admitted, politeness floor closed, caps bound at 7/lane.
- [x] Pass 3 — DONE 2026-07-30: structural hook (abstract per the
      arch/clean-code pass), OperatorSyntaxRewrite (11,491 parents),
      StatRewrite (14,642 near-first parents; length is regex-tier —
      no spaCy needed for (length, up)); gated operators produce
      feature-stock; pool carries credit_gate; MiniFill skips gated.
      REMAINING (human): the two declaration audits unlock credit.
- [x] Pass 4 — DONE 2026-07-30: SupplyIndex (nfcorpus built as smoke:
      44,301 surfaces; build_all is the user's trigger) + rung readout;
      InjectOperator (82 rung-1 pairs on id:medical from nfcorpus
      alone; literal + no-other-floor structural); AugmentationQrels
      minted at write (constructed for Inject, human+inherited_from
      copies for inherit path); QrelSource.CONSTRUCTED inserted at
      rank 2 (closes d40b). first_generation_only=True constructor
      guard on every operator (d43 review). REMAINING (human): d40e
      coherence pilot gates Inject credit; (code, later): labels.py
      merge of augmentation qrels so admitted children get labelled.

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

## From augmentation-loop grill (2026-07-29, SPEC decision 42)

- [ ] Pass 1 (code-implementer): `src/augmentation` package — operator
      registry + declaration schema (d42c/d), Augmenter agentic
      tool-loop engine with bounded retries and final LOCAL verify
      (d42g, closes the d2 gap), Decorate operator end-to-end over the
      marker floors (the only gate-free family — earns credit day one).
- [ ] Supply index build: one-time bank profile per lane corpus →
      `data/<lane>/surfaces.parquet` (floor keys via floors.py mapping);
      finish the augmentation_supply.ipynb readout (3 cells) → assign
      each hungry id floor its ladder rung (d42f).
- [ ] Mini-fill start-from-base mode on WeakestFirstFill +
      `generated_from` column in the selection schema + minimum
      natural-share recipe value (d42i/j; d33 binding now — number at
      recipe review).
- [ ] Declaration pilots (d42h, d34b audit pattern):
      OperatorSyntaxRewrite and StatRewrite(length, up); credit unlocks
      per entry on pass.
- [ ] Inject: blocked on the d40e coherence-gate pilot (existing d40
      item) + the supply index. Constructed-docs lane (d42k) built when
      the synthetic rung first fires — never index new docs into an
      existing collection.
- [ ] Renames: Enricher → Augmenter, enrichment_supply.ipynb →
      augmentation_supply.ipynb (user's Jupyter may hold the notebook
      open — rename at implementation, not mid-session).
- [ ] taxonomy_generators name dialect: registry/catalog use
      group-prefixed feature names ("sentence_markers:politeness") while
      verify counts bare bank names ("politeness") — bit the first live
      Augmenter run 2026-07-29 (LLM passed a bare name to
      generate_surface → KeyError). Engine now returns tool errors to
      the LLM (self-corrects), but align the two dialects at the source.
- [ ] Surface-concentration readout (d42m): per-batch tally of which
      decoration phrase each accepted row used (banks already extract
      the span text) — the model-free diversity check; promote to a
      rejection cap only if the readout shows the seeded exemplars
      insufficient.
- [ ] Kernel-integrity protocol: the first politeness run accepted text
      the on-disk politeness bank rejects (stale kernel modules under
      autoreload with newly created packages). Before any real batch:
      restart the kernel; the 14-row politeness batch doubles as the
      integrity test — fresh-kernel accepts must match on-disk verify.
- [ ] Engine batch concurrency (thread pool in loop.run, rate-tier
      aware) + Anthropic prompt caching — options 3/4 from the
      2026-07-30 runtime session, deferred by choice; pick up before
      the ~861-row greeting/interjection batches (sequential ≈ 45min
      even at the new ~3s/row).

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
- [x] Enrichment grounding layer (d34a) — RESOLVED: golden-set half by
      d40, generation-side design by d42 (supply index, Inject ladder,
      operator declarations). Implementation lives in the d42 section.
- [x] Corruption operators' home (d34d) — RESOLVED by d42(b): the
      operator registry in `src/augmentation`; taxonomy_generators
      stays grounding-blind surfaces + verify.
- [x] Orchestrator-LLM batch lane (d34e) — SUPERSEDED by d42(g): the
      Augmenter agentic tool loop IS the lane; the d2 gap (final text
      never locally re-measured; instructor pass can reword after the
      last in-loop verify — confirmed in code 2026-07-29) is closed by
      the final-local-verify acceptance rule. Implementation in the d42
      section.

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
- [ ] Minimum natural share — BINDING now (d42 opens the lane). Home
      decided by d42(i): a recipe value, enforced by the mini-fill,
      never by the loop's own accounting. Remaining: set the number at
      recipe review.
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
      tuple in the label artifact; two-dense-model kappa pilot on ~500
      rows (low-kappa cells = stack artifacts). Tie margin ε resolved
      by d41(e): ε = 0 — ties are exact, near-ties are thin-margin
      routes_differ.
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
