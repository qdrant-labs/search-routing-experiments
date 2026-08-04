# Router — weakness map

Full diagnostic. Complements PLAN.md's short-list of critical-path issues.
Rows are grouped by status (active / deferred / dropped) and by layer.
Each row names one weakness, what we observe, and the fix path.

## Active

Ordered by impact on the router's current ceiling. #1, #2, #3, and #7 share
the composition fix path (SPEC d45h5, d45h6) and attack a common root
cause.

### Dataset

**1. Features live in only one routing context.**
- Observe. URI has 3 decisive rows, all long documents, all dense-wins.
  The LR sees only one side of the correlation. Each feature needs
  positive AND negative examples across route labels for the model to
  learn "URL surface → probably sparse" as a rule instead of memorising
  three points.
- Fix. Composition redesign to add per-feature routing balance as a
  constraint (each feature ≥ N dense-winning AND ≥ N sparse-winning
  decisive rows). SPEC d45h5.

**2. Archetype coverage holes.**
- Observe. The (feature × length × domain) grid has empty cells. Zero
  decisive rows for (URI × short_query). Similar holes likely exist for
  (math_expression × short), (code × NL-heavy).
- Fix. Inject augmentation (SPEC d42, d45h6) fills empty cells with
  synthetic rows carrying real corpus grounding. Blocked on d40e
  coherence pilot.

**3. Feature-diversity target vs routing-signal target.** *Highest impact.*
- Observe. The composition (SPEC d31/d33) was optimised for taxonomy
  coverage. The 74/20/6 route distribution is a side effect; the recipe
  never targeted routing balance. Upstream of #1, #2, and #4.
- Fix. Reopen recipe values in d45h5. Add routing-outcome balance as a
  second axis alongside feature coverage.

**4. Decisive filter throws 90% of labels.**
- Observe. 24,338 labels → 2,510 decisive. The F3 experiment (relax to
  routes_differ ~9.8K rows) hurt holdout by −0.06. Thin-margin rows carry
  noisier labels. The bottleneck is decisive row supply.
- Fix. Grow decisive by more labelling: more lanes (d39 wave 2) or
  higher-margin lanes (rebalance recipe).

**5. Near-duplicate pairs (~5.5%) may straddle splits.**
- Observe. Cos>0.95 pairs (measured, d33) can leak eval → train because
  the split isn't near-dup-aware. Random_within_lane numbers may be
  inflated by an unknown amount.
- Fix. Near-duplicate-aware split per SPEC d46f, or upstream dataset
  dedup per composition-fill TODO.

**6. Corpus at labelling, absent from the model.** *Structural.*
- Observe. Labels bake in corpus knowledge; features contain only query
  knowledge. Reproducing corpus-driven outcomes from corpus-blind inputs
  is an underdetermined inverse.
- Fix. Privileged corpus features at training, masked at inference
  (LUPI). Three implementations in PLAN.md (dropout / auxiliary
  reconstruction / teacher-student distillation).

**7. Composition query distribution doesn't match canonical archetypes.**
- Observe. Median 15 words, natural-language-heavy. Only 4.1% of decisive
  sparse-winners are short + identifier. Auto-fusion finds nothing to
  bite on; the LR memorises composition-specific patterns.
- Fix. Composition redesign (d45h5) to add archetype-coverage axis.
  Register-realistic supplements (ORCAS/click-lane) as a secondary route.

### Labelling

**8. Labels carry unmeasured asymmetry from qrel holes.**
- Observe. BEIR literature reports dense retrievers hitting 14–32% hole
  rates vs BM25's ~6%. Direction and magnitude on our labels are
  unmeasured. Our decisive-row distribution (74% dense) may already be a
  hole-corrected number, or may be biased further.
- Fix. Hole-filling via list-preference judge on unjudged retrievals
  (SPEC d37k, d44d). Deferred until hole rate is measured.

**9. Labels stack-pinned to bge-small + Qdrant/bm25.**
- Observe. Swap the dense encoder and the labels change; the router
  isn't portable across encoder choices. R2 pilot never ran.
- Fix. Two-encoder kappa pilot (~500 rows) per SPEC d30 R2, cross-
  validated with the list-preference judge (SPEC d44d).

**10. msmarco labels are noisier per row than nfcorpus labels.**
- Observe. msmarco has median 1 judged doc/query. A route's label swings
  on whether one specific doc appears in its top-10. On nfcorpus (median
  16 judged) a route's score reflects the whole relevant set, so
  variance per row is much lower.
- Fix. Deeper judgment on msmarco (parked; expensive) or hole-filling
  (d37k). Low priority; the lane already carries a caveat.

### Model / methodology

**11. LR extracts correlations, not rules.** *Root cause of the URI case.*
- Observe. An LR learns feature→label associations from the training
  slice. It doesn't know retrieval theory or token statistics; it can't
  represent "URLs are lookup-y, BM25 handles them" as a rule. URI case:
  3 rows, all dense-wins, correlation shipped as a decision rule.
  Beating auto-fusion (the LLM incumbent) at generalisation is the real
  threat model.
- Fix. Three complementary paths:
  (a) enough coverage that correlations become rule-consistent
      (composition + augmentation);
  (b) LUPI architecture with privileged corpus features at training
      (PLAN.md options A/B/C);
  (c) benchmark against auto-fusion per-archetype as the ship criterion.

### Evaluation

**12. Aggregate metric hides archetype-level failures.**
- Observe. URL query gets `p_dense = 0.99`, but URLs are ~0.1% of the
  eval slice; the mean doesn't move. Every setup-side intervention so
  far has landed as "aggregate noise, qualitative gain."
- Fix. Per-archetype eval. Group held-out rows by feature signature
  (`has_uri`, `has_uuid`, `is_short`, `is_math`, `is_natural_language`)
  and report headroom per group. Cheap; pending.

**13. One holdout lane represents "transfer."**
- Observe. SPEC d45e originally called for a 16-lane rotation. We test
  rarb-math only. Transfer estimate is a sample of one.
- Fix. Hide-one-lane × 16 rotation, or at least 3–4 representative lanes.

**14. Every intervention measured on the same eval slice.**
- Observe. F1, F2, F1', F3 flags, corpus-stat sweeps all evaluated
  against the same 497 random-within-lane + 586 rarb-math held-out
  rows. Overfitting risk to that specific slice.
- Fix. Reserve a separate final-eval slice; unlock only at
  commit-to-ship time.

**15. Probability calibration never checked.**
- Observe. We use `p_sparse = 0.826` etc. as if they're actual
  probabilities. LR with `class_weight="balanced"` isn't automatically
  calibrated. If it's off, the tuner's operating point is wrong for
  reasons unrelated to feature quality.
- Fix. Calibration curve on a held-out slice; Platt scaling if needed.

### Methodology

**16. Judge pipeline parked; no fallback labelling method.**
- Observe. If human qrels run out (composition expansion, new lanes),
  no way to keep labelling at scale. d44d spike proposed, never run.
- Fix. Run the d44d list-preference judge spike (~500 rows, 3 lanes,
  zero retrieval). Decides whether scale-labelling opens.

## Deferred

Model-layer items to revisit after the composition side moves the numbers.

**D1. Two independent binaries can't model dense↔sparse interaction.**
`pure_rrf` is a residual, not a learned class; the router can't affirm
rrf as the right answer. Fix: LightGBM (SPEC d45c) or a proper three-way
multinomial.

**D2. Tuner conflates fire-thresholds with abstention thresholds.**
Same grid picks both when-to-fire dense/sparse and how-often-to-hedge to
rrf. F1' filter helps; per-class recall floor plus separate abstention
constraint would help more. Own grill.

## Dropped

**X1. Lane mix drags the pooled mean.** (originally #6.)
Reporting caveat, not a system weakness. Notebook §18 already flags
per-lane vs pooled.

**X2. Queries corpus-alien for one production use case.** (originally #7.)
The router should work on any production system, not Qdrant page-search
only. The distinct point ("composition doesn't match any archetype
target well") survives as active #7 above.

**X3. StandardScaler amplifies rare features.** (originally #12.)
A symptom, not a cause. Fixing scaling doesn't fix the URI case; more
coverage does. If active #1 and #2 land, this stops mattering.

## Prioritisation

- **Highest current-ceiling impact.** #1, #2, #3, #6, #7, #11. Composition
  and architecture together. #3 is upstream of #1, #2, #4; a routing-
  balance axis in the recipe collapses several into one fix.
- **Highest silent-bias risk.** #8, #9, #10. These distort what we think
  we're measuring.
- **Cheapest right now.** #12 (per-archetype eval, half day), #16 (judge
  spike, well-scoped).
- **Requires composition surgery.** #1, #2, #3, #4, #7. Same fix path
  (SPEC d45h5 plus d42/d45h6). One intervention with multiple symptoms.

The composition side is the load-bearing intervention. LUPI (item #6 fix
path) is the architectural direction. Per-archetype eval (#12 fix path)
is the diagnostic that drives both.
