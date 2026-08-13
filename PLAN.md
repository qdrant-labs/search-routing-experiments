# Router: current state

The router is a `query → route` function shipped as an artifact. Route is one
of `dense_only`, `pure_rrf`, `sparse_only`. Deployment target is settled
2026-08-12: a generalized product for customers needing strong top-1,
battle-tested first on Qdrant page-search. No corpus features enter at
inference; corpus knowledge is imputed in-model (d63).

## Viability program — where everything lives (2026-08-13)

The router is under a precommitted go/no-go evaluation. Three documents, one
split: **VIABILITY_PLAN.md** is the frozen plan (goal, signed thresholds
Δmin = 0.02, claim table C1/C1b/C2/C3, the three auto-fusion gates, the
zero-tolerance probe gate; §8 = every decision with date and decider);
**VERDICT.md** is the growing results record (every number, cell-referenced);
evidence notebooks are `src/phase0_verify.ipynb` and
`src/phase05_feasibility.ipynb` over `src/scripts/feasibility_gate.py`.

| step | status | outcome |
| ------| --------| ---------|
| Wave 0 scope review | done | thresholds signed; 9 Codex objections folded |
| Phase 0 verify record | done | dataset doubled to 46,142 rows; record re-pinned (table below); qrel holes 89% but route-symmetric; raw scores exist at top-10 |
| Phase 0.5 feasibility | done | precommitted lane-equal estimator CANNOT resolve 0.02 (32/41 lanes <100 answerable test rows); powered re-poolings put router BELOW C1's bar |
| Phase 1 formal C1 + gates | done | **C1 FAIL, confirmed by two adversarial passes** — router −0.0154 vs per-lane best constant, CI [−0.0322, +0.0015], entirely below Δmin; FAIL survives every re-pooling, seed, clustering threshold, and baseline variant tried. Probe gate RED (6/30 unique under the shipped criterion, 1/30 under pure ordering). Rotation vs a GLOBAL constant: +0.0234, INCONCLUSIVE — a lead, not evidence |
| Wave 2 retrieval (C2, M1, external corpus) | not run | stopped by the plan's own gate: if C1 fails, write the verdict and stop |
| **Verdict** | **NO-GO, written** | for the shipped LR router, on this dataset, at the signed bar — see VERDICT.md |

Nothing is pending. The router's obituary is d63's spec: beat the **per-lane
best constant** (not const-dense) by > 0.02 on answerable rows under the
verdict's estimator, fix the short-query slice (−0.04), and clear 30/30 golden
orderings. The eval harness built for this verdict — clusters, estimator,
notebooks — is reusable as d63's grading rubric.

## Numbers

**Re-pinned 2026-08-13** (Phase 0, `phase0_verify.ipynb` item 4): the table
below replaces the 0.751/0.738 and 0.693/0.625 vintages, which described a
~24K-row dataset that no longer exists on disk. Decisive rows, current data:

| protocol                 | router (LR) | const-dense | headroom captured |
| --------------------------| -------------| -------------| -------------------|
| random_within_lane       | **0.712**   | 0.631       | +0.237            |
| holdout_lane (rarb-math) | **0.667**   | 0.607       | +0.153            |

Auto-fusion column removed until the gates run: the classifier cache is empty
and the old 0.381/0.384 were measured on the retired vintage.

**The caveat that outranks the table** (Phase 0.5 diagnostic, pending formal
Phase 1): against C1's actual bar — the **per-lane best constant on answerable
rows** — the router is *negative*: −0.015 (lane-equal, ≥100-row lanes,
SE 0.006) to −0.0196 (row-weighted, SE 0.0025). The router beats const-dense;
a customer picking the right constant per collection beats the router. The
decisive-rows table above is the flattering slice, kept for continuity.

## What's under the hood

Two one-vs-rest logistic binaries: `dense-wins?` and `sparse-wins?`. Thresholds
`t_dense`, `t_sparse` decide when each binary fires; both below threshold routes
to `pure_rrf` as a hedge. `pure_rrf` is never trained; it's the residual.

Features: 57 engineered columns from the taxonomy catalog plus 3 derived columns
computed at fit/serve time (`identifier_density`, `avg_word_length`,
`short_id_query`). No embedding, no LightGBM yet.

## What changed since last week

Shipped baseline was `router 0.678 / 0.608`, losing to constant. Two setup-side
changes to `router.py` (both SPEC d47, no new data):

- **F2 features**: added the three derived columns above, dropped
  `length.length_words`. In the baseline that column was the top sparse-puller
  at +1.55 (a corpus artifact, since long BEIR / CRUMB queries lean sparse
  here). After F2 the artifact is gone.
- **F1' tuner**: 67% of the tune frame is threshold-invariant (all-tied and
  all-zero rows have identical or zero scores across route choices). The
  argmax was diluted; tuner picked `t_sparse = 0.9` (sparse effectively never
  fires). The fix filters the tuner to `routes_differ` rows only. Sparse now
  fires at `t_sparse ≈ 0.6–0.8`.

F1 was tried first as a class-balanced weighted mean over the full frame.
Sent the router into an "always rrf" degeneracy; reverted. Recorded as a
negative result in SPEC d47(a).

## Where the router still fails

`http://localhost.com` routes to dense at `p_dense = 0.99`. That's wrong;
BM25 handles a URL query trivially. Root cause is coverage. Of 2,510 decisive
rows, 3 have any URI feature. All 3 are long documents that mention a URL,
all 3 have dense as the winner. Zero rows are short + URI. The LR learned
"URI feature → dense" from three biased examples.

Feature engineering can't fill an archetype cell with no data. The setup
pass hit the ceiling of what surface features alone encode about corpus
outcomes on this training set.

## Auto-fusion is not a fair comparator

Auto-fusion routes `http://localhost.com` correctly (score 8 → sparse). It
loses on our aggregate eval because only 4.1% of our decisive sparse-winners
are short + identifier, the one archetype auto-fusion recognizes. The rest
are long natural-language queries where sparse wins by corpus-vocabulary
coincidence, which auto-fusion has no way to see.

Auto-fusion is a surface classifier; our labels come from corpus outcomes.
They answer different questions. The gap between what auto-fusion produces
from surface alone and what our labels contain is corpus knowledge. Surface
heuristics can't recover it; the labelling pipeline supplies it.

## Issues on the critical path

Ordered by impact on the current ceiling.

1. **Features live in only one routing context.** URI: 3 rows total, all
   dense-labelled. The LR sees only one side of the correlation. Each
   feature needs positive and negative examples across route labels for the
   model to learn the underlying rule.
2. **Archetype coverage holes.** The (feature × length × domain) grid has
   empty cells. Zero short-URI rows across the whole set.
3. **Feature-diversity target vs routing-signal target.** The composition
   (SPEC d31/d32) was designed to cover the taxonomy. The 74/20/6 route
   distribution is a side effect; the recipe never targeted routing
   balance. Upstream of #1 and #2.
4. **Decisive filter throws 90% of labels.** 2,510 out of 24,338 rows train
   the model. The F3 experiment (training on routes_differ instead) hurt
   holdout, since thin-margin rows are noisier. The bottleneck is decisive
   row supply; a looser filter is the wrong fix.
5. **Near-duplicate pairs (~5.5%) may straddle splits.** ~~Train/test split
   isn't near-dup-aware.~~ RESOLVED 2026-08-13: `dup_clusters.parquet` exists
   (44,280 clusters, 5.6% duplicate rows, largest cluster 189 queries —
   built by `feasibility_gate.py`); Phase 1's grouped split consumes it.
6. **Corpus at labelling, absent from the model.** Labels bake in corpus
   knowledge; features contain only query knowledge. Structural gap; more
   query-side features can't close it.
7. **Composition query distribution matches neither production nor canonical
   sparse archetypes.** Median 15 words, natural-language-heavy. Auto-fusion
   finds nothing to bite on; Qdrant page-search production (0–9 tokens)
   isn't represented.

## The architecture (SPEC d63, grilled 2026-08-11)

`query → route` at inference, unchanged. Everything the model cannot see at
serve time — corpus stats, gold-doc stats, outcome rates, even the taxonomy
— enters as SUPERVISION on branches whose predictions feed the route
decision (feed-forward privileged wiring, Hoffman 2016). Serving is one
frozen-encoder pass plus a small MLP: no extractor, no corpus, no spaCy in
the artifact.

    TRAIN and SERVE — same forward pass:
    query ─► frozen bge-small (384) ⊕ char-3–5-gram SVD (~128) ─► encoder ─► z
            z ─► ĉ: 44 cell sigmoids                  [BCE + pos_weight vs predicate evaluation]
            z ─► â: corpus ⊕ gold-doc ⊕ outcome est.  [z-scored MSE]
            concat(z, ĉ, â) ─► route layers ─► dense_ok / rrf_ok / sparse_ok ─► serve rule

    Branch losses exist only at training; branch OUTPUTS serve at inference —
    the model imputes the privileged knowledge it cannot see.

Inputs are complete transforms, never curated features: bge carries the
semantics (supersedes d46's encoder line — arm 7's e5 control decides the
label-coupling worry empirically), the char-n-gram SVD carries every
surface detail with zero per-format judgment. No feature column supervises
the model; the only human-designed structures touching the weights are
dataset-native — cells, gold docs, measured outcomes. Per-feature
recoverability ("can the embedding see identifiers?") is an offline linear
probe: diagnostic, zero gradient.

### Arms (one eval harness everywhere)

| arm | variant                              | question it answers                        |
| -----| --------------------------------------| --------------------------------------------|
| 1   | the design                           | —                                          |
| 2   | + taxonomy features as input, â only | do features earn input status?             |
| 3   | no branches                          | does branch supervision help at all?       |
| 4   | shuffled branch targets              | information or regularization? (TMLR 2025) |
| 5   | LightGBM ×3 on arm-2 inputs          | is the MLP the right learner at 40K rows?  |
| 6   | fine-tuned bge + 3 heads, once       | the published ceiling (CIKM'21 recipe)     |
| 7   | multilingual-e5-small encoder        | was d46's label-coupling worry right?      |

### Guardrails

- The binding small number is **~15 lanes, not 40K rows**: leave-one-lane-out
  CV over every lane, spread reported — never a single holdout lane.
- Near-dup clusters (cos>0.95) never straddle a split (issue #5 above).
- Every labelled row trains: all_zero rows are masked from the route BCE but
  still supervise both branches; all_tied rows supervise all five outputs.
- Branch λ: z-scored targets, annealed down, cosine-gated (Du et al. 2018),
  tuned on the route validation metric only.
- Baselines in every table: const-dense, LR (+priority serve rule),
  auto-fusion, production hard classifier. `routes_differ` is the headline
  slice; the 7 archetype probes are the standing smoke test.
- Plain PyTorch in `src/encoder_router/` (own package); deps torch +
  lightgbm. Per-head temperature scaling folded into export weights; ONNX
  plus a pure-numpy twin.

### Sequencing (d63g)

Prototype on dataset v2 now; the arm comparison of record rides the v3
dataset build (taxonomy round-trip fix → d62 word-shape guard →
re-extraction, bundled with more data and better augmentation). v2 numbers
are shakedown; v3 numbers are reportable.

Full rationale: SPEC d63. Evidence: `docs/research/qpp-retrieval-routing.md`,
`docs/research/lupi-privileged-information.md`,
`docs/research/tooling-mlp-router.md`.

The trained model doubles as a dataset validator. Its per-archetype failure
map is the spec for the next composition or augmentation pass.
