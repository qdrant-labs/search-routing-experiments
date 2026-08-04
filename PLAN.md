# Router: current state

The router is a `query → route` function shipped as an artifact. Route is one
of `dense_only`, `pure_rrf`, `sparse_only`. Deployment target is unknown at
ship time. No corpus features enter at inference.

## Numbers

Two protocols, evaluated on decisive rows (winner hit rank 1, runner-up missed):

| protocol                 | router (LR) | const-dense | auto-fusion | headroom captured |
| --------------------------| -------------| -------------| -------------| -------------------|
| random_within_lane       | **0.751**   | 0.738       | 0.381       | **+0.057**        |
| holdout_lane (rarb-math) | **0.611**   | 0.604       | 0.384       | **+0.017**        |

Router beats constant on both protocols. Auto-fusion (the LLM incumbent) is
lower here, but the comparison is not apples-to-apples. See below.

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
5. **Near-duplicate pairs (~5.5%) may straddle splits.** Train/test split
   isn't near-dup-aware. Random_within_lane numbers may be inflated by
   duplicate leakage.
6. **Corpus at labelling, absent from the model.** Labels bake in corpus
   knowledge; features contain only query knowledge. Structural gap; more
   query-side features can't close it.
7. **Composition query distribution matches neither production nor canonical
   sparse archetypes.** Median 15 words, natural-language-heavy. Auto-fusion
   finds nothing to bite on; Qdrant page-search production (0–9 tokens)
   isn't represented.

## The architecture question

The router is `query → route` at inference, and that constraint is fixed.
Corpus features can enter at training. The model needs two kinds of
generalization:

- **Feature generalization**: learn "URL surface → probably sparse" as a
  durable rule. Coverage and a learner that captures interactions.
- **Corpus generalization**: route correctly on any target corpus without
  seeing it at inference. The model has to internalize typical corpus
  effects on each kind of query from training-time corpus signal. This is
  Learning Using Privileged Information (Vapnik 2015). Features are present
  at training, masked at inference.

Trees can approximate the first. Corpus generalization needs an
encoder-style architecture: a small MLP with structural inductive bias, not
a transformer. Three implementations, cheapest first.

**A. Dropout on corpus features.** Model takes `(query, corpus)`. Corpus
features are randomly zeroed during training, probability rising toward 1;
at inference they're always zero. Cheapest to prototype. The inductive bias
is weak, since the model just learns to fall back to query-only patterns
when corpus is missing.

**B. Auxiliary corpus reconstruction.** Small MLP encoder on query features,
with two heads. One predicts corpus features from the encoder's latent
(auxiliary loss at training, dropped at inference); the other predicts
route, consuming the same latent. The auxiliary head forces the latent to
encode what corpus this query would typically live in. Middle cost,
clearest inductive bias for corpus generalization.

**C. Teacher-student distillation.** Teacher trained on `(query, corpus)`
with full supervision. Student trained to mimic teacher's decisions from
query alone; the student ships. Heaviest (two models, two training loops),
but the teacher-student gap quantifies exactly what the corpus buys.

The trained model doubles as a dataset validator. Its per-archetype failure
map is the spec for the next composition or augmentation pass.
