# Saved volume-router comparison — 2026-09-08

The results support specific advantages of `zipf_input_nocorpus`, but do not establish
that it behaves better overall. It reproduces all five user-approved decisions,
handles identifiers in prose well, and changes routes less often under paraphrasing.
Its saved policy also sends many ordinary informational questions to sparse retrieval,
which incurs substantial losses on the original TREC queries in this held-out sample.

## What was tested

- The two actual volume-run checkpoints in
  `src/data/encoder_router/classifiers_union_200k/{no_branches,zipf_input_nocorpus}`.
  Both metadata files record `2026-09-08-untied::20:18`, BGE-small, and the same
  held-out lane, `trec-dl-2022`. The reconstructed training/evaluation union has
  233,247 text-bearing rows. These are not the cascade-run checkpoints under `models/`.
- Saved thresholds: base `[0.35, 0.90]`, Zipf `[0.30, 0.85]`, in sparse/dense order.
  Both use the notebook's additional RRF rule, absolute head difference below 0.07.
- 108 fresh behavioral queries: 36 families with three phrasings each. Expected
  routes and acceptable sets were written in `cases.json` before inference. They
  express a query-shape policy, not corpus-grounded retrieval truth. Concepts were
  assigned dense; exact lookups sparse (CVE also permits RRF); identifiers in prose
  sparse or RRF. All 108 lack an exact normalized-text match in the training pool.
- 144 formatting variants, the five user examples, and 25 extensions of those
  examples. Extensions include changes of entity, scheme, or acronym spelling and
  are not all strict meaning-preserving paraphrases; they are diagnostic only.
- 1,185 held-out rows re-encoded locally, including the notebook's 524
  `routes_differ` rows. These are unequal-score rows, not necessarily strict winners.
  No training, threshold optimization, retrieval calls, or paid inference was run.

The fixture hash and checkpoint hashes are in `results.json`. The runner asserts
matching experiment metadata and repeatable probabilities. Original user outputs
and the notebook's held-out capture values were reproduced. Existing models,
notebooks, and label files were not changed.

## Behavioral results

| Test | no_branches | Zipf |
|---|---:|---:|
| User's original preferred routes | 2/5 | 5/5 |
| Fresh concepts assigned dense | 22/36 | 16/36 |
| Fresh exact lookups meeting frozen rubric | 30/36 | 28/36 |
| Fresh identifiers in prose assigned sparse/RRF | 34/36 | 36/36 |
| All fresh queries meeting frozen rubric | 86/108 | 80/108 |
| Families with any route change across three phrasings | 19/36 | 14/36 |
| Route changes under formatting/capitalization | 0/144 | 0/144 |

The fresh acceptance difference, Zipf minus base, is -5.6 percentage points.
A paired bootstrap over the 36 families gives an exploratory 95% interval of
[-14.8, +3.7] points. This is uncertainty within the authored case mix, not a
population estimate for production traffic. Constant sparse meets 66.7% of the
frozen rubric; both models exceed that reference on this set.

A **post-hoc rubric sensitivity check** permits RRF on every exact identifier lookup,
consistent with the user's acceptance of RRF for CVE. Scores then become 88/108
for base and 87/108 for Zipf. The ranking is consequently not a robust behavioral
win for either model. No fixture expectations were changed after observing outputs.

Useful Zipf advantages: all tested URL extensions choose sparse, and all four
additional DHA-acronym phrasings choose sparse. The spelled-out chemical name
chooses dense in both models, a diagnostic spelling change rather than an asserted
error. Zipf's cats family is uneven: `Why does a cat purr?` chooses dense, but
`What causes cats to purr?` and `Explain why cats purr` choose sparse.

Two original examples, `Who likes Curling?` and `why do cats purr`, are exact
normalized-text matches in the training population. There are also 1,301 LIMIT
training rows containing `Who likes `. This shows exposure, not proof of memorization,
and excludes these examples as independent evidence of generalization.

## Retrieval results

All scores below use existing stored route measurements. The metrics do not establish
which route is best on an arbitrary new corpus.

| Evaluation slice | Rows | no_branches | Zipf | Zipf minus base |
|---|---:|---:|---:|---:|
| All unequal-score rows | 524 | 0.4196 | 0.4156 | -0.0040 |
| Original/natural queries, all | 75 | 0.3602 | 0.2827 | -0.0775 |
| Original/natural, unequal scores | 72 | 0.3752 | 0.2945 | -0.0807 |
| Synthetic, unequal scores | 436 | 0.4311 | 0.4397 | +0.0086 |

The shared-column union used by the notebook drops provenance. This audit restores it
from the same winning source rows, leaving text and scores unchanged. Synthetic rows
constitute 436/524 (83.2%) of the headline unequal-score evaluation; another 16 are
neither in the natural nor synthetic slice.

For the 75 original queries, paired row-bootstrap 95% CI for Zipf minus base is
[-0.1459, -0.0136]. For all 524 unequal-score rows it is [-0.0376, +0.0292]. These
are exploratory intervals for fixed models and this one collection, not estimates
of training-seed variability. Synthetic-row intervals may understate uncertainty
because generation families are not clustered. Only exact normalized-text overlap
was screened; no semantic near-duplicate guarantee is made.

The original-query slice favors dense: always-dense scores 0.5055, versus global
training-selected constant RRF at 0.3805. Zipf serves sparse on 50/75 queries;
base does so on 33/75. Both models underperform these two constants on this slice.

Concrete original-query disagreements:

| Query | Base route / score | Zipf route / score |
|---|---|---|
| how fast does a rabbit grow | dense / 0.8532 | sparse / 0.0000 |
| why was the massachusetts bay colony founded | dense / 0.9113 | sparse / 0.0255 |
| what is the name of a baby nurse | dense / 0.8801 | sparse / 0.0157 |
| what hazards come with making paint | dense / 0.1073 | sparse / 0.8388 |
| what does london breed stand for | sparse / 0.0443 | dense / 0.8292 |

## Policy diagnostic, not a proposed replacement

Swapping the two existing threshold vectors, without changing weights or searching
for new thresholds, shows that behavior is materially affected by policy:

| Model | Thresholds from | Fresh acceptance | Original-query capture |
|---|---|---:|---:|
| Base | Base | 79.6% | 0.3602 |
| Base | Zipf | 81.5% | 0.3107 |
| Zipf | Base | 83.3% | 0.3217 |
| Zipf | Zipf | 74.1% | 0.2827 |

Zipf with base thresholds improves both those measures relative to saved Zipf,
but worsens full unequal-score capture from 0.4156 to 0.3993. No variant should
be selected using this test set. The result supports investigating calibration
against an explicit workload and objective; it does not isolate the causal value
of Zipf features from retrained weights, stopping epoch, or learned representations.

## Reproduction and artifacts

From the repository root:

```sh
.venv/bin/python experiments/router_behavior_audit/audit.py
```

- `cases.json`: frozen behavioral rubric and diagnostic examples.
- `behavior_predictions.csv`: every input, predicted route, head outputs, and
  exact-text training-overlap flag for both models.
- `heldout_predictions.csv`: per-query scores, provenance, routes, and score deltas.
- `results.json`: aggregates, exploratory intervals, metadata, and hashes.

The evidence supports retaining Zipf as a useful candidate, particularly for technical
queries. It does not confirm a general superiority claim, and it provides a concrete
counterexample to treating the five approved examples as evidence of broad reliability.
