# LLM judges documents, never routes; route labels come from retrieval outcomes

Status: accepted (2026-07-28, SPEC d37; supersedes SPEC d35/d36)

The R1 calibration pilot asked an LLM for 4-level TREC-DL relevance grades and
gated on Cohen's kappa ≥ 0.6 against human qrels. It failed definitively, not
for lack of power: 4-level kappa 0.120–0.139 with a bootstrap CI upper bound of
0.181 (TREC-DL 2022, 500 stratified pairs), so 0.6 is out of reach at any
sample size. Prompt iteration was not the fix — 12 hand-crafted few-shot
exemplars moved kappa by ±0.02 with fully overlapping CIs (Haiku 0.348 → 0.328,
Sonnet 0.320 → 0.340 at grade≥2), and Sonnet ≈ Haiku throughout, so 3× the cost
bought nothing. `judge.py` and the pilot notebook were deleted (commit
`cae28fc`); `data/r1_pilot/` is the measurement audit trail.

Two design errors, recorded so they are not re-made:

1. **Wrong consumer.** The pilot measured per-document grade agreement, but the
   pipeline consumes a per-query route decision. Whether route agreement
   survives moderate per-document noise was never tested — that is d37's
   framing.
2. **Wrong premise about orcas.** The pilot existed because orcas was called
   "no qrels", requiring a judge to manufacture labels. ORCAS in fact ships
   18.8M click pairs mapped onto msmarco-document doc_ids — positive-only
   qrels, not unlabelable. With clicks, every composition dataset has label
   ground truth with zero LLM involvement.

Consequences: an LLM is never asked which route wins — dense-vs-sparse is a
property of the (query, corpus, index) triple, and IDF is a corpus statistic
the model cannot see (d37f); this is an information gap, so a stronger model
shares the blind spot. The judge's only remaining job is binary
(query, doc) → relevant hole-filling for the qrel-hole asymmetry (d37k),
deferred until the raw per-route hole rate is measured. Do not reuse the
cached r1_pilot judgments to evaluate a *binary* judge: they came from a
4-level prompt over pairs sampled from existing qrels, and the 155-vs-8
false-negative skew is an artifact of collapsing that rubric at grade≥2, not a
property of a judge asked the binary question directly.
