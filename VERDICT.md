# Strategy Router: viability verdict

**Verdict: NO-GO, for the shipped logistic-regression router, on this dataset, at the
signed bar.** The router loses to the per-lane best constant on answerable rows
(C1 FAIL, adversarially confirmed), and the probe gate is RED. The scope line below is
part of the verdict: this measured one model, not routing as an idea. The same run
produced a positive but inconclusive cross-lane readout (+0.0234 vs a train-global
constant in the hide-one-lane rotation): a lead for v2, not evidence of transfer.
Decided by the Wave 1 gate: "if C1 fails, write the verdict and stop."

Date started 2026-08-13, completed 2026-08-13 (Wave 1 evidence only; Wave 2 not run).
Operator: Andrei Cristea + Claude (Fable).
Commit SHA: afeab29 on `dataset-v2` (program artifacts; this file follows in the next commit).
Dependency-lock note (2026-08-13): `query-taxonomy` absolute `file:///Users/andrei/...` pin
replaced with relative poetry path dep (`{ path = "src/query-taxonomy", develop = true }`);
`poetry.lock` regenerated; sync removed dropped-engine leftovers (gliner2, accelerate, peft).
Test baseline: 201 passing, 92 failing: the known taxonomy round-trip reds (SPEC d63(g)),
domain-logic failures, not environment breakage.

## Precommitted thresholds (signed off by Andrei, 2026-08-12/13)

Copied verbatim from VIABILITY_PLAN.md §1 before any result was seen.

**Δmin = 0.02, set by Andrei 2026-08-13** (answerable-rows mean objective; §8). **The ship
bar is a conjunction (§8): GO requires C1 AND the three auto-fusion gates below.** A failed
or inconclusive gate blocks a GO exactly as C1 does.

**Three gates against auto-fusion (settled 2026-08-13 by Andrei).** The incumbent comparison
is layered from broad to adversarial:

- **Gate 1, statistical: the C1b table row.** (router − auto-fusion) over all answerable
  held-out rows, clustered bootstrap CI lower bound > Δmin. "Logically we are better."
- **Gate 2, realistic: the golden set.** The 30-query `golden_set_auto_fusion` under the
  probe gate's probability-ordering criterion: the A/B-shaped check on the query shapes the
  incumbent was calibrated around.
- **Gate 3, hyperrealistic: the disagreement set.** Queries where auto-fusion is known to
  serve the wrong route. Two parts: (a) the measured slice, held-out answerable rows where
  auto-fusion's served route contradicts the labelled winner; passes if the router's mean
  objective beats auto-fusion's on that slice with a paired clustered-bootstrap CI excluding
  zero; (b) a hand-authored confusing-query table, candidates harvested from the measured
  slice, expected routes hand-ruled by the author, frozen into `probes.py`.

| Claim | Metric                                                                                                                                  | Passes if                                                                      | Fails if                                                                         |
| -------| -----------------------------------------------------------------------------------------------------------------------------------------| --------------------------------------------------------------------------------| ----------------------------------------------------------------------------------|
| C1    | (router − best constant) mean objective, grouped split, answerable rows, constant chosen **on the training split**                      | clustered bootstrap 95% CI lower bound > Δmin                                  | CI upper bound < Δmin                                                            |
| C1b   | (router − auto-fusion) mean objective, same grouped split, same answerable rows, auto-fusion served cache-first from `AutoFusionRouter` | clustered bootstrap 95% CI lower bound > Δmin                                  | CI upper bound < Δmin                                                            |
| C2    | (router − train-selected best fixed fusion), paired, identical candidate sets at depth 50                                               | router ahead by > Δmin, or within ±Δmin **and** cheaper than **that same arm** | fixed fusion ahead, or equal quality with no cost advantage over it              |
| C3    | (router − best constant) on **one corpus from outside the pipeline** (§7.4), with the 16-lane rotation as supporting evidence           | external corpus positive by > Δmin, rotation not contradicting it              | external corpus not positive by > Δmin, or rotation pooled CI upper bound < Δmin |
| C4    | reranked routed vs reranked always-dense, same candidate depth                                                                          | routed ahead by > Δmin                                                         | CI upper bound < Δmin, or not applicable if the deployment has no reranker       |

**Three states, not two.** PASS, FAIL, and INCONCLUSIVE when the CI straddles Δmin.
INCONCLUSIVE is not a soft pass: the verdict reports "not proven", which under this standard
has a fail's consequence and a different explanation.

**The probe gate.** "Not proven" is an acceptable deliverable if and only if the router
surpasses the human-ruled expectation on obvious queries: the 7 archetype probes frozen in
`probes.py` AND the 30-query `golden_set_auto_fusion`. Even one query served the wrong
probability ordering is itself a bad verdict, whatever the aggregate CIs say. The condition
is on probability ordering, not only the argmax.

**Bootstrap estimator, precommitted.** Per-lane delta = row-mean of (router − baseline) over
that lane's answerable held-out rows; the pooled statistic = the unweighted mean of per-lane
deltas. Lanes are the diversity instrument, so no lane buys weight with row count. Within-
lane CI: resample duplicate clusters with replacement inside the lane. Pooled CI: resample
lanes with replacement, then clusters within each sampled lane. Train-side selections
(per-lane best constant, fixed-fusion winner) are made once on the training split and held
fixed across resamples: the bootstrap measures evaluation noise, not selection noise.
**Amended 2026-08-13 by Andrei, before any claim verdict, on Phase 0.5's finding: lanes
enter the pooled statistic only with ≥100 answerable held-out rows; smaller lanes are
reported per-lane but do not vote.** (Alternatives offered: row-weighted, both-as-primary-
plus-sensitivity, stop at the precommitted branch. Chosen: min-evidence rule.)

## Delta-min, and who set it

0.02, answerable-rows mean objective. Andrei, 2026-08-13. Candidates 0.01 and 0.015 offered
and declined. Chosen knowing the current model's answerable margin (0.019) sits below it.

## Feasibility (Phase 0.5, src/scripts/feasibility_gate.py, run 2026-08-13)

Estimator: the precommitted block above, replayed exactly. Duplicate clusters built from
stored bge embeddings (44,280 clusters over 46,142 rows; 5.6% of rows in >1-member
clusters, confirming d33's ~5.5%). dup_clusters.parquet written; Phase 1 reads it.

Design A (C1: grouped random split, router − per-lane train-selected constant):
  pooled CI half-width ±0.053   power at Δmin=0.02: 0.02
Design B (C3: hide-one-lane rotation, router − train-global constant):
  pooled CI half-width ±0.063   power at Δmin=0.02: 0.02

**As precommitted, NEITHER design can resolve Δmin = 0.02.** Diagnosis: the lane-equal
pooled statistic gives every lane one vote, and 32 of 41 lanes have <100 answerable
held-out rows (nine have ≤4; bright-aops has zero and is excluded, count corrected by
checkpoint 4). Their per-lane deltas
are noise (between-lane std 0.153, e.g. crumb-stack-exchange −0.609 on n=3,
beir-nfcorpus +0.300 on n=3), and lane resampling propagates that noise into the pooled CI.

Diagnostic re-poolings of the same deltas (NOT precommitted, decision required before any
claim verdict): lane-equal over the 9 lanes with ≥100 answerable test rows → between-lane
std 0.018, naive SE 0.006 (resolvable); row-weighted over all rows → naive SE 0.0025
(resolvable). Both variants have power; the all-lanes-equal variant does not.

Directional flag from the same diagnostic run (single grouped split, no Codex attack;
recorded as trajectory, not as a claim verdict): the C1 point estimate is NEGATIVE under
every weighting with power. Router − per-lane best constant on answerable rows:
−0.015 (lane-equal, ≥100-row lanes) / −0.0196 (row-weighted). The historical +0.019
margin was measured against GLOBAL const-dense; C1's precommitted bar is the per-lane
best constant, which sits ~3.7 points higher on current data (0.602 vs 0.565).

## Label bias (Phase 0 item 6)

Hole rate per route (share of stored top-10 with no judgment at all, pooled over 42 lanes;
phase0_verify.ipynb item 6): **dense 89.1%  sparse 89.7%  rrf 88.2%   gap −0.7 points**
Literature range for reference: dense 14-32%, BM25 ~6%

How much of the measurement this lets us trust: two separate readings.
(1) The feared between-route bias is ABSENT pooled: the dense−sparse gap is −0.007, so the
judgment pool does not systematically favor one route. The 61.7% dense skew in decisive rows
(3,149/5,100, current vintage) is not explained by holes.
(2) The absolute hole rate (~89%) is far outside the literature range: 22 of 42 lanes have a
median ≤3 judged docs per query, including the largest (msmarco, orcas, webfaq, gooaq,
clerc, all at median 1), so the objective is measured on very thin judgments and a route
retrieving relevant-but-unjudged documents scores zero. Per-lane spread is extreme
(trec-dl-2022 0.2%, antique 14-26%, dbpedia ~35%, BRIGHT family 80-100%), so per-lane deltas
in low-judgment lanes carry much less evidential weight than the same delta in a deeply
judged lane. Per-lane gaps also exist in both directions (freshstack penalizes dense by
~+17pts of holes; crumb-code/stack-exchange penalize sparse by ~+12pts) even though they
cancel pooled.

## Claims and gates

C1  record is real:           **FAIL: confirmed by adversarial checkpoint 2.**
    Delta −0.0154, 95% CI (t₈ on SE 0.0073, n_boot=1000 seed 0) **[−0.0322, +0.0015]**;
    checkpoint 4 replayed at 20,000 replicates: [−0.0324, +0.0016]. Grouped split,
    amended estimator (9 voting lanes, 32 reported-only). The CI sits entirely below
    Δmin = 0.02. The earlier "and below zero" clause is WITHDRAWN (the t-correction
    admits +0.0015 at the top).
    Robustness, all recomputed independently by the reviewer: negative at every
    pooling threshold (all-lanes −0.056, ≥25 −0.070, ≥50 −0.058, ≥200 −0.016,
    ≥500 −0.016), row-weighted −0.0196, median lane −0.0106, routes_differ-only
    −0.0209 (FAIL), split seeds 0–5 all FAIL (−0.0118…−0.0166), clustering at
    cos 0.85/0.90/0.95 all FAIL, oracle test-optimal baseline −0.0199, shipped
    75/25 tune protocol −0.0166, hedge δ=0.05 −0.0142 and δ=0.14 −0.0081.
    The ≥100-row amendment moved the estimate +0.041 IN THE ROUTER'S FAVOR; it
    did not manufacture the FAIL. 0 of 20,000 bootstrap replicates reached +0.02.

    Corrections adopted from checkpoint 2, superseding three earlier support numbers:
    (a) the −0.0074 "leakage estimate" was split noise, not leakage: the two test
    sets share only 20.2% of rows; true leaky-row share in the ungrouped split is
    3.1% with a mechanical contribution ≈0.0005, WITHDRAWN as a leakage figure;
    (b) the −0.0387 "corpus-dependence gap" mixed per-lane and global baselines:
    like-for-like (global constant both sides, same 9 lanes) is ≈ −0.0226, and the
    grouped-split-with-global-constant point is +0.0247;
    (c) the decisive-slice +0.0027 was a silent estimator switch (row-weighted)
    carried by one lane. Under the precommitted estimator it is −0.0244
    [−0.0622, +0.0134]; the row-weighted +0.0027 falls to −0.0183 once
    scirgen-geo-en's 87 rows are removed. No positive slice of the record survives
    the precommitted estimator.

    Rotation (supporting readout, train-global constant): +0.0234, t₁₉ CI
    ≈ [−0.0122, +0.0590], INCONCLUSIVE under its own ≥100-row rule (20 voting
    lanes). A non-precommitted subset onto the C1 split's 9 lanes gives +0.0474, but
    that slice follows from no rule, carries no CI, and falls to +0.0194 (below
    Δmin) without clerc alone (checkpoint 4's leave-one-out, the same test that
    retired the decisive slice). Read it as a lead, not a finding: the router may
    beat a one-size-fits-all constant across lanes; nothing here proves it.
    Per-archetype: losses concentrate on SHORT queries (−0.040 to −0.045; short rows
    are 14.5% of the held-out answerable set). All numbers: src/phase1_record.ipynb
    plus the checkpoint 2/4 records in the objections section: four checkpoint-2
    items (clustering variants at cos 0.85/0.90, fit substrates, the 75/25 tune
    figure, the percentile CI) remain reviewer-asserted without a committed script
    and are labelled as such; checkpoint 4 independently reproduced seventeen others.
C1b beats auto-fusion (G1):   NOT RUN: moot, recorded as unmeasured rather than passed.
    C1's failure already blocks the ship conjunction; deciding an already-decided
    conjunct would cost ~7.6K HTTP classifier calls on the precommitted held-out rows
    (~38K to fill the whole cache) against an empty cache (Phase 0 item 7: 0/46,142).
Gate 2 golden set:            FAIL: same instrument as the probe gate; 6 of 30 golden
    queries wrong under the shipped mixed criterion, 1 of 30 under pure ordering
    (src/phase1_record.ipynb probe cell; criterion split per checkpoint 4). Zero
    tolerance was the signed rule: FAIL either way.
Gate 3 disagreement set:      NOT RUN: requires auto-fusion routes per held-out query
    (empty cache, same ~7.6K-call cost). Moot for the ship decision; retains
    diagnostic value for the d63 encoder router and is the first thing worth buying
    if this reopens.
Probe gate:                   RED: 6 of 30 unique queries fail under the shipped mixed
    criterion (probability ordering for single-route expectations; band membership for
    banded and rrf cases, where a dense/sparse ordering is undefined). Checkpoint 2
    corrected the earlier 29/36: the 7 archetype probes duplicate golden-set queries,
    so the instrument is 30 unique cases. Under the pure-ordering criterion alone,
    1 of 30 fails ("explain quicksort": p_dense 0.410 < p_sparse 0.488 against an
    expected dense_only): RED either way at zero tolerance. Artifact correction
    (checkpoint 4): the δ=0 C1 router emits pure_rrf on 1/7,597 held-out rows, but
    the served δ=0.05 artifact (the one the probe table measures) routes
    1,219/7,597 (16.0%, thresholds 0.15/0.50) to rrf; the earlier "hedge is dead /
    banded rrf expectations fail near-construction" phrasing conflated the two
    artifacts and is withdrawn. Still flagged: C1 measures δ=0 while the probe table
    measures δ=0.05, two artifacts in one record, named rather than hidden.
C2  not available cheaper:    NOT RUN: the Phase 1 gate stopped Wave 2 ("no result left
    to defend against cheaper alternatives"). The §10 steering probe's finding stands as
    prior, not proof: RRF was the strongest fixed arm on the one lane measured.
C3  survives new corpus:      NOT RUN: no external corpus was ever selected or indexed
    (the §8 rule was never exercised), so no transfer claim is made in either direction.
    The hide-one-lane rotation is recorded as supporting readout only: +0.0234
    [−0.0101, +0.0568] vs a train-global constant, INCONCLUSIVE.
C4  survives a reranker:      NOT APPLICABLE: no reranker in the deployment (§8)
C5  represents a deployment:  SETTLED: generalized strategy-selection product for customers
    needing strong top-1, battle-tested first on Qdrant page-search. Standing limitation on
    every number below: the generalized tier has no workload histogram; the
    composition-selected query mix stands in for it.

## M1 latency

NOT RUN: Wave 2 was stopped by the Phase 1 gate. `SERVING_COST`'s ordering remains an
unmeasured assumption; if any successor model reaches a quality tie, this benchmark is
required before its C2-equivalent cost clause can be read.

## Scope of a failure verdict

This tested the shipped logistic regression, not routing in general. What the d63 encoder
router (the built v2) would have to beat: C1 and the three auto-fusion gates at Δmin = 0.02
on answerable rows, plus the probe gate at zero failures.

What this run hands d63 as targets, each with its number: (1) the bar is the per-lane best
constant, not const-dense: the LR loses to it by 0.015–0.020 while beating const-dense,
so the missing signal is per-corpus, exactly what d63's privileged corpus branch imputes;
(2) the cross-lane lead to test: +0.0234 (inconclusive) vs a global constant in the
rotation: promising, unproven; (3) the worst slice is SHORT queries (−0.040 to −0.045;
14.5% of the held-out set; whether that matches production is unmeasurable without a
workload histogram, per C5); (4) the record contains two artifacts: the δ=0 C1 router
never uses its hedge (1/7,597) while the served δ=0.05 form routes 16% to rrf; v2 must
pick one artifact and defend it under one protocol; (5) 6/30 golden cases wrong under the
shipped criterion, including "explain quicksort" read as sparse.

## What we measured

- 2026-08-12/13, Wave 0: handoff review. All scoping questions settled (VIABILITY_PLAN.md §8).
  Codex adversarial pass produced nine objections; all nine folded into the plan in place,
  each marked "Codex objection n" (ship-bar test defined, bootstrap estimand precommitted,
  probe gate criterion fixed to probability ordering, proxy-status of 23/30 disclosed,
  duplicate clustering deduplicated to Phase 0.5, C2 three-lane verdict rule, external-corpus
  selection rule, steering-probe bound made provisional, "structural scarcity" demoted to prior).
- 2026-08-13, Phase 0.5 (src/scripts/feasibility_gate.py + src/phase05_feasibility.ipynb):
  precommitted lane-equal estimator cannot resolve Δmin (±0.053/±0.063, power 2%); cause
  diagnosed (32/41 lanes <100 answerable test rows); estimator amended by Andrei
  (≥100-row min-evidence rule) before any claim verdict; dup_clusters.parquet built
  (44,280 clusters, 5.6% duplicate rows).
- 2026-08-13, Phase 1 (src/phase1_record.ipynb): C1 measured and FAILED; probe gate RED;
  rotation readout positive-inconclusive vs global constant; per-archetype diagnostic
  (short queries worst). All offline, zero retrieval calls, zero spend; wall-clock not
  instrumented.
- 2026-08-13, checkpoint 2 (Claude Opus adversarial agent, 28 tool calls): FAIL confirmed
  under every variant; three support numbers corrected (see Codex objections section).
- 2026-08-13, Phase 0 (src/phase0_verify.ipynb, executed; Brief 1 cross-checked): 15 checks,
  10 mismatches, all explained by ONE cause: **the dataset roughly doubled since the
  documented figures were written.** Working frame 46,142 rows (doc: 24,338); decisive 5,100,
  split dense 3,149 / sparse 1,731 / rrf 220 (doc: 2,510 / 1,858 / 507 / 145); ceilings moved
  UP: always-dense 0.565 / best-constant-per-collection 0.602 / oracle 0.660, headroom
  0.057 (doc: 0.481 / 0.511 / 0.557 / 0.046); 43 lanes with corpus.parquet, 42 oracle dirs
  covering all 46,142 rows (doc: 16 lanes). Neither published record reproduces exactly:
  today's run gives **router 0.712 vs const-dense 0.631 decisive/random-split (n=1,021)**,
  a third vintage beyond PLAN.md's 0.751/0.738 and route_experiments.ipynb's 0.693/0.625
  (n=1,082), consistent with labels being regenerated between runs.
- Clean findings from the same run: round-trip mismatches now 0 of 46,142 (documented ~147,
  fixed since); nfcorpus stale-cache rows now 12 (documented 323, cleaned);
  `autofusion_cache.parquet` does not exist: gate coverage 0%, every gate row costs one HTTP
  classifier call; `route_raw_scores` persists for ALL rows at top-10 depth (plan's "never
  written" premise was stale; verified across all 42 oracle dirs by Brief 1 and checkpoint 4,
  the notebook cell samples one dir). Gate 3, qrel holes, and C1/C1b replay are fully
  offline; the depth-50 relog is still required for exact fusion sweeps;
  collection_features.parquet does not exist anywhere under src/data (Brief 1 recursive
  search) → Phase 1.6 auto-skipped per its own precondition.

## Codex objections, including the ones we could not answer

Wave-0 pass: nine objections, all answered by plan edits (see "What we measured").
Checkpoint 2 (2026-08-13): **run by a Claude Opus agent instead of Codex**: the codex
plugin was unusable at run time (persistent CLI failures); substitution decided by Andrei.
Independence caveat recorded: the attacker shares a model family with the orchestrator,
though it ran with fresh context, read-only, briefed to refute, and recomputed every
number itself (28 tool calls). Eleven objections; outcome: **FAIL intact under every
attack** (five pooling thresholds, row/median weighting, six split seeds, three clustering
thresholds, oracle baseline, shipped tune protocol, two hedge widths, three fit substrates,
percentile and t CIs; every variant negative, 0/20,000 replicates ≥ +0.02).
Corrections adopted: t₈ CI replaces 1.96·SE (drops the "below zero" clause); "leakage
−0.0074" withdrawn (split noise; true mechanical leakage ≈0.0005); "corpus-dependence gap
−0.0387" restated like-for-like as ≈ −0.0226; decisive-slice +0.0027 restated under the
precommitted estimator as −0.0244 (one lane, scirgen-geo-en, carried the positive sign);
probe count corrected to 6/30 unique. Minor code defects logged, all immaterial and all
leaning in the router's favor: cross-lane cluster straddle (0.09% of clusters = 0.32% of
rows, 147 rows); split loop can place a tiny lane entirely in test (fired for 2 non-voting
lanes); probes.py banded cases score by band membership, not ordering (mixed bands have no
defined ordering).

Checkpoint 2 objections we could NOT answer (carried into the verdict as limitations):
(1) label validity: everything rests on bge-small + Qdrant BM25 pinned labels with the
89% hole rate; differential error between router-chosen and baseline routes cannot be
ruled out offline; (2) template-family duplication below cos 0.85 cannot be bounded by an
embedding threshold alone; (3) whether Δmin = 0.02 is the right unit on a scale where
39.6% of answerable rows are structurally zero-delta (all_tied), precommitted by the
author, consistent, but not validated by anything runnable.

Checkpoint 1 was subsumed by Phase 0's notebook cross-check of Brief 1 (noted rather
than silently skipped). Checkpoint 3 died with Wave 2.

Checkpoint 4 (2026-08-13, Claude Opus adversarial auditor, pre-verdict as §3.3 requires):
recomputed the load-bearing numbers from the shipped code; 13 findings, disposition
"publishable with the listed fixes", **all folded in place**: the rrf-hedge claim was
measured on the wrong artifact and is withdrawn/restated (finding 1); the rotation's
+0.0474 was a rule-less subset and is demoted with its clerc leave-one-out attached
(finding 2); the headline C1 interval is re-derived as [−0.0322, +0.0015] from the
shipped bootstrap (finding 3, the quoted [−0.0328, +0.0021] traced to nothing); four
checkpoint-2 numbers are labelled reviewer-asserted (finding 4); 74% → 61.7% (finding 5);
gate cost 38K → 7.6K held-out (finding 6); Gate 2 criterion split 6/30 mixed vs 1/30
pure-ordering (finding 7); this checkpoint recorded as run, not deferred (finding 8);
six → nine tiny lanes (finding 9); judgment-depth claim recomputed as 22/42 lanes at
median ≤3 (finding 10); the scirgen leave-one-out reattributed to the row-weighted slice
(finding 11); t-correction applied to both intervals and dated in VIABILITY_PLAN §1
(finding 12); provenance gaps closed or restated (finding 13: straddle unit, wall-clock
not timed, short-query share 14.5% stated as a composition fact, collection_features
verified absent by Brief 1's recursive search).

## Judge STOP verdicts, and how each was satisfied

None issued. The judge gate exists for unattended runs; the author was present for every
important decision (the C1 vintage pin, the estimator amendment, the checkpoint-2
substitution, the stop-at-Wave-1 call), so each was decided by him directly and recorded
with his name in place of a judge verdict.

## What we did not measure and why

- **C1b / Gate 1 and Gate 3 (auto-fusion comparisons):** ~38K HTTP classifier calls
  against an empty cache to decide a ship conjunction C1 had already blocked. Unmeasured,
  not passed. Gate 3's disagreement slice is the highest-value deferred measurement.
- **C2 (fixed-fusion sweep), M1 (latency), the depth-50 relog:** Wave 2, stopped by the
  Phase 1 gate. The §10 single-lane probe result (RRF strongest fixed arm) stands as
  prior, not proof.
- **C3's external corpus:** never selected; no transfer claim made in either direction.
- **The d44(d) judge spike:** no frontier-model budget; a local substitute would measure
  agreement with qrel-limited labels, which cannot settle validity; the 89% hole rate
  stays an open wound on every number here.
- **Phase 1.6 collection-tier probe:** its input artifacts (collection_features.parquet)
  do not exist; skipped per its own precondition.

## If GO: the three things budget should buy first

Not applicable.

## If NO-GO: what would have to change for this to be worth reopening

1. **A model that clears the real bar.** The d63 encoder router exists for exactly the
   measured gap (imputing per-corpus knowledge from the query). Reopen when an arm beats
   the per-lane best constant by > Δmin on answerable rows under this verdict's estimator;
   the harness, clusters, and notebooks here are the ready-made eval.
2. **Labels that can be trusted at the margin.** 89% qrel holes cap how much any 0.02
   claim can mean; deeper judgment pools (or the deferred judge spike) on even a few
   lanes would convert the three UNANSWERED objections into measurements.
3. **Production-shaped evidence.** The router's worst slice is short queries (the
   page-search shape), and they are 14.5% of the held-out answerable set. Whether that
   is under-representation cannot be said without the workload histogram C5 records as
   missing; getting one, plus a short-query stratum with real judgments, would let the
   next verdict speak for the deployment that matters.
4. **Cheaper auto-fusion comparison.** Fill the classifier cache once (~38K calls,
   embarrassingly parallel) and Gates 1–3 become permanent offline instruments instead
   of a spend decision.

---

*The two sections below are ported verbatim from the retired `VIABILITY_PLAN.md` (§10, §11). Their `§n` cross-references point into that plan, which is in git history at commit 26b9a93 — the rest of it was either copied into this file before any result was seen or superseded by SCOPE_DECISION.md.*

## Steering probe, run 2026-08-12

One assumption tested in advance, on a machine with no DVC pull and no repository install, to check whether Phases 2 and 3 are worth their compute. Scripts and the six commands to reproduce are in `docs/probe/`: a throwaway venv with only `fastembed` and `qdrant-client`, a curl of the BEIR nfcorpus zip, an isolated Qdrant on port 6399. Nothing local is touched.

**Setup.** nfcorpus, one of the 16 lanes: 3,633 documents, 323 test queries, median 16 judged per query, matching WEAKNESSES #10. Dense `bge-small-en-v1.5`, sparse `Qdrant/bm25` with the IDF modifier, both logged at depth 200 with raw scores, scored with 0.7·HitRate@1 + 0.3·NDCG@10.

### 10.1 Result

| fixed arm | mean objective |
|---|---|
| rrf_k10 | 0.4584 |
| rrf_k60 | 0.4562 |
| rrf_k2, Qdrant default | 0.4536 |
| weighted, w=0.5 | 0.4300 |
| weighted, w=0.7 | 0.4277 |
| weighted, w=0.3 | 0.4257 |
| DBSF | 0.4222 |
| dense_only | 0.4027 |
| sparse_only | 0.3963 |

Best constant of the three project routes: `pure_rrf`, 0.4536. Best fixed arm overall: `rrf_k10`, 0.4584. Per-query oracle over the three routes 0.5037; over all nine arms 0.5297. Decisive queries 23 of 323, 7.1%.

**What it changes.**

1. **RRF is not the weak link.** Every RRF variant beats DBSF by ~0.034 and weighted fusion by ~0.028. `PureRRFStrategy`'s premise, that rank fusion dilutes a confident top-1 and that this is what routing exists to avoid, does not hold here: RRF is the strongest fixed arm. Moving to another fixed fusion buys +0.0048 over the best project route. **This does not test C2**, which is router versus best fixed fusion, and there is no router arm here. What it establishes is that the alternatives the project deleted without measuring are worse than the one it kept, which lowers the odds a better fixed fusion is waiting. §7.1 is bounded on that basis, not because C2 was answered.
2. **Tuning k is not worth a campaign.** k=2, 10, and 60 span 0.005. If the sweep reproduces that, the knob is closed.
3. **The wider action space is where the ceiling moves.** Nine arms raise the oracle from 0.5037 to 0.5297, and that 0.026 is half again the 0.0502 the three-route oracle offers over the best constant. It is a ceiling, not an achievement, and more arms means more classes over the same thin decisive supply. But the three-route frame was a choice with a measurable cost, which belongs in the next plan.
4. **Best constant is per lane, and the reported bar may be wrong.** Here it is `pure_rrf`, with dense back at 0.4027; globally it is always-dense, and PLAN.md compares against const-dense while d44(a) records best-constant-per-collection as 6.3% better than always-dense. Different populations, so they cannot be subtracted, but the shape is real: the router may be beating an easier bar than a customer would deploy. Hence C1's per-collection bar.
5. **Decisive scarcity recurs off-pipeline.** 7.1% here against 10.3% overall, from an independent implementation — consistent with scarcity being structural, but one lane on a non-comparable population (see limits) cannot settle it, so treat this as a prior, not a finding.

**Limits, all unrepaired.** No router arm, so nothing speaks to C2 directly. "All 323 test queries" is not the project's *answerable* set, which drops `all_zero` rows (`labels.py:39-64`), so the means and the 7.1% are not like-for-like with 10.3%. Fusion is Python, not Qdrant's `Fusion.RRF`, so calling k=2 "the Qdrant default" is unverified against server ranking and ties. The sparse arm uses the IDF modifier while `src/composition/indexer.py` leaves `modifier=None`, so it may be a different sparse system until label provenance says otherwise. DBSF's mean ± 3σ and the weighted arms' per-query min-max are this probe's normalizations, and three weights are not a sweep. One lane, one encoder pair, no confidence intervals. The five points above are the most this can support; anything stronger needs the trained router in the comparison.

---

## Cut, with reasons

Named so nobody re-adds them mid-run.

- **Offline DBSF or weighted-fusion sweep from stored artifacts.** Impossible: raw component scores were never persisted (`golden.py:66`, `objective.py:59-63`). This was the plan's original centrepiece, and it is why Phase 2 exists.
- **Exact RRF-k sweep from stored rankings.** Ranks 11 onward are gone, so a document can enter a fused top-10 when k changes and no one-sided bound exists. A top-10-only approximation is triage; keep triage numbers out of the verdict.
- **The d44(d)/d45(g) judge spike.** Needs a frontier model and there is no budget. A local substitute measures agreement with qrel-limited labels, which cannot settle viability.
- **LightGBM, new features, LUPI, composition redesign, augmentation runs.** Downstream of the verdict; d45(h) already gates composition and augmentation behind the measurement.
- **Auto-fusion comparison as evidence.** ~~A surface classifier scored against corpus-outcome labels answers a different question. The bar is the best constant.~~ **REVERSED 2026-08-12 by Andrei:** auto-fusion, the deployed LLM-based classifier, is a first-class comparison baseline even though the product target is broader than that deployment. Auto-fusion is the second conjunct of the ship bar — C1b in §1's table; a ship requires beating both it and the best constant by > Δmin (d37(h)'s best-constant bar survives as the first conjunct, C1). The machinery exists: `AutoFusionRouter` (`router.py:715`) with cache at `route_labels/autofusion_cache.parquet`, already pluggable into `RouterExperiment` (`router.py:828`). Phase 0 reports the cache's row coverage so the marginal HTTP-call cost of scoring it on held-out rows is known before Phase 1 commits to it.
- **Another single-lane `rarb-math` holdout run.** One lane cannot support a transfer claim.
- **Manufactured sub-collections.** Shards of one corpus are not independent corpora, so this raises the collection count more than the evidence. Settled in §8: collections are a diversity instrument, and C3 rests on the external corpus.

### Objections heard and kept anyway

An adversarial pass argued for two deletions this plan declines, recorded so the argument is neither lost nor re-litigated mid-run.

- **The judge gate and named worker roles** were called bureaucracy that consumes attention without producing data. They stay because this plan runs mostly unattended, and the failure mode they guard against, an agent reinterpreting a threshold after seeing a result, costs more than the overhead. §3.4's boundary between the latitude and the gate exists because of that objection.
- **Per-archetype eval** was called a diagnostic with no branch in the verdict, which is true. It stays because it is cheap and is the only thing explaining why a claim passed or failed. It is now labelled a diagnostic so nobody mistakes it for evidence.

The same pass argued the central framing was benchmark-internal, that the experimental claims could all pass on a product with no buyer. That was accepted: it became C5, settled in §8 by the two-tier deployment target.
