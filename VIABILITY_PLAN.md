# Strategy Router: viability plan

**Status: EXECUTED 2026-08-13 — verdict NO-GO, see `VERDICT.md`.** Wave 1 ran (Phases 0, 0.5, 1); C1 failed and the Phase 1 gate stopped Wave 2. This document is kept as the frozen record of what was precommitted; the numbers inside it that predate execution (e.g. the 24,338-row figures, the "74% dense skew") describe the retired dataset vintage and are superseded by VERDICT.md.

**Status before execution: READY.** Wave 0 (the handoff review) completed 2026-08-12/13: every scoping question is settled in §8, the §1 thresholds are signed by Andrei, and a Codex adversarial pass ran against the final document — all nine of its objections are folded in place, each marked "Codex objection n".

A runbook for one Claude Code session. Output: a go/no-go verdict on the Strategy Router, backed by numbers that meet a "we only ship what we prove" standard. It does not improve the router; it decides whether the router should exist. Every phase can return "no", ordered so the cheapest "no" arrives first.

Authored 2026-08-12 from PLAN.md, WEAKNESSES.md, SPEC.md (d30, d37, d41, d44, d45, d47, d60), TODOS.md, and `src/`, with three adversarial passes folded in. Numbers below were read from those documents; Phase 0 re-verifies them against pulled data before anything is built on them.

### How to treat this document

**The phases are a suggestion. The goal in §1 is not.** This was written before the data was in hand, so parts of it are guesses about what the measurements will look like. When a result points elsewhere, go there. You are expected to drop a step whose answer is already implied (§10.1 already demoted one), add an experiment that bears on a claim more directly, reorder when a cheaper falsifier appears, and stop early: if C1 fails, write the verdict and end the plan rather than finishing the rest as theatre.

Two things are not yours to change. The **goal** in §1, meaning the product question and the five claims. And the **precommitted thresholds**, which can be revised only before a result is seen, and only for a measurement error rather than a disappointing number. §3.4 holds that line. Every deviation gets one line in `VERDICT.md`: what you changed, and what evidence changed it.

---

## 1. The goal

### The product question

Is there a retrieval-strategy selection product that beats the best single fixed strategy, on a corpus it has never seen, by a margin large enough to ship?

Strategy means one of `dense_only`, `pure_rrf`, `sparse_only`. The bar is a conjunction: the best fixed strategy per SPEC d37(h) — a selector that loses to always-dense is worthless — AND the deployed auto-fusion classifier, through the three gates below. "Never seen" is the deployment reality, since d47(b) fixes the API at `query → route` with no corpus input. The objective's 0.7 weight on HitRate@1 matches the settled target (§8): customers who need a strong top-1 result.

### The five claims

C1 through C3, with C1b and the gates in the threshold block, are falsifiable by experiment, and a failure in any is a no-go. C4 is settled NOT APPLICABLE — no reranker in the deployment. C5 is settled by the author: the two-tier target recorded in §8.

**C1. The record is real.** The gain survives a leakage-safe split and holds over all answerable rows, not only the 10.3% decisive subset.

> `router.py:909-918` says in a comment that near-duplicate-aware splitting is deferred, so duplicates can straddle train and test. The reported margin is 0.751 vs 0.738, a gap of 0.013, against ~5.5% near-duplicate pairs: contamination and result are the same order. Training and evaluation both filter to decisive rows (`router.py:258-289`, `:879-895`), so the headline describes a subset selected after retrieval. The bar is per-collection best constant, not global always-dense: §10.1 measured `pure_rrf` as best constant on nfcorpus while dense wins globally, so a global bar flatters the router.

**C2. The gain is not available more cheaply.** No fixed fusion configuration reaches the same quality with no selector at all.

> The hybrid arm is Qdrant default RRF with no exposed weight or k (`fusion.py:162-179`). Weighted RRF and DBSF were removed because a continuous weight is not a label a router can emit (`fusion.py:21-27`), which is about output format, not a measurement that they are worse. If a better fixed fusion closes the gap, a customer sets one config value and needs no router.

**C3. The gain survives an unseen corpus.** It holds under a hide-one-lane rotation across all 16 lanes, not one chosen lane.

> Only `rarb-math` is ever held out (`router.py:733-734`, `:920-924`), so transfer is a sample of one, and d45(e) called for the full rotation. The deeper limit: all 16 lanes came out of the same harvesting pipeline, so they are 16 draws from one process rather than 16 independent corpora, and no quantity of extra queries changes that. This is why the primary test is one corpus from outside the pipeline (§7.4) and the rotation is supporting evidence.

**C4. The gain survives a reranker.** **NOT APPLICABLE — settled 2026-08-13 by Andrei: no reranker in the deployment, just sparse, dense, hybrid targets over the same search primitives.** Kept in the table so the verdict reports it as settled rather than silently passed.

> Reranking appears nowhere in SPEC, PLAN, WEAKNESSES, or the source. If production reranks the top-50, all three routes feed the same reranker and route choice may stop mattering. The counterargument, to test rather than assume: a reranker cannot recover a document that never entered its candidate list, so routing may still change recall at depth.

**C5. The measurement represents a deployment.** The labelled queries, qrels, objective, index configuration, and workload weighting resemble a real target deployment closely enough that a measured delta means something to a customer.

> C1 through C4 are benchmark-internal: they can all pass on composition-selected, qrel-limited queries against a stack nobody deploys. Labels are pinned to bge-small plus Qdrant BM25 (WEAKNESSES #9); qrel holes run 14–32% for dense against ~6% for BM25 and are unmeasured here (WEAKNESSES #8); d44(a) states outright that this dataset is not an optimal routing dataset; no real workload histogram exists. C5 was settled by the author naming the target — §8, first entry — with the workload-histogram gap carried as a standing limitation.

### Precommitted thresholds

**Signed off by Andrei, 2026-08-12/13 (§8). Copy this block into `VERDICT.md` verbatim before the first experiment.**

Everything hangs on **Δmin, the smallest objective gain that would make him ship**: "the CI excludes zero" is not a ship test, since with 24,338 rows a useless delta clears it and across 16 correlated collections a useful one can fail it. Every test is superiority against Δmin.

**Δmin = 0.02, set by Andrei 2026-08-13** (answerable-rows mean objective; §8). **The ship bar is a conjunction (§8): GO requires C1 AND the three auto-fusion gates below.** A failed or inconclusive gate blocks a GO exactly as C1 does.

**Three gates against auto-fusion (settled 2026-08-13 by Andrei).** The incumbent comparison is layered from broad to adversarial:

- **Gate 1, statistical — the C1b table row.** (router − auto-fusion) over all answerable held-out rows, clustered bootstrap CI lower bound > Δmin. "Logically we are better."
- **Gate 2, realistic — the golden set.** The 30-query `golden_set_auto_fusion` under the probe gate's probability-ordering criterion: the A/B-shaped check on the query shapes the incumbent was calibrated around.
- **Gate 3, hyperrealistic — the disagreement set.** Queries where auto-fusion is known to serve the wrong route (e.g. *"What do you think of Dr. Jenkins' take on paleolithic diets?"* read as identifier-heavy when it is a plain opinion question). Two parts, built in Phase 1.4b: **(a)** the measured slice — held-out answerable rows where auto-fusion's served route contradicts the labelled winner; passes if the router's mean objective beats auto-fusion's on that slice with a paired clustered-bootstrap CI excluding zero (zero, not Δmin: the slice is adversarial and small, and the demand is "wins where the incumbent fails", not "wins by two points" — revisit before results if that is too lax); **(b)** a hand-authored confusing-query table — candidates harvested from the measured slice, expected routes hand-ruled by the author — frozen into `probes.py` beside the golden set and reported like the probe table.

| Claim | Metric | Passes if | Fails if |
|---|---|---|---|
| C1 | (router − best constant) mean objective, grouped split, answerable rows, constant chosen **on the training split** | clustered bootstrap 95% CI lower bound > Δmin | CI upper bound < Δmin |
| C1b | (router − auto-fusion) mean objective, same grouped split, same answerable rows, auto-fusion served cache-first from `AutoFusionRouter` | clustered bootstrap 95% CI lower bound > Δmin | CI upper bound < Δmin |
| C2 | (router − train-selected best fixed fusion), paired, identical candidate sets at depth 50 | router ahead by > Δmin, or within ±Δmin **and** cheaper than **that same arm** | fixed fusion ahead, or equal quality with no cost advantage over it |
| C3 | (router − best constant) on **one corpus from outside the pipeline** (§7.4), with the 16-lane rotation as supporting evidence | external corpus positive by > Δmin, rotation not contradicting it | external corpus not positive by > Δmin, or rotation pooled CI upper bound < Δmin |
| C4 | reranked routed vs reranked always-dense, same candidate depth | routed ahead by > Δmin | CI upper bound < Δmin, or not applicable if the deployment has no reranker |

**Three states, not two.** PASS, FAIL, and INCONCLUSIVE when the CI straddles Δmin. INCONCLUSIVE is not a soft pass: the verdict reports "not proven", which under this standard has a fail's consequence and a different explanation.

**The probe gate (settled 2026-08-12 by Andrei, instrument widened 2026-08-13).** "Not proven" is an acceptable deliverable if and only if the router surpasses the human-ruled expectation on obvious queries. The instrument is both fixed sets: the 7 archetype probes frozen in `probes.py` (5 hand-ruled, 2 corpus-dependent) AND the 30-query `golden_set_auto_fusion` (route_experiments.ipynb, banded 0–9 expectations checked via `_production_route`). Even one query served the wrong ordering — e.g. an exact-error-string query with p(dense_only) > p(sparse_only) — is itself a bad verdict, whatever the aggregate CIs say, because it is a failure to capture a large theoretical difference. `VERDICT.md` reports the full table (query, expected, served, predicted probability vector — the condition is on probability ordering, not only the argmax). Standing at 2026-08-13, measured under **proxy criteria only** (band membership for the golden set, argmax for the probes — not the precommitted ordering test, which `probe()` cannot yet produce; Codex objections 3 and 4): 23/30 golden, 4/6 decided probes (notebook cells `0545a10f`, `probes-run`). The gate's official reading is produced in Phase 1 under the ordering criterion; the proxies say RED is the likely starting state, consistent with the author's own statement that the model is not yet generalizable to all edge cases. Phase 1 freezes the 30-query set out of the notebook into `probes.py` so the gate reads a fixture, not a cell.

**One measurement, not a claim: M1, latency.** p50 and p95 per route against a local Qdrant, including the query-side embedding pass and the router's own feature extraction and classification, which is work the baselines do not do. It cannot produce a no-go on its own, since nobody kills a project over latency when quality holds and latency is moot when quality fails. It exists because it settles the `SERVING_COST` assumption that has been outstanding since `fusion.py:38-46` was written, and because **C2's cost clause reads it**: a router that ties a fixed fusion on quality passes C2 only if M1 shows it cheaper than that same arm.

Three rules that make the table reproducible.

- **Baselines are selected on training data, never on held-out rows.** Picking best constant or best fixed fusion by looking at test labels makes the baseline optimistically selected while the router stays fixed. If the product story is that a customer tunes the constant on their own data, say so and give the baseline the same tuning budget the router gets.
- **The bootstrap resamples clusters, not rows — estimator precommitted 2026-08-13 (Codex objection 2).** Rows are dependent through duplicate clusters, source templates, lane, and shared qrel pools. The estimand: per-lane delta = row-mean of (router − baseline) over that lane's answerable held-out rows; the pooled statistic = the **unweighted mean of per-lane deltas** — lanes are the diversity instrument (§8), so no lane buys weight with row count, and msmarco's 32% of rows does not get 32% of the verdict. Within-lane CI: resample duplicate clusters with replacement inside the lane. Pooled CI: resample lanes with replacement, then clusters within each sampled lane. Train-side selections (per-lane best constant, fixed-fusion winner) are made once on the training split and held fixed across resamples: the bootstrap measures evaluation noise, not selection noise. Copy this paragraph into `VERDICT.md` verbatim before the first run. **Amended 2026-08-13 by Andrei, on Phase 0.5's finding: lanes enter the pooled statistic only with ≥100 answerable held-out rows; smaller lanes are reported per-lane but do not vote.** The all-lanes-equal form could not resolve Δmin (±0.053 vs 0.02) because 32 of 41 lanes carry too few answerable test rows for their means to be evidence. **Second amendment, 2026-08-13 (adopted from checkpoint 2, applied to every interval in the record): with lanes as the resampling unit and few of them, the CI uses t with (voting lanes − 1) degrees of freedom, not 1.96·SE.**
- **Every arm sees identical candidate sets at identical depth.** Labels were produced at `fetch_limit=50` (`fusion.py:73-85`). Giving a new arm depth 200 measures the depth change.

Headroom captured, the figure PLAN.md quotes, is `(candidate − best constant) / (oracle − best constant)`. Report it beside the raw delta, never instead: it flatters small absolute gains.

### The honest prior

d44(a) records the ceiling over all 24,338 rows: always-dense 0.481, best constant per collection 0.511, per-query oracle 0.557. Total headroom 0.076. The same decision cites arXiv:2504.01101, where published selective-processing systems *achieve* about 4% or less of that, and whose Figure 4 reports single-predictor selection "seldom outperform[s] the individual system."

The prize is small and most attempts miss it. That is the reason to precommit thresholds and to treat a clean no-go as a successful outcome, not a reason to skip the measurement.

### Out of scope

No LightGBM, no new features, no augmentation run, no composition redesign, no LUPI. All downstream of "should this exist", and d45(h) already gates composition and augmentation behind the measurement.

One consequence to state in the verdict rather than hide: this plan evaluates the shipped logistic regression, and the built v2 is the d63 encoder router (`src/encoder_router/`, privileged branches imputing corpus knowledge from the query; d45(c)'s LightGBM survives as arm 5 of its seven-arm comparison). So a C1 or C3 failure is evidence that **this model** is not viable, not that routing is impossible. The verdict must draw that line and name what the encoder router would have to beat: C1 and the three auto-fusion gates at Δmin = 0.02 on answerable rows, plus the probe gate at zero failures.

### Waves and checkpoints

Three waves, three conversations with Andrei, and no more than that. Each wave ends with a decision about whether the next one is worth running.

| Wave | What runs | Cost | Ends with |
|---|---|---|---|
| **0. Scope** | **Done 2026-08-12/13.** The handoff review; §8 is its record | one conversation | Signed thresholds — delivered |
| **1. Offline** | Phase 0, 0.5, 1. No retrieval, no new labels, nothing but the parquets already on disk | two to three days | C1, the feasibility readout, the label-bias readout. **Checkpoint with Andrei** |
| **2. Retrieval** | Phase 2 relog, then Phase 3. Local Qdrant, overnight runs | three to four nights | C2, M1, the external-corpus transfer read (C4 cut — no reranker, §8). **Final verdict review** |

**Wave 1 is where most of the risk dies.** It can retire the published headline, and it can report that the design cannot resolve Δmin at all, either of which ends the project without a single retrieval call. Do not start Wave 2 before its checkpoint.

**The Wave 1 checkpoint is the only mid-flight consultation.** Bring three things: C1's verdict with the leakage delta, Phase 0.5's power readout with a plain statement of which claims this design can and cannot resolve, and the hole-rate readout. Then one question: given these, is Wave 2 worth three to four nights? Everything else in Wave 1 is supporting detail and does not need his time.

---

## 2. Prerequisites, a hard gate

Do not start Phase 1 until every line passes.

**Machine.** Apple Silicon Mac, Docker running, ~40GB free, able to run unattended overnight, Python 3.11–3.14.

**The install portability bug is FIXED (2026-08-13).** `pyproject.toml` pinned `query-taxonomy` by an absolute `file:///Users/andrei/...` path; it now resolves via `[tool.poetry.dependencies] query-taxonomy = { path = "src/query-taxonomy", develop = true }` and the lock was regenerated (the sync also removed the dropped GLiNER leftovers: gliner2, accelerate, peft). Note in `VERDICT.md` that the lock changed, since that changes the dependency set the evidence was produced under.

**DVC.** SATISFIED 2026-08-13: the operator confirms `src/data` is already at the latest version locally — no pull needed. (For any other machine: install `dvc[gs]`, remote `gs://qdrant-hybrid-search-rrf/data`, `gcloud auth application-default login`, `dvc pull` fetches 410 files / 6.5GB.)

**spaCy model.** `poetry run python -m spacy download en_core_web_sm`, a separate download from the package.

**Qdrant.** `docker compose up -d`, then `curl localhost:6333/healthz`. Collections are **not** in DVC; they live in the docker volume. Whether they exist locally decides whether Phase 2 needs a re-index.

**Baseline health.** `poetry run pytest` and `poetry run ruff check src/query_taxonomy tests`. The documented baseline at 2026-08-13: **201 passing, 92 failing** — the 92 are the known taxonomy round-trip reds SPEC d63(g) records as blocked on the taxonomy repo's re-extraction, domain-logic failures (`taxonomy_generators/registry.py:45`), not environment breakage. A count other than 92 red is a new problem; investigate before proceeding.

**Working directory.** Notebooks under `notebooks/` read `data/` relatively and expect `src/` as cwd.

**No spend.** No paid API calls. d44(d)/d45(g)'s judge spike needs a frontier model and is out of scope. Do not substitute a local model and present it as that spike: it measures agreement with qrel-limited labels, which cannot settle viability.

---

## 3. Tools, workers, and rules

### 3.1 Skills

| Skill | When |
|---|---|
| `/qdrant-advisor` | Before any indexing, collection config, or retrieval-performance work in Phase 2. |
| `/codex:rescue` with `--model gpt-5.6-terra` | The four checkpoints in §3.3 and nowhere else: the operator is on a free plan. Brief it for refutation, never approval. |
| `/ponytail` | Active by default. The smallest script that answers the question, not a framework. |
| `/dataviz` | Only if the verdict needs charts. Read before writing chart code. |
| `/humanizer` | On `VERDICT.md` before a human reads it. |

### 3.2 Rules carried over

- **Never commit or push** without an explicit in-turn request. Write files, stage if useful, then stop and say it is ready. Hard rule; a productive rationale does not override it.
- **Verify before claiming done: run the thing.** Report failures plainly with output. A phase is not complete because the code exists.
- **Do not trust a subagent or Codex at face value.** Check every load-bearing claim against the file or the data before it enters `VERDICT.md`. Codex over-compresses; diff its output against a recorded list of numbers and identifiers before accepting a rewrite.
- **Qdrant is a vector search engine**, never a "vector database", in everything this plan produces.
- **macOS "Operation not permitted"** under `~/Documents` is a lapsed TCC grant. Stop retrying, ask the operator to restart the terminal.

### 3.3 Who does what

Fable orchestrates, Opus builds, Fable judges, Codex attacks.

| Worker | Model | Allowed to | Never |
|---|---|---|---|
| Orchestrator | Fable | Own the plan, sequencing, and `VERDICT.md`. Decide. Do small work directly when delegating costs more | Accept a builder's code without reading the diff; skip a §3.4 judge gate |
| Builder | Opus | Write one experiment's script to a spec, run it, return the path and the numbers | Change scope, touch files outside its spec, interpret a result as a verdict |
| Judge | Fable | Check one decision for grounding, goal-fit, over-engineering | Write code, propose a redesign |
| Grounding agent | Haiku | Locate, count, catalogue what is on disk | Edit anything, interpret a result |
| Research agent | Sonnet | Read many files or sources, return a synthesis | Edit anything |

Reading a parquet's shape, a grep, a ten-line pandas check: do it directly. A script that will be committed and rerun: give it to a builder. Two builders at most in flight, no subagent spawns its own. Long runs go in background shells with a `Monitor` on the log, not a polling loop. Launch independent subagents in one message.

**Token discipline.** SPEC.md is 163KB, TODOS.md 68KB, CONTEXT.md 36KB. Never tell a subagent to read one whole: give a decision number, a line range, or have it grep. Builders get a spec, not this plan. Judges get the decision and its evidence, not the transcript.

**Builders may edit the repository; the read-only rule below covers the grounding and research agents only.** This overrides normal practice, where delegated edits have introduced bugs with summaries that did not match the diff. Two mitigations: every builder writes a new standalone script rather than modifying existing modules, and the orchestrator reads the full diff and runs the script itself before accepting a number. An edit to `src/hybrid_search_rrf_dataset/` goes back to the orchestrator.

**In every subagent brief**, since they do not inherit this file: read-only, no edits. Every number carries the path and line or parquet column it came from. If you cannot verify something, say so rather than inferring it from documentation. Report what contradicts the brief's framing.

#### Brief 1, grounding (Haiku), Phase 0, parallel with the reproduction run

> Read-only inventory of `src/data` after `dvc pull`. Report each number and where it came from. (1) `labels.parquet`: total rows, non-null `route`, rows per `dataset`, count per `outcome_shape`. (2) Rows passing `decisive_rows()` in `router.py`, and the winner split across the three routes. (3) Which lanes have a `corpus.parquet`, and each document count. (4) Whether `collection_features.parquet` and `lane_corpus_stats.parquet` exist, and their shape. (5) Whether any golden rows carry a non-empty `route_rankings`, and the longest list length in that column. Compare 1 and 2 against the documented 24,338 labelled / 2,510 decisive / dense 1,858 / sparse 507 / rrf 145, and flag every mismatch. Item 5 decides whether the Phase 2 relog is needed, so be precise.

#### Brief 2, research (Sonnet), Phase 1, launched with 1.1 and read before 1.4

> Read-only. What does this repository already know about near-duplicate queries and per-lane routing outcomes? Search SPEC.md, TODOS.md, WEAKNESSES.md, `notebooks/`, `src/composition/`. With file and line: (1) every recorded near-duplicate measurement, its method and threshold, lanes affected, and whether any cluster assignment was written to disk. (2) Any existing per-lane table of route outcomes, best-constant route, or decisive counts, including notebook output cells. (3) Anywhere the ~5.5% figure appears with a different number or threshold, which would mean it is less settled than it looks. If (2) turns up nothing, say so: the 16-lane rotation then has no prior to check itself against.

#### The four Codex checkpoints

Free plan, so four briefs total, one per phase, never one per experiment. `--model gpt-5.6-terra`, briefed with the code and the numbers, asked to refute. Record its objections in `VERDICT.md` including the ones you cannot answer.

1. **After Phase 0.** "Here are the reproduced numbers against the documented ones. Attack any declared match that is not one, and any documented figure accepted without reproducing it."
2. **After Phase 1, the important one.** "Here is the grouped-split implementation, the bootstrap procedure, and the resulting CIs. Attack the statistics: clustering threshold, whether grouping actually prevents leakage, whether the bootstrap respects cluster and lane structure, whether the all-answerable comparison is like-for-like. Then tell me whether C1 passed or whether I fooled myself."
3. **After Phase 3.** "Here is the fixed-fusion sweep, the latency bench, and the external-corpus run. Attack the DBSF and weighted normalizations, whether every arm got a fair depth and cutoff, and whether the external corpus was selected by the §8 rule and labelled exactly as the lanes were. Then tell me whether C2 and C3 survive."
4. **Before the verdict.** "Here is `VERDICT.md`. Every claim should trace to a number this plan produced. Find the ones that do not, and any threshold reinterpreted after seeing a result."

If the quota runs out before checkpoint 4, protect checkpoint 2: it gates the only claim that can retire the existing record.

### 3.4 The judge gate

Every important decision gets a fresh Fable judge before it is acted on, because an unattended agent drifts three predictable ways: accepting an ungrounded number, wandering from viability into improving the router, and building a framework where a script would do.

**Important, and needs a judge:** declaring any claim passed or failed; reinterpreting a threshold, where the default answer is no; deciding an experiment is not worth running or a result is close enough to skip a step; building something not named here; proceeding after a prerequisite or data check failed; writing or revising the verdict; escalating from the three-lane pilot to all 16 in Phase 2, which is where the compute goes.

Routine work gets no judge: writing a script to spec, running it, reading a parquet, fixing your own bug. Spinning up judges for those burns budget and teaches the orchestrator to ignore them.

**Judge brief, verbatim, three slots to fill.**

> You are the judge on an unattended experiment plan. You write no code and propose no redesign. Read `VIABILITY_PLAN.md` sections 1 and 11 only, then judge one decision.
>
> The decision: `<what the orchestrator is about to do>`
> The evidence offered: `<numbers, file:line, script output>`
> The alternative considered: `<what else was on the table, or "none">`
>
> 1. **Grounded?** Is every number traceable to a file, a line, or printed output? Name anything asserted without provenance. A figure quoted from PLAN.md or SPEC.md is documentation, not measurement; say so when it is being treated as measurement.
> 2. **On goal?** Does this advance the verdict on the five claims in section 1? If it improves the router, widens scope, or answers an adjacent question, say so and name which claim it was meant to serve.
> 3. **Over-engineered?** Is there a smaller thing that answers the same question, or a step whose cost exceeds what it resolves? Section 11 lists what was deliberately cut; check none of it is being smuggled back.
> 4. **Verdict:** PROCEED, PROCEED WITH CHANGE (name it in one line), or STOP (name what must happen first).
>
> Return STOP if the decision reinterprets a threshold after a result was seen, unless the reason is a measurement error rather than a disappointing number. If you cannot tell whether a decision is routine or important, return STOP and say which. Ambiguity defaults to stopping.

**The latitude and this gate can be played against each other, so here is the boundary.** Changing *how* you measure, *which* rows enter a comparison, *what* the baseline is, or *whether* a claim passed is always important, whatever it is called. The latitude covers the route to the verdict: skipping an implied step, adding an experiment, changing the order. It does not cover the estimand. An agent that reclassifies a method change as routine and logs it afterwards has defeated the precommitment this plan exists to protect.

**With a verdict.** PROCEED: act. PROCEED WITH CHANGE: apply it, or record in `VERDICT.md` why not. STOP: do not act; fix what the judge named or escalate to the operator. STOP verdicts belong in `VERDICT.md` even once satisfied, because they are the record that the plan supervised itself.

### 3.5 Artifacts

| Path | What |
|---|---|
| `VERDICT.md` | The verdict, thresholds, every number, every Codex objection. The deliverable. |
| `src/scripts/feasibility_gate.py` | Phase 0.5: CI width and power against Δmin. |
| `src/scripts/relog_components.py` | Phase 2: the retrieval pass that logs raw component scores. |
| `src/scripts/fusion_sweep.py` | Phase 3: fixed-fusion candidates. |
| `src/scripts/latency_bench.py` | Phase 3: the benchmark `SERVING_COST` has been waiting for. |
| `src/data/relog/<lane>.parquet` | The relog. Add to DVC; do not commit the data. |

Each script gets one runnable assert-based self-check. No frameworks, no fixtures.

---

## Wave 1 — offline

## 4. Phase 0: verify the record (half a day)

Confirm the documented facts hold on disk before testing claims about them.

1. Does `labels.parquet` hold 24,338 labelled rows, 2,510 decisive under `decisive_rows()`, winner split dense 1,858 / sparse 507 / rrf 145?
2. Reproduce always-dense 0.481, best constant per collection 0.511, per-query oracle 0.557.
3. Are all 16 lanes materialized with a `corpus.parquet` each?
4. Reproduce the record by running the existing `RouterExperiment`. **Two vintages are in circulation and Phase 0 must say which one C1 defends:** PLAN.md quotes 0.751 / 0.738, while route_experiments.ipynb's current validate cell shows router 0.693 vs const_dense 0.625 on decisive rows (random_within_lane, n=1,082) and 0.699 vs 0.680 on answerable rows (cell `65885f55`, n=8,206). If neither reproduces, stop and report: nothing downstream is interpretable.
5. Quantify the two known defects so no number silently includes them: the `GoldenRoutingBuilder` parquet round-trip inconsistency (~147 of 46K rows where `route_rankings` contradict `route_scores`) and the stale `beir-nfcorpus_oracle` cache (323 rows from an unservable cell). Neither blocks this plan.
6. **Count the qrel holes per route.** For each route's stored top-10, what share of retrieved documents carry no judgment at all? Report per route, per lane, and the dense-minus-sparse gap.
7. Report `route_labels/autofusion_cache.parquet` coverage: cached rows vs the 24,338 labelled queries, per lane. This prices the auto-fusion gates (settled 2026-08-12, §11 reversal) before Phase 1 commits to scoring them.
8. Verify §6's premise: do raw per-strategy component scores persist anywhere on disk or in the golden pipeline (Andrei recalls they may, 2026-08-13)? If yes, the Phase 2 relog shrinks to gap-filling or is unnecessary — check before any retrieval is scheduled.

Delegate 1–3 to Brief 1. Run 4, 6, 7, and 8 yourself.

**Why item 6 matters more than its cost.** Every delta this plan computes assumes the labels are unbiased between routes, and nobody has checked. The BEIR literature puts Hole@10 at 14–32% for dense retrievers against about 6% for BM25, and WEAKNESSES #8 records that the direction and magnitude here are unmeasured. A route that retrieves relevant-but-unjudged documents scores zero for them, so a large gap means the 74% dense skew in the decisive rows is partly an artifact of who contributed to the judgment pool. This is pure counting over `route_rankings` joined against `QrelStore`: no judge, no model, no spend, minutes of work. It cannot fix the bias, but it tells the verdict how much of the measurement to trust, and a gap far outside the literature's range is itself a finding about this dataset. If the gap is large, say so beside every number rather than in a footnote.

**Kill condition.** If the published numbers do not reproduce, the verdict is "the record is not reproducible" and the plan stops. That is a complete answer.

## 4.5 Phase 0.5: check the measuring instrument, not the router (half a day, no retrieval)

**Binding. Run before Phase 1.**

**This phase proves nothing about the router.** It checks whether our measuring
procedure is capable of seeing the effect we promised to look for. The analogy: we agreed
to ship only if the router is 20 grams heavier than the baseline — Phase 0.5 checks whether
the scale we own wobbles by ±5 grams or by ±50. If the scale wobbles ±50, every weighing
session ends "can't tell", and that outcome was decided by the scale, not by the object.
Discovering that AFTER three nights of experiments would be the expensive way; the wobble
can be computed today, from the scores already on disk, because CI width depends only on
the data design (how many queries, how many collections, how noisy each collection's
average is, how many queries are near-duplicates of each other) — never on how good the
router actually is.

Mechanics: build the duplicate clusters once (pinned bge-small embeddings, cos > 0.95,
written to `dup_clusters.parquet` — the same artifact Phase 1.1 reads), then replay the
exact split-and-average procedure the §1 thresholds name, on both designs (C1's grouped
random split; C3's hide-one-lane rotation). Two numbers come out per design:

- **Resolution — the CI half-width.** Must be smaller than Δmin = 0.02, or every
  experiment ends INCONCLUSIVE by construction. Concretely: a half-width of ±0.05 means a
  router that is *truly* 0.02 better produces intervals like [−0.03, +0.07] — straddling
  the ship bar every single time, guaranteed, whatever the truth is.
- **Power — the chance of detecting a real effect.** If the router truly were 0.02 better,
  how often would this procedure actually say PASS? Power of 0.02 means one time in fifty:
  running that experiment is buying a lottery ticket, not taking a measurement.

Three precommitted outcomes: resolvable under both designs → run Phase 1, record achieved
power. Resolvable within-corpus but not across lanes → C3 is declared a limit of the
dataset *before* it runs, so a later negative rotation reads "the design cannot answer
this" rather than falsely reading "the router does not transfer". Resolvable under
neither → stop; "this dataset cannot decide the question at the margin that matters" is
the cheapest complete answer available.

A judge-gate decision, since proceeding past a failed feasibility check is exactly what an
unattended agent talks itself into.

**RUN 2026-08-13 — outcome: branch three, with a repairable cause.** Half-widths ±0.053
(C1 design) and ±0.063 (C3 design) against Δmin = 0.02; power at a true 0.02 effect: 2%.
The wobble comes from one identifiable place: the precommitted pooling gives every lane
one equal vote, and 32 of 41 lanes have fewer than 100 answerable held-out queries (six
have ≤4) — a 3-query lane average can swing ±0.5 by pure luck, and one-vote-per-lane
feeds that luck straight into the pooled CI. The same deltas re-pooled with a
minimum-evidence rule (only lanes with ≥100 answerable rows vote: 9 lanes) give ±0.012 —
resolvable. Decision on amending the estimator is Andrei's, recorded in VERDICT.md; see
`src/phase05_feasibility.ipynb` for every number.

## 5. Phase 1: is the record real? (one to two days, no retrieval)

The cheapest phase that can retire the headline. Offline pandas and scikit-learn over already-labelled rows.

### 1.0 Same-depth, train-selected baseline replay

**First, and cheap.** Re-score the existing router and every constant baseline on identical candidate sets at one fixed depth, with the constant chosen on the training split alone. Today the comparator is picked with knowledge of the held-out rows (`router.py:939-959`), and the fixed-fusion arms proposed later would run at a different depth than the router's labels. Both make a delta that is partly an artifact of unequal conditions.

**No retrieval needed; Phase 2 is not a prerequisite.** `labels.parquet` stores all three route objective values per query (`labels.py:138-155`), derived from `route_scores` (`golden.py:347-364`), produced at the default `fetch_limit=50` (`fusion.py:73-85`).

**But `RouterExperiment` cannot run it as written.** `_run_one()` filters held-out data to decisive rows before scoring (`router.py:869-895`) and the existing `all_rows` argument changes fitting only (`:258-289`). Write a small offline evaluator: predict routes on every non-`all_zero` held-out row, choose each lane's constant from training rows alone, score both from the stored `score_*` columns.

If this replay alone moves the margin materially, say so immediately: everything after it is precision around a number that has already changed.

### 1.1 Near-duplicate clusters

The 5.5% figure was measured once (d33) but clusters were never persisted, which is why `_split_random` defers duplicate-awareness. **The clusters are built once, in Phase 0.5** (`dup_clusters.parquet`, keyed (dataset, query_id, cluster_id)); this step reads that artifact — never rebuilds it (Codex objection 5) — and reports the realized rate per lane. A material difference from 5.5% is itself a finding.

### 1.2 Re-run both protocols, grouped

Add a grouped split to `RouterExperiment` keeping a whole duplicate cluster on one side of the line, then re-run `random_within_lane`. The grouped-versus-ungrouped delta is the leakage estimate. **This number decides whether PLAN.md is quotable.**

### 1.3 Report over all answerable rows

Keep training on decisive rows (d45(a) has a real argument for it), but add an evaluation scoring every answerable row, meaning everything except `all_zero`. A router that wins on 10.3% of rows and is neutral elsewhere has a much smaller production effect than the headline implies. Both numbers appear side by side.

### 1.4 The full 16-lane rotation

Replace the single `rarb-math` holdout with hide-one-lane across all 16, reporting per-lane deltas with CIs and the pooled estimate. Also report the (random-grouped) − (rotation) gap, which is d44(c)'s corpus-dependence measurement.

### 1.4b Baseline arms and the probe table

Two additions settled 2026-08-12/13. (1) Score the auto-fusion arm (`AutoFusionRouter`, cache-first) on the same held-out rows as 1.0's replay — **this is C1b, the second ship-bar conjunct, judged by its own table row in §1**. The production hard identifier-count classifier is scored beside it as a reported arm with no threshold. (2) Extend `probes.py:probe()` to read `router.explain()` — `predict()` alone cannot show probability ordering (Codex objection 3) — freeze the 30-query golden set out of the notebook into `probes.py`, and record the full table. The probe gate in §1 reads the ordering columns: one hand-ruled query with the wrong probability ordering is a red verdict line on its own. (3) Build Gate 3's disagreement set (§1): join held-out answerable rows against the auto-fusion cache, keep the rows where auto-fusion's route contradicts the labelled winner, and score router vs auto-fusion on that slice (paired, clustered). Harvest the most confusing of those queries as candidates for the hand-ruled table; the expected routes are Andrei's to rule, not the orchestrator's.

### 1.5 Per-archetype eval — diagnostic, no threshold reads it

Group held-out rows by feature signature (`has_uri`, `has_uuid`, `is_short`, `is_math`, `is_natural_language`) and report captured headroom per group. WEAKNESSES #12 has it as cheap and pending. It explains *why* a claim passed or failed, and it turns "aggregate noise, qualitative gain" into a statement about which query shapes the router helps.

### 1.6 Collection-tier probe — optional

**Only if the corpus indexes already exist.** An adversarial pass argued for deleting it and the argument is sound: it costs a full corpus-tokenization pass, no threshold reads its output, and it is circular, training a 16-row classifier to predict a target derived from the same decisive labels.

`src/scripts/collection_features.py` implements d47(f): a corpus index per lane, then leave-one-out prediction of each lane's best constant from six corpus statistics. It has never been run and no result is recorded. If the indexes exist, run it; the leave-one-out takes seconds. If not, skip it.

Read the result carefully. The band is pre-committed at ≥12/16, and the script warns when the score fails to beat the majority baseline. Since dense is likely the best constant in most lanes, that baseline may already sit at or above 12, making the band unearnable. **Treat it as a no-go detector only:** at or below chance (≤7/16), or at or below majority, is strong evidence against the corpus-statistics mechanism. A pass is encouraging and proves nothing.

### Phase 1 gate

Codex checkpoint 2 reviews the grouped split and the CIs, briefed to attack the statistics. **If C1 fails, write the verdict and stop.** Phase 2 does not run, because there is no result left to defend against cheaper alternatives.

## Wave 2 — retrieval

Do not start before the Wave 1 checkpoint.

## 6. Phase 2: the relog (one overnight, plus a pilot)

One retrieval pass producing the artifact four experiments need. It exists because persisted `route_rankings` are top-10 document IDs only (`golden.py:66`, via `objective.py:59-63`) and `route_scores` are final objective floats. Raw cosine and BM25 scores, and ranks 11 onward, were never written, so DBSF, weighted fusion, an exact RRF-k sweep, depth sensitivity, and any reranking probe are impossible rather than approximate.

**`relog_components.py`.** For every labelled query in a lane, query Qdrant twice and persist `(dataset, query_id, route, rank, doc_id, raw_score)`. No fusion, no scoring, no objective. Three requirements a first draft will get wrong:

- **Log both depths, 50 and 200.** Depth 50 is parity, the depth the labels came from, and the only depth valid for comparison against the router. Depth 200 answers a separate question, reported separately. Mixing them silently is the easiest way to manufacture a result.
- **Reproduce the original depth-50 route scores as the acceptance test.** Re-score the three routes from the relog and check against stored `route_scores`. A mismatch means the index or config differs from the one that made the labels, and everything downstream is invalid.
- **Log provenance.** Collection name, Qdrant server version, dense and sparse model ids and revisions, index and quantization config, corpus snapshot identity, qrels version. `docker-compose.yml` pins `qdrant/qdrant:latest`, so capture the version at run time.

The reranking probe needs query and document *text*, which the relog does not carry; it reads text from `corpus.parquet` by `doc_id`. Keep the relog narrow.

**Pilot three lanes first:** `beir-nfcorpus` (3,633 docs), `rarb-math` (the incumbent holdout, all three classes substantial), `msmarco-passage-dev` (100K docs, 32% of rows). Measure wall-clock, then decide whether 16 lanes fit in one night.

**Risk and fallback.** Collections are not in DVC. If the docker volume is empty each lane must be re-indexed, meaning every corpus document embedded. README says `dvc pull` fetches caches, so check for a usable embedding cache before assuming a full re-embed. If msmarco's 100K must be embedded from scratch it may consume the night: run the two small lanes first and let msmarco have the second. Invoke `/qdrant-advisor` before configuring collections, and record the exact index config, since labels are stack-pinned and a config difference silently invalidates the comparison.

## 7. Phase 3: cheaper, worth anything, and does it transfer? (one to two days)

All three run off the relog.

### 7.1 Fixed-fusion sweep, C2 — bounded

The steering probe (§10.1) already found RRF the strongest fixed arm on nfcorpus with k worth about 0.005, so this confirms rather than hunts. Time-box it — but the probe ran without stack parity (Python fusion, IDF-modified sparse; §10 limits), so the bound is provisional: if the parity sweep contradicts it, any fixed arm beating RRF by more than Δmin, the time-box is void and the sweep widens (Codex objection 8).

**The C2 candidate set, at depth 50 only**, scored per query against the same qrels and `RouterObjective`, on the same held-out rows Phase 1 used: RRF at several k including Qdrant's native default; DBSF; weighted score fusion over a weight grid with normalization declared.

**Select the winner on the training split, then compare once.** Sweeping k, weights, and normalizations against held-out rows and reporting the best is how a fixed arm wins by search rather than quality. Then answer one question: does that fixed fusion, one config value a customer sets once, reach the router's quality? If yes, and the router has no cost advantage over that same arm, the router has no proven reason to exist and the verdict is no-go regardless of C1 and C3.

**Depth and cutoff sensitivity is separate.** Fetch depth beyond 50 and `top_k` at 5, 10, 20 change the candidate set and the metric, so they cannot sit inside the C2 comparison without turning it into a depth experiment. Report them as their own sensitivity readout.

Report the oracle ceiling over the widened arm set at depth 50 too. If it is much higher than 0.557, the three-route action space was the wrong frame, which is a finding for the next plan rather than an argument to continue this one.

**Scope — verdict rule precommitted 2026-08-13 (Codex objection 6).** The C2 state is rendered from the pilot's three contrasting lanes: fixed fusion ahead by > Δmin there → **FAIL** (a bar already beaten on contrasting lanes is not rescued by more lanes); router ahead by > Δmin → **PASS on available evidence**, with the all-lane run required before any customer-facing C2 claim; within ±Δmin → escalate to all lanes behind the §3.4 judge gate before rendering a state. The verdict names which scope it reports.

### 7.2 Reranking probe, C4 — CUT

No reranker in the deployment (§8): C4 is NOT APPLICABLE and nothing here runs. Recall@50 per route still gets reported from the relog — it costs nothing and shows where depth-level value lives.

### 7.3 Latency benchmark, M1

Warm and cold p50 and p95 plus throughput for the three routes against a local Qdrant, including the query-side embedding pass, which is why sparse is assumed cheaper, and including the router's own feature extraction and classification.

Two outputs: whether `SERVING_COST`'s ordering is real, and what fraction of end-to-end latency routing can save. Feeds C2's cost clause. Not a gate on its own: if the saving is a few milliseconds inside a pipeline spending hundreds, record it and let the quality claims decide.

### 7.4 One external corpus, the real C3 test

**The most informative experiment in the plan, and the one the 16-lane rotation cannot replace.** All 16 lanes came out of the same harvesting pipeline, so they are 16 draws from one process rather than 16 independent corpora. That is why the rotation can only ever be suggestive about a corpus nobody has seen.

Take one corpus that has never touched this project, with its own queries and its own judgments, chosen by the precommitted selection rule in §8 (public, native human qrels, outside the registry, uncovered domain — hand-written queries are excluded). Index it with the pinned stack, label it exactly as the 16 lanes were labelled by running all three routes and scoring with `RouterObjective`, then evaluate the router trained on all 16 lanes against it, with the best constant chosen from the 16.

That is the new-customer question asked once, properly. Its result outranks the rotation in the verdict: if the router beats the best constant by more than Δmin on a corpus from outside the pipeline, C3 has evidence the rotation cannot supply, and if it does not, no rotation number should be reported as transfer.

**Scope discipline.** One corpus, chosen for being genuinely external rather than for being favourable, and named in the verdict before the result is known. A single corpus is n=1 and cannot carry a confidence interval, so report the per-query CI within it and be explicit that corpus-level uncertainty is unquantified. Pick something small: the point is independence, not scale.

---

## 8. Settled decisions

Wave 0 ran 2026-08-12/13 as the handoff review. Every scoping question was answered by Andrei; each decision's consequences are already edited in place throughout the plan. This section is the record of which constraints are fixed and why — the executing agent does not reopen them.

- **Target deployment (C5).** A generalized retrieval-strategy product for customers who need strong top-1 results, battle-tested first on Qdrant's page-search, where the auto-fusion classifier ships today. Scope is not limited to the page-search query mix. Standing limitation carried into the verdict: the generalized tier has no workload histogram, so the composition-selected query mix stands in for it. — Andrei, 2026-08-12
- **"Not proven" is deliverable, conditionally.** Acceptable if and only if the probe gate (§1) passes: the 7 archetype probes plus the 30-query `golden_set_auto_fusion`, zero tolerance on probability ordering. The gate reads RED today (23/30 golden, 4/6 decided probes). — Andrei, 2026-08-12, widened 08-13
- **API.** Inference-time query-only, not negotiable. Corpus knowledge is imputed in-model: the d63 encoder router (`src/encoder_router/`) supervises privileged branches (44-cell archetype head, ~30-dim corpus/gold-doc/outcome head) on dataset-native artifacts at train time, so the model learns a latent representation of the generalized corpus from the query alone. C3 is the hard test of that imputation bet; the failure-scope line names d63 as the v2 this verdict does not evaluate. — Andrei, 2026-08-13
- **Ship bar.** A conjunction: beat BOTH the best constant AND auto-fusion. Beating auto-fusion while tying the best constant is not a ship. Refined same day into the three-gate structure in §1: Gate 1 statistical (C1b, > Δmin), Gate 2 the golden set (realistic), Gate 3 the disagreement set (hyperrealistic — queries auto-fusion gets wrong). — Andrei, 2026-08-13
- **Reranker.** None: the deployment is sparse, dense, and hybrid targets over the same search primitives. C4 NOT APPLICABLE; the reranking probe and its research brief are cut. — Andrei, 2026-08-13
- **Δmin = 0.02**, answerable-rows mean objective. Chosen with the consequence on the table: the current model's answerable margin (0.699 − 0.680 = 0.019, route_experiments.ipynb cell `65885f55`) sits below it, so absent a real gap-opening C1 is expected to read INCONCLUSIVE or FAIL — the intended bar, not an accident. Candidates 0.01 and 0.015 declined. — Andrei, 2026-08-13
- **Relog schema.** Approved as §6 proposes — row per (query, route, rank) with `raw_score`, depths 50 and 200 kept separate, full provenance block — combined with per-strategy raw scores. Andrei recalls raw component scores may already persist somewhere; Phase 0 item 8 verifies before any retrieval is scheduled, since if they exist Phase 2 shrinks to gap-filling. — Andrei, 2026-08-13
- **Collections are a diversity instrument, not a transfer sample.** Specialized datasets with unique query skeletons, collected against `cells.yaml` — 16 different schools to find a chess grandmaster, not one school searched harder. The hide-one-lane rotation is a coverage/consistency readout with no transfer burden; C3's proof rests on the external corpus and the probe gate. Lane count is stale (~21+ now vs documented 16); Phase 0 item 3 reports the true count, and every "16" in this plan reads "all materialized lanes". — Andrei, 2026-08-13
- **Verdict ownership.** PLAN.md is changeable — some claims are already outdated. The verdict's consequences are applied to PLAN.md directly as part of delivering `VERDICT.md`; no separate retraction process. — Andrei, 2026-08-13
- **External corpus for §7.4: selection rule precommitted 2026-08-13 (Codex objection 7); the name lands in `VERDICT.md` before indexing.** A candidate must be (a) absent from the dataset registry and untouched by any phase of this project, (b) publicly downloadable with **native queries and human relevance judgments** — hand-written queries are excluded, since a team writing the exam it grades is not an external test, (c) indexable in one night (≤200K docs), (d) from a domain no lane covers. The orchestrator lists every candidate satisfying (a)–(d), sorts by judgment depth (deepest qrels first), takes the top one, and records the whole list plus the choice in `VERDICT.md` before indexing, behind a §3.4 judge gate. Qdrant's documentation corpus fails (b) and is demoted to supplementary battle-test evidence, not the C3 instrument.

---

## 9. Verdict template

`VERDICT.md`, written as the plan runs.

```
# Strategy Router: viability verdict

Verdict: GO / NO-GO / BLOCKED
Date, operator, commit SHA, dependency-lock note

## Precommitted thresholds (signed off by Andrei, 2026-08-12/13)
<the claim table, the three auto-fusion gates, the probe gate, and the
bootstrap estimator paragraph from §1, verbatim, before any result>

## Delta-min, and who set it
0.02, answerable-rows mean objective, Andrei, 2026-08-13

## Feasibility (Phase 0.5)
Estimator: <written out>
CI half-width within-corpus: <n>   power at Delta-min: <n>
CI half-width 16-lane pooled: <n>  power at Delta-min: <n>
Which claims this design can resolve, and which it cannot:

## Label bias (Phase 0 item 6)
Hole rate per route: dense <n>%  sparse <n>%  rrf <n>%   gap <n> points
Literature range for reference: dense 14-32%, BM25 ~6%
How much of the measurement this lets us trust:

## Claims and gates
C1  record is real:           PASS / FAIL / INCONCLUSIVE — <delta, clustered CI, leakage delta>
C1b beats auto-fusion (G1):   PASS / FAIL / INCONCLUSIVE — <delta, clustered CI>
Gate 2 golden set:            PASS / FAIL — <n/30 correct orderings; every failure listed>
Gate 3 disagreement set:      PASS / FAIL — <slice size, paired delta and CI, hand-ruled table>
Probe gate:                   PASS / FAIL — <n/7 archetype probes under ordering criterion>
C2  not available cheaper:    PASS / FAIL / INCONCLUSIVE — <train-selected fixed fusion vs router at depth 50, plus M1 cost vs that arm; scope per §7.1 rule>
C3  survives new corpus:      PASS / FAIL / INCONCLUSIVE — <external corpus chosen by the §8 rule, its delta, then the rotation>
C4  survives a reranker:      NOT APPLICABLE — no reranker in the deployment (§8)
C5  represents a deployment:  SETTLED — <the two-tier target; the workload-histogram limitation>

## M1 latency
p50 / p95 per route, router overhead included:
Fraction of end-to-end latency routing can save:

## Scope of a failure verdict
This tested the shipped logistic regression, not routing in general. What the
d63 encoder router (the built v2) would have to beat:

## What we measured
<wave, experiment, artifact read, result, wall-clock>

## Codex objections, including the ones we could not answer
## Judge STOP verdicts, and how each was satisfied
## What we did not measure and why
## If GO: the three things budget should buy first
## If NO-GO: what would have to change for this to be worth reopening
```

A no-go with these numbers attached is a better outcome than another labelling campaign. Write it that way.

---

## 10. Steering probe, run 2026-08-12

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

## 11. Cut, with reasons

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
