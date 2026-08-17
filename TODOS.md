# TODOS

## Ceiling levers — one re-measurement arbitrates all three (2026-08-12, SPEC decision 64)

Design closed via grill-me. Context: this session's harness repairs
(val loss now uses the training objective — the old unweighted-BCE val
metric RISES as pos_weighted training converges, so every fold restored
epoch-0 weights; cost dropped from thresholds and serving) invalidate every
number in `arm_results_panel_v3.parquet`. `src/router_playground.ipynb` is
the new per-run instrument.

Before the re-run (ordering is load-bearing — the relabel changes labels):

- [ ] Raise beir-nfcorpus `min_relevance` in lanes.py (§1b audit: 95.3%
      grade-1 qrels) + relabel:
      `poetry run python src/scripts/label_routes.py --only beir-nfcorpus --force`
- [ ] Fix `GoldenRoutingBuilder` round-trip contradiction (~147 rows) —
      also blocks Option A regardless.
- [ ] Rebuild or delete the stale `beir-nfcorpus_oracle` cache.
- [ ] Implement the two new arms (d64c): `zipf_channel` (wordfreq dep;
      query-local rarity scalars as input block) and `feature_branch`
      (taxonomy features as third privileged-branch TARGETS — the
      serve-safe form of arm 2).

The re-run batch (readout arbitrates every lever):

- [ ] 10-lane panel sweep, all arms incl. the two new, fresh results path.
- [ ] Learning curve: 25/50/100% of fit rows, fixed val split
      (playground). Flat by 50→100% ⇒ composition lever closed this cycle;
      still sloping ⇒ buy decisive rows in thin archetypes (d50g), never
      fat-lane mass.
- [ ] TIE_WEIGHT sweep 0/0.25/1.0 (playground) — free decisive-rows proxy.

Decision rules on the readout:

- [ ] Input arms: adopt whichever beats design on differ_agreement,
      checked on sparse-win rows; both win ⇒ combined arm gets its own
      attribution run first (deferred question).
- [ ] Option A (LLM-judge tied tails, PPI spine): arms move ⇒ waits;
      all arms flat under honest training ⇒ labels are prime suspect,
      A jumps the queue.
- [ ] Serve-time constraint stays HARD (d64b): raw query string only —
      generalize features via branches, never extract at inference.

## Encoder router (2026-08-11, SPEC decision 63)

Design closed via grill-me + compressed sanity-check (KEEP; flips if
LightGBM matches the design arm on holdout routes_differ — then ship the
trees). Evidence reports in docs/research/. Prototype against dataset v2
now; the arm comparison of record waits for the v3 dataset build.

- [ ] `src/encoder_router/` package (own package, composition/ precedent):
      training-table builder (bge embeddings, char-3–5-gram SVD fit on
      train queries, predicate multi-hot targets, â target blocks), model
      (encoder → z; cell branch ĉ = 44 sigmoids, BCE + per-cell pos_weight;
      corpus branch â = z-scored MSE; route heads on concat(z, ĉ, â) —
      feed-forward wiring, d63b), hand training loop (λ annealed down,
      Du-et-al cosine gate), LOLO-CV eval harness reused by every arm.
- [ ] New offline artifacts for â (no LLM spend): per-lane corpus profile
      (corpus.parquet scan + ~5K-doc sampled taxonomy extraction — retires
      d48i's CorpusIndex), per-query gold-doc profile + query↔gold lexical
      overlap, fold-local route-outcome rates (training rows only).
- [ ] `src/encoder_experiments.ipynb`: arms 1–7 as user-run cells, per-cell
      AP monitoring for ĉ, offline linear probe (per-feature
      recoverability — diagnostic, zero gradient, d63f).
- [ ] Deps: torch (direct pin), lightgbm (arm 5), via poetry.
- [ ] GATE for reportable numbers (d63g): the v3 dataset build — taxonomy
      round-trip fix → d62 word-shape guard → re-extraction → predicate
      re-eval, bundled with more data and better augmentation. Every v2
      run is shakedown only and must be labelled as such.
- [ ] Deferred (mirrored in SPEC): PFD teacher-student if arm 1 ≤ arm 3;
      prototype/contrastive shaping if ĉ flatlines; multilingual serving;
      gold-doc aggregation + sampling-size defaults. (The rarity-channel
      trigger is superseded by d64c's direct `zipf_channel` arm.)

## Augmented rows reach evaluation via read-time supplement (2026-08-10, SPEC decision 61)

- [x] DONE 2026-08-10: `QuerySupplement` (retrieval/base.py), `_augmented_rows`
      shared filter + `generation` param wired into `RouteLabels.label()`
      (labels.py). Tests in tests/test_labels.py. Closes the "labels.py merge
      of augmentation qrels" item open since d40's grill.
- [x] `src/scripts/label_routes.py` — terminal sweep (index + label every
      lane, cheapest corpus first), ports notebooks/route_labels.ipynb §16-17.
- [x] `provenance` column on every labelled row (2026-08-10): threaded
      natively through `RetrievalDataset.provenance()` → `QueryContext` →
      `FusionRow`, not bolted on in `label()` — so `BaselineDataset`/
      `HybridRoutingDataset` get it too, for free.
- [x] FIXED same day: `_augmented_rows`'s `floor_based` branch never checked
      `credit_gate`, so 240 gated rows (the exact ones behind the deferred
      audit above) leaked into `labels.parquet` unreviewed for
      beir-nfcorpus/crumb-legal-qa. Caught by eyeballing real provenance
      values post-label, not by test — tests/test_labels.py now has a
      regression case. Both lanes re-labelled clean; the other 18 lanes
      from the interrupted floor_based sweep were never touched by the bug
      (they hadn't run yet).
- [x] FOUND AND FIXED later same day: the SAME leak had already reached 8
      more lanes (bright-theoremqa-questions, crumb-tip-of-the-tongue,
      crumb-theorem-retrieval, bright-aops, crumb-paper-retrieval,
      rarb-math, crumb-set-operation-entity-retrieval, quest) — 66 gated
      rows, from a run against the code before the fix above landed. Purged
      by query_id (gated in the pool AND absent from `cell_selection.parquet`
      -> never legitimately admitted). Separately, the naive
      `provenance.fillna("natural")` backfill (done to close the "how do we
      know" gap) wrongly overwrote 580 real augmented rows that had NaN
      provenance for the same reason — corrected from the pool's own
      provenance column, the actual source of truth. Lesson: a blanket
      fillna over a column two independent pipelines write into is exactly
      how a real augmented row silently becomes indistinguishable from a
      natural one — the next such backfill needs to match on `query_id`
      against `pool.parquet`, never against nulls alone.
      Final clean state: 46,856 rows, 714 augmented (all decorate/floor-based
      — zero cell-based rows have been labelled yet, `--generation
      cell_based` hasn't been run to completion).
- [x] `label()` made incremental (2026-08-10): skips per query_id already on
      disk instead of per whole dataset — `--force` still means "redo
      everything," the default path now only scores query_ids missing from
      `labels.parquet` and appends, so adding a handful of new augmented
      rows to an already-labelled lane no longer re-ranks the whole lane.
      `RouteLabelSweep.run()`'s own dataset-level pre-skip removed (it would
      have bypassed this and required `--force` for no reason). Verified
      live: re-running an already-labelled lane is now a ~7s no-op instead
      of a full re-rank.

## Cell-based gated rows admitted WITHOUT the audit (2026-08-10, deliberate)

User decision: don't wait for the audit tool below — admit the 377 gated
cell-based rows now, validate later. `CellFill().admit()` was called against
an in-memory copy of the pool with `credit_gate` forced to `none` for rows
where `floor.isin(CELLS_BY_NAME)` — `pool.parquet` on disk was NOT touched,
so it still honestly records these as unaudited (`coherence_gate` /
`declaration_audit`). 308 of 377 were actually admitted (69 hit ordinary
per-cell shortfall/lane-share caps, same as any admission — nothing to do
with the bypass). `cell_selection.parquet`: 57,653 → 57,961 rows, across 15
lanes (beir-nfcorpus, crumb-code-retrieval, crumb-legal-qa, rarb-code,
crumb-clinical-trial, rarb-math, crumb-set-operation-entity-retrieval,
crumb-tip-of-the-tongue, bright-aops, trec-dl-2022, crumb-paper-retrieval,
antique, bright-theoremqa-questions, crumb-stack-exchange, scirgen-geo-en).

- [ ] These 308 rows carry real routing labels once `label_routes.py
      --generation cell_based` runs, but their underlying meaning-
      preservation/coherence claim has never been human-checked. If the
      audit tool below eventually rejects a declaration or a floor, its
      rows already in `labels.parquet` need to be pulled back out — this is
      the concrete cleanup debt the skip creates.
- [ ] Re-running this same override is needed for any FUTURE campaign batch
      that stages more cell-based gated rows — nothing here is a permanent
      pipeline change, it's a one-off decision to redo each time until the
      audit tool ships.

## Gated-row human audit (d42h/d40e) — deferred, tool not built (2026-08-10)

617 pool rows are staged and waiting: 113 across 3 DECLARATION_AUDIT entries
((length_words, up): `length_words:60+` + `pasted_code_fragment`, 60 rows;
(length_words, down): `bare_acronym` + `single_token_char_blob`, 21 rows;
OperatorSyntaxRewrite's one declaration: `logical:operator_syntax` +
`boolean_operator_query`, 32 rows — floors SHARE a declaration, so one pilot
review unlocks every floor built on it, present and future) and 500 across
Inject's 24 COHERENCE_GATE floors (per-row, no shared unlock — SPEC d40e's
per-row LLM meaning-gate needs its own human-audited validation sample
first).

Confirmed this session (not assumed) why the loop's own `verify()`/
`structural()` checks cannot substitute: every `structural()` implementation
checks for *unauthorized addition* only (new content tokens, new spans,
missing literal surfaces) — none of them check for readability or silently
dropped/damaged meaning. Decorate's own `structural()` docstring says so
explicitly: "no parent-relative machine check... the observed restructuring
cases are exactly what that audit rules on." This is the real, narrow,
still-needed gap — not leftover caution.

- [ ] Build the audit tool (medium: a notebook, matching every other human-
      review surface in this repo — route_labels.ipynb, augmentation_supply.ipynb,
      selection_audit.ipynb). Open questions for when this is picked back up:
      what exactly gets shown per row (parent query + child query + floor +
      operator, at minimum); how a verdict gets written back to flip
      `credit_gate`; for COHERENCE_GATE specifically, whether to review-and-
      unlock row by row (500 rows, no new machinery) or run a smaller
      diagnostic sample first to measure whether an LLM would agree with a
      human often enough to justify building the d40e auto-gate (unlocks
      nothing itself, a go/no-go measurement) — user has not chosen between
      these yet.
- [ ] Until this ships, the 377 cell-eligible gated rows (part of the 617)
      stay unreachable by `CellFill.admit()` — `_admissible()` only pulls
      ungated rows into `cell_selection.parquet`.

## Acceptability view over route labels (2026-08-07, SPEC decision 60)

Design closed via grill-me. Ties are judgment-resolution artifacts (0 of
15,037 have identical top-10s); the upgrade is a derived acceptability view
(`ok = score >= oracle − 0.3`, hit parity), three binary training heads,
all_zero stays null. Vocabulary in CONTEXT.md (Acceptability label, Serve
decision); full rationale in SPEC d60.

- [x] Implement the view class — DONE 2026-08-07: `AcceptabilityLabels`
      (labels.py) + `RouteLabels.acceptability()`; default tolerance derived
      as `RouterObjective().ndcg_weight`, never typed. Must-pass test
      (`tolerance=0` reproduces stored `route`) + per-shape cases in
      tests/test_labels.py.
- [x] Heads router + notebook comparison — DONE 2026-08-07:
      `AcceptabilityRouter` (router.py, 3 binaries on ok_*, serves cheapest
      clearing threshold; no-head-fires fallback = most probable, the SPEC
      deferred policy) + route_experiments.ipynb §6 (readout table vs serve
      oracle / argmax router / constants; smoke-verified end to end).
      Threshold tuning and the auto-fusion baseline row still to run/judge
      in the notebook — user-driven.
- [ ] **Option A (next)**: LLM-judge the differing tails of tied rows,
      PPI-rectifier spine (docs/research/route-label-sourcing.md) —
      ~14.4K queries × ~15–20 unjudged tail docs, sampled not exhaustive.
      Blocked by the two hygiene items below.
- [ ] Fix `GoldenRoutingBuilder` parquet round-trip (~147 rows where
      route_rankings contradict route_scores) — option A judges from those
      rankings, so the bug must die first.
- [ ] Rebuild or delete the stale `beir-nfcorpus_oracle` cache (323 rows
      vs 12 selected).
- [ ] **Option C (conditional)**: harden webfaq/gooaq/msmarco corpora with
      adversarial distractors — only if A finds the ties real.
- [ ] **Option D (last)**: augment tied parents into harder children —
      after A/C show which perturbations break ties.

## Chances scheduler for AugmentationCampaign (2026-08-07, SPEC decision 59)

Design closed via grill-me + sanity-check (KEEP, stdlib-only). DONE
2026-08-07 — `FaultStreak` (loop.py) + `_FloorTurn` (campaign.py), both
plain `@dataclass`es (not pydantic — mutable in-loop accumulators, same
shape as `Spend`, never cross a real boundary). 170/170 tests passing, ruff
clean.

- [x] `AugmentationCampaign._schedule()` runs a `collections.deque`-based
      floor queue with per-floor `_FloorTurn` state (`chances_left`,
      `tried: set[str]`, `remaining`, `accepted`); `spend_chance()` owns the
      chances_left decrement (not external mutation).
- [x] `AugmentationLoop.run()` gained `exclude: frozenset[str]` (unioned
      with `parents_used`) and `max_consecutive_faults: int | None` —
      defaults preserve every existing caller's behavior exactly.
- [x] Fault = `not (outcome.accepted and not problems)` — one boolean
      (`banked`), computed once, feeds both the accept/drop branch and
      `FaultStreak.record()`. No distinction by reason.
- [x] `FaultStreak.tripped()` marks `stopped` itself (not set externally);
      3 consecutive -> `stopped_early` in `.attrs` -> campaign spends a
      chance and requeues at the literal back; 0 chances left -> dropped for
      the rest of the run.
- [x] **The must-pass test**: `test_a_floor_that_always_faults_costs_at_most_the_k_ceiling`
      (tests/test_campaign.py) — asserts `engine.calls == MAX_CHANCES * FAULT_STREAK`
      (9) exactly against 15 real available parents, never a 10th call.
- [x] Summary distinguishes `DROPPED_EXHAUSTED_CHANCES` from `PRODUCE`, plus
      an explicit `print()` line listing dropped floors when any exist.
- [ ] Not directly tested: the chances scheduler against a *cell* floor
      (multi-call `produce()`, e.g. inject+stat_rewrite bundled) — both new
      tests use a bare single-call floor for simplicity. The fault-counting
      logic doesn't distinguish floor type, so this should generalize, but
      isn't exercised the way `test_plan_resolves_a_cell_floor_through_demand_not_operator_for`
      covers `plan()`'s cell path.

## `AugmentationCampaign.plan()` never understood cells (2026-08-06)

Caught live: `campaign.plan()` reported "skip: no operator" for all 22
hungry floors. It predates the d51 cell rebuild and still called
`loop.operator_for(floor)` directly — which only ever recognized bare floor
labels ("id:tech", "length_words:60+"), never a cell name. `loop.run()`
never had this bug because `demand()` checks `CELLS_BY_NAME` first.

- [x] `campaign.plan()` now routes through `loop.demand()` +
      `loop.planned()`/`loop.owner()` — the same cell-aware path `run()`
      already used — DONE 2026-08-06. `_planned`/`_owner` on
      `AugmentationLoop` made public (`planned`/`owner`) since they're now
      called across the module boundary from `campaign.py`, not just
      internally. Added `SKIP_UNSERVABLE` distinct from `SKIP_NO_OPERATOR`
      (a cell with no servable parent left vs. a bare floor nothing
      registers for are different failure reasons).
- [x] Verified against the real order sheet: all 22 cells now resolve their
      real `operator`/`gate` (e.g. `inject, stat_rewrite` /
      `coherence_gate`) and correctly cap at `pilot_n=30` — previously all
      22 showed `None`/`None`/`skip: no operator`.
- [x] Regression tests in `tests/test_campaign.py` (new file): a cell floor
      resolves correctly, a gate-free bare floor still works unchanged, an
      unregistered bare floor still reports `SKIP_NO_OPERATOR` distinctly
      from an unservable cell. 168/168 passing, ruff clean.
- [ ] Not yet checked: whether `AugmentationCampaign.run()`'s downstream
      consumers (if any exist outside this repo, e.g. a notebook) read the
      old single-operator-name `operator` column expecting exactly one
      name — it's now a comma-joined string when a cell plans more than one
      operator (e.g. `"inject, stat_rewrite"`).

## From the Spend-tracker-surfaced acronym collision (2026-08-06, SPEC decision 58)

The new `Spend` cost tracker turned an abstract worry ("is this expensive?")
into a concrete, investigable number: a live `legal_citation_canonical` run
dropped 9/14 attempts, all for `"child gained spans: ['acronym']"` — a
deterministic collision between two correct, independent banks
(`LegalCitationBank`'s `"U.S.C."` form and `AcronymBank`'s dotted shape), not
model variance.

- [x] `StatRewrite.structural()` now authorises a gained span by CHARACTER
      RANGE against `parent["surfaces"]`, not just by feature name — DONE
      2026-08-06 (d58). `_spans_by_name()` + `_explained_by_surfaces()`
      (operators.py). Audited all four operators; only StatRewrite's check
      crosses feature-group boundaries, so only it needed the fix.
- [x] Regression tests: an authorised U.S.C. citation must not be flagged;
      a genuinely smuggled acronym OUTSIDE the authorised surface still must
      be — DONE, both in `tests/test_operators.py`. 165/165 passing, ruff
      clean.
- [x] Re-run live against `legal_citation_canonical` — DONE 2026-08-06:
      5/5 accepted, 0/5 dropped (was 5/14, 9 dropped, all `['acronym']`).
      Spend before -> after: 77 -> 26 hops, 437,714 -> 145,293 tokens,
      161.0s -> 53.6s wall — roughly a 3x reduction, entirely from removing
      guaranteed-fail attempts (attempted now equals accepted).

## From "why StatRewrite?" (2026-08-06, SPEC decision 57)

Asked plainly why StatRewrite runs at all — whether a parent's own corpus
mint might already be enough — and it traced to a real bug: `eligible()`'s
`movable` filter excluded parents already sitting inside a two-sided band, so
a genuinely free parent was invisible to the pool rather than merely unedited.

- [x] `already_holds`, `needs_a_move` (dispatch.py) + `StatRewrite.eligible()`'s
      broadened `movable | already` mask + parent-aware
      `calls_for(plan, cell, parent)` — DONE 2026-08-06 (d57). Measured on
      `version_pinned_technical`: candidate pool 4,183 -> 4,802 parents; 689
      of them need ONLY Inject once picked, StatRewrite dropped from the call.
- [x] BUG `StatRewrite.eligible()`'s `.assign(stat_value=joined["__value"])`
      crashed whenever `eligible()` matched zero rows — assigning a column
      from the UNFILTERED frame onto an empty filtered frame reintroduces a
      row via index alignment (pandas has no shape left to constrain against
      on a fully-empty frame). FIXED 2026-08-06: align to `out["__value"]`
      instead. Caught only by a synthetic test with a non-matching catalog
      row — a case real traffic never hits, since a hungry cell always has
      some eligible row.
- [x] Test coverage: `tests/test_dispatch.py` (`calls_for` drops/keeps/never-
      drops-CONSTRAIN/all-free, 4 tests), `tests/test_operators.py`
      (`eligible()` includes an already-in-band row and sorts it first, 2
      tests), `tests/test_augmentation_loop.py` (one engine call, not two,
      when Inject alone already satisfies the length band, 1 test). 151/151
      passing, ruff clean.
- [ ] `already_holds`/`needs_a_move` only has one real axis to test against
      today (`length_words`, the only `WORD_AXES` member). Revisit once a
      second bounded-above-with-a-floor axis declares a direction — nothing
      today exercises the multi-axis case.
- [ ] `AugmentationLoop.grounded()` carries an EARLIER, still-uncommitted fix
      from this same investigation: gated on
      `parent.get("grounding_doc_id") is None` rather than `Stage.CONSTRAIN`
      classification, because `StatRewrite.instruction` decides to cut from
      the PARENT's actual value, not the plan's stage — a two-sided band the
      parent already overshoots also cuts, and did so blind (no document)
      before this fix. Applied, but no test exercises the gating itself yet
      (`test_a_cut_reads_the_gold_document_when_one_is_known` in
      `test_cells.py` only covers `_cut_instruction`'s wording, not whether
      `grounded()` decides to fetch).

## From the empty-string investigation (2026-08-06, follows d55)

The user reported "tried: ''" on two live parents and asked whether the
harness was sending the model something unwinnable. Verified with zero LLM
spend: neither reported parent (`fayetteville`, `by-product meal`) was
arithmetically incompatible — 1 mandatory word each against any realistic
ceiling. So those two empty outcomes are the model failing a FEASIBLE task,
not the harness's fault; but the engine had no way to say that distinctly,
and no way to catch the cases that genuinely are the harness's fault.

- [x] `AugmentationOutcome.text` is now `str | None`; `error: ErrorCase | None`
      replaces stuffing a protocol message into a fake `TargetCheck`;
      `attempted_tools: tuple[str, ...]` traces every tool called across a
      round-exhausted attempt, in order — repeating the same tool distinguishes
      a stuck loop from one doing varied legitimate work that simply ran out of
      budget. `ErrorCase`: `NO_TEXT`, `EMPTY_SUBMIT`, `ROUNDS_EXHAUSTED` (engine-
      raised, mid-conversation), `INCOMPATIBLE_PARENT` (loop-raised, pre-flight).
- [x] `dispatch.unreachable(call, parent)` + `mandatory_words(parent)`: before
      spending a call on a CONSTRAIN step, sum the words in every mint's
      mandatory literal surface and compare against the ceiling. If already
      broken, `loop.produce` returns `INCOMPATIBLE_PARENT` at `attempts=0` —
      zero spend — instead of letting the model discover it costs 6 rounds.
      Proven both ways: a poison engine that raises if ever called confirms an
      incompatible `bare_acronym` parent never reaches it.
      Scoped to `length_words` (the only axis with a declared DOWN direction,
      d53f) — extend `dispatch.WORD_AXES` (now public, shared with d57's
      `already_holds`) if another axis ever gets one.
- [ ] `_pair_line`/the drop-report branch now print `error` + `attempted_tools`
      instead of a fake failed-check line — never re-run against a live batch
      log to confirm the readout is actually clearer, only checked against
      hand-built outcomes.
- [ ] The trace only ever showed `verify`/`verify`/... in every example checked
      so far. If a real round-exhausted batch shows genuinely VARIED tool use
      (list_features, generate_surface, verify in sequence) rather than one
      tool repeated, that is evidence for raising `max_rounds` — the opposite
      conclusion from a repeated-tool trace. Decide only from the trace, per
      d53's engine agent's own verdict: raising the budget before measuring
      would mask whether other fixes worked.

## From the request-shape grill (2026-08-05, SPEC decisions 55 + 56)

Fix the three bugs FIRST — the leaked-requirement one changes what the model is
even asked, so measuring the new prompt before it lands measures nothing.

- [x] BUG `dispatch.planned_targets` never excluded an unsatisfied requirement
      (`id()` compared across two `requirements(cell)` calls, always False) —
      FIXED 2026-08-05: keyed on the requirement's VALUE, frozen bands compare
      by value. **d52(f) partial fulfilment now works for the first time.**
- [x] BUG `ParentPool.gold_text` took a NaN `grounding_doc_id` over the qrel
      fallback (NaN is truthy) — FIXED 2026-08-05 with a `pd.isna` guard.
- [x] BUG `StatRewrite._cell_band` resolved via the global `CELLS_BY_NAME` —
      FIXED 2026-08-05 by d55's own ABC change: `instruction`/`targets` receive
      the requirement the call serves, so a generated cell resolves without any
      registry, and a caller passing nothing gets a clear `ValueError` instead
      of unpacking None.
- [x] Sequential request pipeline (d55b/c) — DONE 2026-08-05: `dispatch.Call` +
      `calls_for`, `AugmentationLoop.produce` / `brief` / `_one_call`. Verified
      on `symbol_pile_no_grammar`: 2 calls, targets accumulating 2 -> 3
      requirements, the cut seeing the minted text.
- [ ] Generator-facing cell field (d55d): schema + wiring DONE
      (`ArchetypeCell.looks_like`, emitted LAST in the brief and worded to
      override an operator's default shape claim). **1 of 44 cells written** —
      `symbol_pile_no_grammar`. The other 43 are human-authored content: a cell
      with no `looks_like` silently keeps the operator's "must read as one
      coherent request a real person would type", which is right for most cells
      and wrong for every telegraphic one. Nothing can verify prose, so a wrong
      line shows up as oddly-shaped rows, not as a test failure.
- [x] `NUM` dropped from `CLOSED_CLASS` (d56) — DONE 2026-08-05 in the nested
      taxonomy repo, uncommitted. 402 bank tests pass; all 7 probe symbol piles
      now score 0.000 and prose still scores 0.5-0.6.
- [ ] RE-EXTRACTION, now owed by two repairs: `version_string` (d50b follow-up)
      and `NUM`/`nl_share` (d56e). `catalog.parquet` is stale for both, plus
      `number` (freed decimals fall through to NumberBank). Cell membership
      shifts wherever `nl_share` is banded, so the fill and every per-cell count
      move with it. Surfaces need `force=True` — `SupplyIndex.build` skips
      existing files and they were last built Jul 30.
- [ ] Was `symbol_pile_no_grammar` the only cell this band broke? d56 was
      diagnosed from one cell and a 7-example probe. Re-check every
      `nl_share`-banded cell after the re-extraction; the 2.0% band-crossing
      figure says prose cells barely move, but that was measured on natural
      queries, not on generated ones.

## From the harm-ordering fix (2026-08-05, SPEC decision 53)

Landed: `Stage.CONSTRAIN` + `reduces()` in dispatch.py, stage reorder in
`plan()`, `headroom()` deleted, `(length_words, DOWN)` declared,
`ParentPool.gold_text` + `_corpus_text` pushdown, `StatRewrite._cut_instruction`,
`AugmentationLoop.grounded`. 53 tests pass, ruff clean, nothing committed.
Supply 7,855 -> 208,465 parents; 22/22 cells servable; 13 cover their shortfall.

- [x] Inject weaves n surfaces (SPEC d54) — DONE 2026-08-05. `wanted(floor)`
      reads the band's count, `_offers` groups `(doc, bank)` holding that many
      DISTINCT surfaces, `surface` -> `surfaces` tuple. Verified:
      `symbol_pile_no_grammar` now offers `('cPGES', 'lipoxinA4')` against a
      `code_identifier >= 2` target where before it offered one and forbade a
      second. Cost: that cell 176 -> 93 parents (a doc must supply both).
- [x] Empty-query parents (2026-08-05): `ParentPool.available()` filtered on
      `checkable` but not on whether the query has text, so a blank parent
      reached the LLM (`before: ''`). Now dropped in `hydrate`, where the text
      arrives, with a count printed. 8-12 rows per cell. The upstream cache bug
      is still open under the composition-fill section.
- [ ] `InjectOperator.wanted()` takes the MAX count across a cell's bands, so a
      cell demanding 2 of one bank and 1 of another would over-ask on the
      second. No cell does today (only one has any count > 1); revisit if the
      d50(d) expansion writes one.
- [ ] UNEXPLAINED: three low-supply cells lost 4-8 parents against the
      pre-reorder baseline — `travel_transport_code` 34->30,
      `standards_compliance_lookup` 58->50, `business_temporal_reference` 9->5.
      All three have a LOWER-bound length band (QUERY_ONLY, not CONSTRAIN), so
      the reorder should not touch them, and tracing the steps in either order
      lands on the current number. Changes no decision (all three remain far
      short of their ~397 targets), but the baseline delta is unaccounted for
      and worth one look before trusting per-cell counts to the row.
- [x] `StatRewrite._cell_band` resolves a cell's band through the global
      `CELLS_BY_NAME` registry rather than being handed the cell, so a
      test-local or generated cell is invisible to it (hit while writing
      `test_a_cut_reads_the_gold_document_when_one_is_known`, which had to use a
      real cell name). DUPLICATE of the bug above marked fixed under d55 —
      the requirement now passes down through `instruction`/`targets`.
- [ ] `gold_text` is untested and is the riskiest new path: a wrong doc silently
      produces a cut aimed at the wrong evidence. Worth a test that the
      `grounding_doc_id` path and the qrel-fallback path resolve the same
      document for an Inject parent.
- [ ] The cut's escape hatch (d53e) is instruction-only: a model that cannot fit
      the band replies with the shortest answerable version, which then fails
      its target and drops. That is safe but wasteful — those rows are exactly
      the construction candidates, and nothing records them as such.

## Bank precision — repair on demand, not as a programme (2026-08-05)

No SPEC decision: three candidate systemic fixes were proposed and all three
refuted with data, so there is nothing to decide. A generic surface filter
rejects 0% of the `bic` / `env_var` / `error_code_like` / `ticket_like` junk
while killing 92% of real `datetime` and 100% of real `legal_citation` — the
discriminating property is per-format (a bare digit run is junk for
`version_string` and IS the format for `postal_code`). A stricter pattern as
the reference is circular: if it is correct, it IS the repair. Claim
concentration does not separate — sound banks span top-10 share 1.9-42.5%,
junk banks 1.7-87.0%, and `securities_id` (99.9% bare digit runs) scores best
of all 20 because those runs are genuinely high-entropy.

So: repair a bank when a cell you intend to fill depends on it. Cheap to check
first — `SupplyIndex.load(lane)` filtered to the bank, `.value_counts().head()`
tells you in seconds whether its surfaces are real.

Measured over the pooled surface index (18 of 42 lanes, 4,555,069 claims):
**52.1% of non-`number` claims are format-mismatched.** Ranked by damage
(junk rate x hungry-cell dependence):

- [x] `version_string` — DONE 2026-08-05, taxonomy repo, uncommitted.
      180,864 -> 5,098 surfaces (2.82%). Keyword-gated / v-prefixed / bare
      semver branches only; bare 2-part decimals gone. Gave up zero-padded
      parts, product-gated forms (`Python 3.11`), bare `v2`. 402 bank tests
      pass. Residual documented: state statute citations (`RCW 10.46.190`),
      dotted dates, dotted phones.
- [ ] `env_var` (RIGID, 14,363 claims, ~100% junk) — `--the` x525, `--and`
      x525 prose double-dashes plus LaTeX (`$T_1`). SOLE supplier for hungry
      `env_var_configuration` (270). Highest priority: the cell cannot be
      served at all until this lands.
- [ ] `securities_id` (113,415, 99.9% bare digit runs) — `998244353` x1572,
      an NTT prime from competitive-programming code; 112,409 of 113,415 from
      one lane. Dominates hungry `registry_structured_identifier` (249).
- [ ] `bic` (21,062, 77% eight bare uppercase letters) — `CONCLUSIONS`,
      `HOMEPAGE`, `CRITERIA`. Same cell as above; with `securities_id` that
      cell has no clean supplier (`iban` 4, `tax_id` 16, `phone_number` 663).
      NOTE: a real BIC is also 8 uppercase letters, so no pattern separates
      them — this one needs a lexicon, and `wordfreq` is not installed
      (spaCy's `en_core_web_sm` ships no lexeme probabilities: `the` and
      `qwzxjk` both return `is_oov=True, prob=-20.0`).
- [ ] `http_status_code` (19,590, 99.8% contain whitespace) — SPAN BOUNDARY
      bug, not a permissive branch: `100 Years`, `100\nThe`. Even a true
      positive yields an unusable surface. Hungry `status_code_idf_split`
      (235). Shares a root cause with `airport_airline_code` and `uri`
      (trailing punctuation) — worth one diagnosis for all three.
- [ ] `error_code_like` (16,023, 99.9% `E`+letters, no digits) — `ECOG` x2695,
      `EXISTS`, `ELSE`, `ELISA`. Same cell as `http_status_code`, so both its
      suppliers are junk. Lexicon case again (`EEXIST` vs `EXISTS`).
- [ ] `postal_code` (39,326, 99.7% bare 5-digit) + `alt_geocoding` (484, 66%
      arithmetic like `1622+161`) — both suppliers for hungry
      `geo_coordinate_postal` (278). Its third, `geo_coordinate`, has ZERO
      supply, so the cell has no sound source at all.
- [ ] `package_coordinate` (18,981, 58.7% `lower:lower` code colons — `x:x`
      x658, `coding:utf-8`) — sole supplier for hungry
      `package_coordinate_dependency` (198).
- [ ] `issn` (1,284, 97.7% bare NNNN-NNNN — year ranges `1861-1865`) +
      `library_classification` (273, 84.6% version-shaped `V1.1`) +
      `astronomical_designation` (1,662, 78.6% `M`+1-2 digits in clinical
      text) — all three feed hungry `bibliographic_catalog_identifier` (398);
      only `academic_identifier` there is sound (92.8% real DOI/ISBN).
- [ ] `genomic_accession` (3,605, 96.8% digits-then-letters — `6MWT`, `131I`,
      `3GPP`) — hungry `bio_clinical_identifier` (398); co-supplier
      `medical_code` is ~60% real.
- [ ] `hazmat_code` (569, 75.4% is cytochrome `P450` — 426 of 569 one surface)
      — hungry `standards_compliance_lookup` (394); `standards_citation`
      there is sound.
- [ ] `airport_airline_code` (148, 100% contain prose — `IATA airport`) +
      `license_plate` (125, `G-F 20`) — hungry `travel_transport_code` (397),
      which with `aircraft_vessel_reg_like` (69) has no usable supply.
- [ ] `uuid` — d50(b)'s repair is INCOMPLETE. 94.1% of claims are all-digit
      33-char binary strings (`111111101010101111100101001111111`); `0` and
      `1` are valid hex digits and the `(?!(.)\1*\b)` lookahead only rejects
      single-repeated-char runs. Verified still claiming under the repaired
      bank. No hungry cell depends on it today.
- [ ] `ticket_like` (13,751, 90.8% SHAPE-LEGAL) — a different failure: the
      population is `COVID-19` x2644, `ICD-10`, `IL-10`, `CIFAR-10`, the
      corpus's commonest tokens, in a cell named
      `rare_key_buried_in_chatter` (356). The detection is right and the CELL
      SEMANTICS are inverted; its only alternative `cve` has ZERO supply.
      Decide the cell, not the bank.

Not blocking, no hungry cell depends on them: `stock_ticker_like` (693,117
claims, 39.8% dictionary words / Roman numerals — the largest absolute
over-claim, but its cell `capsword_shape_ambiguity` is arguably ABOUT that
ambiguity), `ip_address` (64.3% C++ scope operator `::2`), `social_handle`
(40% code directives), `file_path` (39% date fragments `/1/2020`),
`betting_odds` (57.6% plain fractions), `lei`, `market_code`,
`clinical_trial_id` (49% are PMIDs).

Sound, safe to inject from: `datetime` (96.4% ISO — d50b's epoch deletion
verified working), `value_with_unit`, `currency_amount`, `code_identifier`
(759,815 genuine symbols, 4.3% short-camel leakage), `academic_identifier`,
`email`, `legal_citation` / `legislative_citation` (clean but supply-starved:
457+238 claims for a 400-missing cell), `standards_citation`,
`business_temporal`, `phone_number`, `hex_color`.

Zero supply in all 18 indexed lanes: `celex`, `chemical_id`, `cve`,
`geo_coordinate`, `iso_code`, `tracking_number`. Cell
`single_token_char_blob` (398 missing) maps to no bank at all.

- [ ] STALE ARTIFACTS from the version_string repair: `catalog.parquet` and
      every `src/data/<lane>/surfaces.parquet` are wrong for `version_string`
      AND for `number` (freed decimals fall through to `NumberBank`, which
      previously lost those ranges). Re-extraction is user-initiated. Note
      `SupplyIndex.build` is idempotent and skips existing files, so surfaces
      need `force=True` — they were last built Jul 30, six days before the
      d50(b) bank repair, so every number above measures PRE-d50b banks for
      `uuid` / `datetime`.
- [ ] `-Like` doctrine is 17% applied: 41 non-RIGID banks, 7 carry the
      suffix, and no test enforces it. NOT worth fixing from the current
      tiers — `code_identifier` is MODERATE and sound while `env_var` is
      RIGID and ~100% junk, so a rename driven by today's tiers would mark a
      good bank and miss the worst one. Also cascades into 33 `cells.yaml`
      bands across 23 of 44 cells. Revisit only if tiers are ever re-derived
      from measured precision.

## From augmentation-loop rebuild grill (2026-08-05, SPEC decision 51)

Supersedes d49(i) ("reused, not rebuilt"). Order matters: (1) and (2) are
offline and precede any LLM spend.

- [x] Cells → augmentation adapter, the half that was silently broken —
      DONE 2026-08-05 (uncommitted). `serves()` accepted cell names while
      `eligible()` looked supply up by `floor`, so every cell dispatch found
      zero surfaces and `continue`d. Added `ArchetypeCell.required_banks` +
      `CELL_TO_BANKS` (bands demanding presence only — a `below`-only band
      forbids its feature), `Inject.drawable()` filtering the per-bank surface
      index, `Operator.unsatisfied()` on the base, Decorate's cell dialect
      (`marker()` resolving a cell to one required decoration), and
      cells.yaml routing `conversational_courtesy_wrapper` to `decorate`.
      Verified: 7 inject cells resolve real supply; before, all were 0.
- [ ] `SupplyIndex.build_all()` over the 25 unindexed lanes (d51h). Every rung
      number in d51 is measured over 17 of 42 lanes, so the 31%/69% augment-vs-
      generate split is an artifact of an unbuilt index. Offline scan, no LLM
      spend, user-initiated. Re-measure the split afterwards.
- [x] Derived dispatch (d51c/d/e) — SUPERSEDED by SPEC d52 (2026-08-05). The
      single-operator rule could not say "this cell needs selecting, not
      augmenting"; replaced by the staged pipeline (`dispatch.plan`). Landed:
      `Stage`, `stage_of`, `plan`, `headroom`, `planned_targets`, `plan_report`.
- [x] Cell-predicate acceptance (d51b) — AMENDED by d52(f): targets are the
      requirements the plan could serve, not the whole predicate, so a partial
      row is kept and credited wherever it measures into.
- [ ] Taxonomy repo commit (d52g): `query-taxonomy.csv` gains the
      `Relevance Changing` column and `taxonomy.py` gains `RELEVANCE_CHANGING`.
      `src/query-taxonomy` is a NESTED repo — its own commit, and it must land
      before this repo compiles against it.
- [ ] `plan()` is untested end to end (only `stage_of` / `headroom` are). The
      path that most needs it: a plan whose CORPUS stage empties, which should
      mark that step unsatisfied and still produce the survivors partially.
      Nothing exercises `partial=True` today — every real cell comes back
      fully served or empty.
- [ ] Parent pool = catalog minus the whole selection (d51g): `checkable` +
      predicate-1 over 440,534 rows, less every query the selection touches
      (candidate + reused + control), text joined via `compose.join_text`.
      5,009 parents. Reimplement the `first_generation_only` guard — it reads
      `generated_from.isna()` off the selection frame, which a catalog pool
      does not have.
- [ ] Parent reservation in `CellFill` (d51g): exclude every (dataset,
      query_id) appearing in the pool's `generated_from` from future selection.
      Without it the no-near-duplicate guarantee holds only at draw time and
      decays on the next fill — silently, months later.
- [ ] `CellFill.admit(pool)` (d51j): admission test is
      `cell.select(mini_catalog)`; respect `_take`'s lane-share cap; lift
      `MiniFill._mini_catalog` to a shared helper. MiniFill stays for the d32
      slice artifact.
- [ ] `provenance` column on the cell selection (d51k): natural / augmented /
      doc_grounded / synthetic, with the natural-share ceiling testing
      `== 'natural'`. Blocks the synthetic rung — until it lands, a parentless
      row counts as natural and inflates the augmentation budget.
- [ ] Synthetic rung = d50(g) (d51l), reached by dispatch rather than deferred.
      Still blocked on distractor borrowing: one written doc per query with no
      distractors returns at rank 1 for every route, so the rows land
      `all_tied` and teach the router nothing.

Deferred by this grill: see SPEC "Deferred questions" — rung 2 (inversion),
`pilot_n` staging under cell demand, and whether `operator:` survives in
cells.yaml as a human override.

## From cell-quality grill (2026-08-05, SPEC decision 50)

This session, in order:

- [ ] Predicate repair (SPEC d50b), at the bank + a re-extraction:
      UUID bank keeps the 32–64 hex digest branch but rejects
      single-repeated-char / low-entropy runs (regex negative-lookahead if
      edify exposes backreferences; else bank post-filter + an `OVERRIDES`
      generator entry). DateTime bank drops the bare-epoch branch (ISO 8601
      only). Rebuild `catalog.parquet` (380K rows); generator↔bank round-trip
      test must stay green. IP-vs-version left irreducible; email left as-is.
- [x] Predicate repair DONE 2026-08-05 (SPEC d50b): UUID gained a
      `(?!(.)\1*\b)` negative-lookahead (edify exposes backrefs; no OVERRIDES
      needed — sampler skips `assert_not`); DateTime bare-epoch branch deleted.
      Submodule bank tests 366 pass. Catalog rebuilt via
      `feature_table.py --force` → 440,534 rows × 75 cols.
- [ ] FOLLOW-UP (makes d50b + d50c effective): re-run the fill against the
      rebuilt catalog — `CellFill(...).build(force=True)` — cell_selection.parquet
      and every selection_audit number are stale (built from the old 380K
      catalog). Then re-run `src/selection_audit.ipynb`.
- [ ] FOLLOW-UP: the round-trip gate `tests/test_taxonomy_generators.py` is RED
      independent of d50b (92 failed / 3 passed on clean baseline). Cause is the
      key-dialect skew already logged under the taxonomy-generators grill:
      `BANKS_BY_FEATURE` keys on bare `str(bank.name)`, `generator.feature`
      returns `"{group}:{name}"` → KeyError. The gate is meaningless until fixed;
      d50b's banks were verified by 50-surface self-heal sampling instead.
- [ ] Two cells.json guards (SPEC d50c), no re-extraction: `uri_in_query`
      + `length_words below 15`; `opaque_token_any_domain` drops
      `http_status_code` from its `any_of`.
- [x] Jaccard coverage audit script (SPEC d50e) — DONE 2026-08-05,
      `src/scripts/cell_divergence.py` (self-check + ruff clean). Globs
      `data/route_labels/*_oracle/rows.parquet`, joins cell_selection, computes
      per-cell dense/sparse top-10 Jaccard, sorts most-divergent first.
- [ ] BLOCKER for d50e to be meaningful: persist `route_rankings` during
      labeling. `RouteLabels.label()` (labels.py:136) builds GoldenRoutingBuilder
      rows that contain the per-route top-10 doc lists but writes only the score
      columns, discarding the rankings — so only `beir-nfcorpus_oracle` has them
      (11 of 27,145 cell rows). Fix: save the golden rows per dataset
      (`GoldenRoutingBuilder(...).save(out_dir/f"{key}_oracle")`, as the lone
      nfcorpus file was made), then re-run labeling. Until then the divergence
      screen scores ~0% of cells.
- [x] Improve the cell-generation brief (SPEC d50d) — DONE 2026-08-05.
      `docs/composition-cells-prompt.md` rewritten: cut `MEASURED FACTS` (d48h
      label-stat contamination) and `SUPPLY` (supply-as-validity error d50a);
      dropped `BUDGET`/`n_per_route`; restated the pooled-vs-within-corpus
      confound as a principle; added the cross-group INTERACTION + statistical-
      permutation emphasis; fixed the STALE output schema (was `lo/hi`,
      `hypothesis`, `n_per_route`, `min_lanes`, `supply` — now `at_least/below`,
      `any_of`, `predicts`, `source`, matching `ArchetypeCell`, which would have
      failed validation). Subsumes the d48h deferral.
- [ ] Run the revised brief through an LLM (separate task) → new static
      `cells.json`; validate it loads via `composition.cells._load`, then
      re-run `src/selection_audit.ipynb` against the new set.

Deferred:

- [ ] Cell-conditioned generation — the spine, own session (SPEC d50g).
      Given a cell's multi-band predicate, generate a coherent query hitting
      every band and, for zero-supply cells, the constructed document that
      answers it — from LLM knowledge, no parent. New capability; the current
      augmentation stack is entirely parent-based. Leans on the constructed-docs
      / synthetic lane (promote from edge-case to core).
- [ ] Taxonomy-extractor backlog (SPEC d50d): archetypes the proposal pass
      wants but no current feature measures (entity specificity, compound-noun-
      with-common-parts, …). New banks; only after the conjunction space of
      existing features is exhausted.
- [ ] Post-label divergence pruner (SPEC d50e): after labels exist, keep the
      cells whose measured route-split diverges, merge/drop the rest. `predicts`
      is never an allocator.

## From router-improvement work (SPEC decision 47)

Current state (2026-08-04): setup pass shipped (F2 + F1'); router beats
constant on both protocols (0.751 / 0.611). d47(b) inference-time corpus
stats retired — deployment target unknown at ship time. See PLAN.md for
the current-state summary.

Done:

- [x] Pass 1 — setup pass. F2 + F1' landed, F1 tried and reverted.
      Numbers and the F1 negative result: SPEC d47(a); current-state
      summary: PLAN.md.

Retired:

- [x] d47(b) corpus stats as router inference features, and passes 2-5
      under it — retired 2026-08-04. Rationale: SPEC d47(b).

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
- [x] LightGBM v2 (SPEC d45c) — SUBSUMED by SPEC d63 arm 5 (2026-08-11):
      same learner, richer inputs (embedding ⊕ SVD ⊕ features), same
      ceiling question, shared eval harness.
- [x] LUPI prototype (PLAN.md option B) — SUPERSEDED by SPEC d63
      (2026-08-11): grew into the encoder-router design after the
      three-report research pass. Key deltas from option B: feed-forward
      (hallucination) wiring instead of dropped heads, cell-predicate +
      gold-doc + outcome targets instead of feature reconstruction,
      frozen bge ⊕ char-ngram-SVD input instead of query features.
      Option C (teacher-student) is a d63 deferred question.
- [x] Composition redesign (SPEC d45h5) — LANDED as SPEC d48,
      2026-08-04. 32 archetype cells in `src/composition/cells.json`,
      generation brief
      `docs/composition-cells-prompt.md`. Measured that a recipe change
      alone was worthless (d32 already took 100% of feature-bearing rows
      in labelable lanes), so cells + allocator + train/control split
      replace the slice fill. Remaining work below.
- [x] BRIGHT splits registered (SPEC d49e) — 2026-08-04. All 12 now in
      `DatasetName` / `DATASETS` / `LANES`; `DATASETS` switched to
      `*(BrightSplit(s) for s in sorted(BRIGHT_SPLITS))`, matching the
      existing CRUMB idiom, so a future split needs only an enum member.
      `lanes.py` lists them explicitly — that package deliberately does not
      import `dataset_registry`. 30 datasets / 29 lanes; `orcas` is the one
      registered dataset without a lane, pending its class.
- [ ] RAR-b commonsense pools (SPEC d49e) — BLOCKED on repo slugs.
      `RARB_POOLS` holds only `math`/`code`; the catalog names the pools
      (αNLI, HellaSwag, TempReason — TempReason feeds
      `temporal_expression_natural`) but not their HF repo paths, and those
      must not be guessed. Needs one look at the RAR-b org listing, which
      is a network call and therefore user-initiated.
      *DRY defect to fix in the same pass:* the pool map is duplicated —
      `RARB_POOLS` (`hf.py:217`) and `RarbLane._POOLS`
      (`retrieval.py:562`). Adding a pool currently means editing both.
- [ ] Wave 2 boxes (SPEC d49e): FreshStack, ANTIQUE, LoTTE, WebFAQ (en
      config), ScIRGen-Geo (en slice), CLERC (verify availability first),
      GooAQ (doc_grounded — 62.7% passage-answer coverage measured; needs a
      real language filter, the file head is Portuguese and ASCII-only),
      nq_open (verify it ships a corpus; card says QC, text says "short
      answers only"). Queries first; corpora + embeddings per d49(d).
      Cache fills are user-initiated.
- [ ] ORCAS lane (SPEC d49f). `RetrievalDataset` subclass +
      msmarco-document corpus + `QrelSource.CLICK` labelling. Unblocks the
      head-of-traffic cells' single-corpus concentration and four
      otherwise-empty cells. Retire the `deferred` label lane while here.
- [x] Cells → augmentation adapter (SPEC d49i) — SUPERSEDED by SPEC d51
      (2026-08-05). The sheet side shipped (`floor = cell.name`) and the
      dispatch side is done, but "extend `operator_for`" turned out to be the
      wrong shape: dispatch is derived per (cell, parent), not mapped per cell.
      Remaining work is the d51 block at the top of this file. The claim that
      `pilot_n` needs no new tiering is now an open question, not a finding.
- [ ] d40e coherence pilot, IN PARALLEL with box loading (SPEC d49i).
      Gates Inject credit, and Inject is the only path for nine cells.
      Needs no new data (existing corpora + gold docs), so it does not
      compete with the downloads.
- [ ] d47 CorpusIndex + 16-row side test BEFORE any fill (SPEC d48i).
      `max_lane_share` caps by row count; whether contributing lanes are
      actually different needs corpus stats. ≥12/16 ⇒ lane diversity
      becomes corpus-stat band spread; ≤7/16 ⇒ keep the share form on
      names and drop the path. Governs which 50,000 rows get labelled,
      so it precedes the spend. Skip d47 pass 5 (6-config ablation) for
      now — model question, not dataset question.
- [ ] `ArchetypeCell` model + `composition/cells.py` from cells.json
      (SPEC d48c/d). Needs `any_of` (OR bands) and `max_lane_share`;
      `min_lanes` is superseded and must not be reintroduced; `max_lane_share`
      is computed at fill time, not stored. Repo
      convention is a Python declaration tuple like `LANES` / `BANKS`,
      so cells.json converts to `CELLS: tuple[ArchetypeCell, ...]`.
- [x] Per-dataset allocator (SPEC d48f) — DROPPED by d49(c). The global
      ≤20% cap is gone; `max_lane_share` inside each cell is the whole
      constraint. Global shares are reported, never targeted. Removes the
      cell-vs-ledger conflict that would have forced the deferred ILP
      escalation.
- [ ] Box-model fill (SPEC d49d). Assign cell candidates from query
      features across every box (no corpus needed); reuse existing labels
      on overlap; materialise and embed ONLY the corpora that won quota
      slots; label; re-check route quotas; top up. Shortfalls go to the
      augmentation adapter. Cells short of natural supply stage a pilot
      sample first, scaling only on confirmation. Writes a NEW artifact —
      do not overwrite `data/composition/selection.parquet` (d49k).
- [ ] Dark forest as leftovers (SPEC d49j): random draws from queries that
      entered no cell, ~20% of cell rows. Label it; it is the control
      slice and is never trained on.
- [ ] Per-archetype eval on held-out CELL rows + aggregate eval on the
      control group (SPEC d48f). Three protocols with lane-holdout
      (d45e); rare archetypes are by definition rare in the unbiased
      control, so they cannot be evaluated there. Closes WEAKNESSES #12
      and #14.

Deferred:

- [ ] F3 (relax decisive-only training) — F1' subsumed the tuner half of
      this concern. The training-set half (loosen the decisive filter to
      routes_differ) still SPEC-touching; the F3 experimental flag on
      `StrategyRouter.fit` stays in the code for future testing but the
      default keeps decisive-only per d45(a).

Deferred by the d49 grill (2026-08-04):

- [ ] Dark-forest size (d49j). The ~20% is provisional. The defensible
      number needs the per-row disagreement rate between the router and the
      best constant, which is only measurable once these rows are labelled.
      Resize next pass.
- [ ] CLERC availability (d49e) — `jhu-clsp/CLERC` may require terms
      acceptance. Verify before counting it as a box.
- [ ] `predicts` contamination (d48h). The brief that generated the cell
      set fed the runs our own label statistics, so some rationales argue
      from facts about our 16 lanes rather than from retrieval mechanics. A
      clean re-run means cutting the measured-facts sections from
      `docs/composition-cells-prompt.md`. Decide whether that matters
      before the priors are tested by labelling.
- [ ] Redefine label lane by qrel-hole rate rather than grounding — the
      original d37k idea, still open now that `deferred` is retired and
      lanes are named by qrel source.

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
      coherence pilot gates Inject credit.
      "labels.py merge of augmentation qrels" design closed 2026-08-10 —
      see d61 and the new section below; ready for code-implementer.

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

## Opaque-token repairs: word-shape guard + identifier-class band (2026-08-10, SPEC decision 62)

Design closed via grill-me. Two independent defects put a hex digest on the
dense route: spaCy tags unseen tokens into closed classes (`nl_share` 1.0 on a
32-char digest), and `bare_concept_token` bands only `number` out of 54
identifier columns. Both prior repairs of this class (d56, d50c) were recorded
against named artifacts and did not survive. Acceptance is the probe table, not
an aggregate.

Cell side — no re-extraction, lands first:

- [x] Derived identifier-span aggregate column, summed from the
      `structured_identifiers.*` columns, applied at all three catalog-shaped
      producers (`TargetComposition.build`, `CellFill._catalog`,
      `mini_catalog`). Derived, never stored (d62h).
- [x] `bare_concept_token`, `keyword_telegram_short`,
      `short_grammatical_question` band identifier absence as a class; the
      now-subsumed `number` band goes. `bare_concept_token` also gains a
      `sentence_markers.acronym` guard (`HTTP 502`).
- [x] `test_a_short_concept_cell_admits_no_identifier` — a cell with a
      `length_words` ceiling ≤ 10 that demands no identifier presence must band
      the aggregate. Trigger reads length + identifier demand, never `predicts`.
- [x] `test_parser_scalars_require_a_natural_language_floor` — any cell banding
      `nesting_depth` / `statement_count` / `widest_list_size` must band an
      `nl_share` floor. Passes on all four such cells today; the point is that
      the next regeneration cannot drop it.
- [x] `test_band_columns_exist_in_the_catalog` reads through the derivation, not
      the raw parquet, or the new column fails it.
- [ ] Re-run `CellFill().build(force=True)` — writes `cell_selection.parquet`,
      NOT `selection.parquet` (that is `TargetComposition`'s d32 artifact), and
      `build()` returns the cached file unless forced. Labels are reused on
      overlap (d49g), so only backfill rows cost retrieval.
- [x] Carried `cell` staleness on stored labels — NOTHING TO DO, verified
      2026-08-10. 217 of 46,856 rows (0.46%) carry a `cell` the predicate no
      longer claims, but `scripts/cell_divergence.py` (the d50e readout) reads
      `cell` off `cell_selection.parquet` and merges on `(dataset, query_id)`;
      nothing reads `labels["cell"]`. Rebuilding the selection is the fix. Do
      not add a refresh method for a column no analysis path consults.
- [x] Incremental corpus indexing (d62k) — DONE 2026-08-10.
      `BaseIndexer.missing()` diffs point ids and uploads only the difference;
      `_index()` keeps the count as the cheap grew?-trigger. Was: one new
      document re-embedded the whole lane (119,976 docs for `crumb-code-
      retrieval`). Never fired while lanes are frozen snapshots; fires every
      round once d50(g) generation writes constructed documents.
      `tests/test_indexer.py` pins the second-pass-uploads-nothing case.
- [x] Corpus re-embedding for d62 — NONE NEEDED, verified 2026-08-10. `_index()` guards
      on `client.count(collection) < len(corpus)` and `ensure_collection()`
      no-ops when the collection exists; d62 changed no corpus. All 42 selected
      lanes are already labelled, so already indexed. Re-check after the rebuild
      with the selected-minus-labelled lane diff: non-empty means a lane needs
      full corpus indexing.

Bank side — BLOCKED on `src/query-taxonomy` going green (92 round-trip failures
at `6008f40`):

- [ ] Word-shape guard at the closed-class counting site (`metrics/pos.py`): a
      digit-bearing token is never a function word. Comment states the rule, not
      `AUX`.
- [ ] Rebuild the catalog for digit-bearing queries only — a provable superset
      of the 0.45% that can change.
- [ ] Re-check the probe table; `nl_share` on the digest must read 0.0.

Deferred by the grill — do not start (full text in SPEC "Deferred questions"):

- Enumerated function-word lexicon replacing the POS test (d62c). Reopens on
  evidence only: a pure-alpha out-of-vocabulary token observed landing in a
  closed class. 0 of 13 sampled.
- Stale `cell` values on rows labelled under the old predicate (d62f). Scores
  are unaffected; any per-cell readout over existing labels reads the old
  partition until a rebuild.
