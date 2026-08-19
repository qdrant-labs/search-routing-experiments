# v3 composition — representation × diversity × utility (layered)

*The v3 target framework. Companion to `~/.claude/plans/breezy-sleeping-star.md`
(this doc is the DESIGN; the plan is the execution tracker). Read
`docs/cells_adversarial_review.md` and `docs/v3_generation_brief.md` first.*

Date: 2026-08-19. Status: **v2 — rewritten after two Fable adversarial reviews
(machinery/feasibility + objective-coherence).** The v1 draft proposed a
three-factor product target; both reviews independently rejected it (see
"Why the product-form was dropped"). This version LAYERS the three objectives
instead of multiplying them.

---

## Purpose

The query taxonomy is complete. The composition MACHINERY has been overhauled
(cells, operators, augmentation loop, per-lane caps, Qdrant-in-loop). But the
TARGETS driving it are still v2's: `recipe.py`'s flat `n_per_route=200` per cell,
which over-represents identifier archetypes ~14× (the cells-review "mass
inversion": 27 of 44 cells require an identifier/code/math/blob token, together
4.36% of the real pool). v3 must optimize three EXPLICIT objectives —
representation × diversity × utility — expressed so they don't collide.

## The reframe, corrected: machinery + targets, but not everything v3 needs exists

"Re-weight the 44 v2 cell quotas" IS targets-only — the loop reads any parquet
order sheet (`loop.py:124,126`). But every v3-SPECIFIC addition (corruption
strata, corpus-stat cells, v3 cells, per-lane rates, waste budget) is **missing
machinery, not just a stale target** (see "Machinery gaps" below). The honest
frame: *re-weighting the existing cells is cheap; the new axes need code.*

## Prerequisites (block ALL target estimation)

1. **Commit the working tree + regenerate labels.** `src/augmentation/*` is
   modified-uncommitted and `corruption.py`/`constructed.py` are UNTRACKED;
   `labels.parquet` is Aug-12, predating the inject depth-fix. So every number a
   target would be estimated from (decisive-yield, feasibility, waste shares,
   per-stratum supply) measures behavior the tree no longer has. Commit, relabel,
   THEN estimate. Nothing downstream is trustworthy until this is done.
2. **Run the cheap probes first** (see Utility) — they decide whether a
   query-side utility term should exist at all, before we derive any target.

---

## The three objectives are LAYERS, not factors

They act at different points in the pipeline; composing them into one product is
what the reviews rejected. The layers:

- **Diversity = hard floors** (constraints on WHAT must be covered).
- **Utility = the sole scalar objective** (the score selection/generation
  maximizes).
- **Representation = realism + inversion bounds at composition, traffic
  weighting at EVAL** (a bound at build time, a weight at measure time — never a
  selection-time mass prior).

### Layer 1 — Diversity = hard floors

Cover every retrieval-mechanics archetype; cells are the diversity instrument.
Expressed as **per-axis marginal floors** (per cell, per corpus-stat band, per
corruption degree, per lane), NOT crossed boxes — the selector satisfies the
marginals jointly. This is the plan's already-locked "additive strata, not a
grid" decision; the v1 target table violated it.

Machinery to build (these are NOT stale targets):
- **`CELLS_ALL` threading.** v3 cells are unreachable: `loop.py:41` +
  `operators.py:39` import `CELLS_BY_NAME`/`CELL_TO_BANKS` built from `cells.yaml`
  only; a merged registry touches loop, cellfill, operators (shared v2 files) —
  real plumbing or a v3-namespaced loop, not a one-liner.
- **`predicts` normalization** — `cells_v3.yaml` writes class names, labels use
  route names; `ArchetypeCell.predicts` unvalidated.
- **Membership gate** — 2 hand-written queries per cell the predicate MUST admit
  (catches the 5 inverted cells; the best verification available).
- **qcs→catalog join** — `catalog_v3.parquet` carries no PMI/IDF columns, so
  authoring corpus-stat cells now crashes `attach_strata` (`AxisBand.mask` →
  KeyError); join `query_corpus_stats.parquet` in first.
- Fidelity errata (26 orphaned banks, 7 `looks_like` route leaks) do NOT gate
  generation and the erratum/override layer they need has no mechanism in
  `cells.py` — defer.

### Layer 2 — Utility = the sole scalar objective

A row has utility if it discriminates routes with a trustworthy label. Key the
objective on the axes that carry the signal — **lane×route coverage + qrels-depth
/ decisive supply** (the à-branch: corpus and label-depth levers INSIDE the
objective) — NOT on the cell/stratum axis (cell explains ~0.5% of outcome; lane
4.9%, dataset 12–21%). The v1 draft put lane and qrels-depth in the CONSTRAINT
row and keyed utility on the 0.5% axis; that is backwards.

- **Fix the coverage bug first** (cells-review #4, still live at
  `select_v3_prototype.py:335-343`): key coverage on `(cell, class)` and make
  `_gain` lexicographic `(-new_lane_route, -new_cells)` so lane×route is
  exhausted before cells break ties. Today it sums 1:1, weighting the 0.15% axis
  equal to the 4.9% one.
- **Multi-leg is a ROLE framework, not a score — yet.** leg-1 (retrieval,
  `RouterObjective`, exists) is the shipped-stack ground truth *by definition*.
  leg-2 (stronger indexer) disagreement marks a label STACK-SPECIFIC (not wrong).
  leg-3 (LLM-judge) disagreement means qrels are shallow/wrong. These are three
  DIFFERENT disagreement types. `leg_confidence` may NOT enter any selection
  score until three things are written down: (a) referent — confidence in what,
  per leg; (b) direction — does the selector prefer agreement (clean) or
  disagreement (informative)?; (c) a disagreement→number rule across legs with
  different output types. Until then: leg-1 only.
- **Probes gate the whole layer.** The leg-3 spike (~$5–20) and the skeleton
  minimal-pair test decide whether a query-side utility term deserves to exist,
  OR whether utility is purely lane/qrels selection. Run BEFORE deriving targets.
- **Objective variants** (`WastedRecallObjective`, in
  `route_objective_variants.ipynb`) → first-class `objective.py` subclasses.

### Layer 3 — Representation = bounds at composition, weighting at eval

Two mechanisms, neither a selection-time mass prior:
- **Row realism** — every row is something a user could send (real-parent
  machinery + census-rate dirt from `corruption_census.parquet`).
- **Inversion BOUND** — no archetype over-represented ≥K× under any plausible
  source weighting. This kills the 14× without a fictitious prior. (The "traffic
  prior" is NOT traffic — the 440K pool's mix is our own acquisition caps: orcas
  100K, gooaq/webfaq/clerc 50K each. Tracking it is proxy-circular.)
- **Traffic weighting at EVAL**, not selection — the SPEC d49c/d50f precedent
  ("balance applied at EVAL as a weighting, not a selection target"). If a true
  prior is ever wanted, anchor it EXTERNALLY (Qdrant page-search production
  traffic), never to the pool's own caps.

**The named conflict the v1 draft hid:** a prevalence prior would send the bulk
of mass into the single-answer lanes (msmarco/gooaq/webfaq/scirgen/clerc = 70.7%
of the pool) whose labels utility classifies as waste — matching a prior and the
waste cap are near-jointly-infeasible without qrels deepening. Eval-weighting +
inversion-bounds dissolves this: representation stops competing with utility for
selection slots.

---

## Language — placed per objective, not a fourth objective

- **Code-switch operator** → a **diversity floor** (manufacturable, v3-now): a
  `CorruptOperator` sibling, deterministic apply, LANGID-verified.
  `is_code_switched` exists (`features.py:123`), but `lingua` is NOT in
  `poetry.lock` (verified: 0 occurrences — the langid tests SKIP until
  `poetry lock`), so the dep must be locked first. Feasible — but also inherits
  the corruption admit-path gap (below).
- **Multilingual share** → NOT a representation target until an external language
  anchor exists (else it's a hand number — violates computed-not-assumed). A
  supply wish for now.
- **Cross-lingual (translate query + corpus)** → creates a **new LANE**, so it
  belongs on the supply/utility track (lanes carry outcome), NOT under diversity.
  Heaviest; parallel to new-datasets.

---

## Machinery gaps (NOT "stale targets" — these need code)

| Gap | Evidence | Blocks |
|---|---|---|
| Corruption-floor admit path | `cellfill.py:215-217` skips non-cell floors; no writer emits corruption floors; `loop.run` raises off-sheet (`loop.py:464-469`) | dirty-query + code-switch targets |
| Per-lane corruption draw control | `CorruptOperator.eligible` lane-shuffles, no per-lane quota | census-rate enforcement |
| Waste cap is a comment | `waste_cap` used once (`select_v3_prototype.py:60`); no capped waste draw in `greedy_select` | waste budget |
| `CELLS_ALL` threading | shared v2 imports (loop/cellfill/operators) | v3 + corpus-stat + language cells |
| Corpus-band double-structure | `_attach_corpus_band` bands `avg_idf` `[0.2,0.4]` (degenerate 2.7% low), NOT the empirical `min_pmi<=-1.0`/`max_idf>=0.87` — two structures, one axis (CLAUDE.md design bug) | reconcile before authoring cells |
| qcs→catalog join | `catalog_v3` has no PMI/IDF cols → KeyError | corpus-stat cells |
| Selector coverage key | `select_v3_prototype.py:335-343` per-cell, 1:1 gain | utility keying |

## Corrected readiness tags (fixing v1's over-optimism)

- Dirty-query rates: ~~machinery-ready~~ → **target-pending + machinery gap**.
- Global split + waste budget: split ready; **waste budget = unbuilt**.
- Decisive-yield/lane×route coverage: exists but implements the WRONG key
  (coverage bug) → **machinery-partial**.
- Enriched-unit corpus band: **stale** (`avg_idf` hand-edges, not empirical).
- `utility_multiplier`: **removed** (its inputs — legs 2-3, fresh labels — don't
  exist; the layering makes it unnecessary).

---

## Sequence

0. **Prereq:** commit working tree + regenerate labels (post-inject-fix).
1. **Probes:** leg-3 spike + skeleton minimal-pair — go/no-go on a query-side
   utility term.
2. **Machinery gaps:** corruption admit path, waste cap, `CELLS_ALL` threading,
   qcs→catalog join, coverage-key fix, corpus-band reconcile.
3. **Diversity floors + cells revalidation:** membership gate, `predicts`
   normalization, corpus-stat + code-switch cells.
4. **Utility objective:** leg framework WITH defined `leg_confidence` (referent /
   direction / mechanism); variants out of the notebook; as the selection driver.
5. **Representation:** inversion bound (pick K) at composition + traffic
   weighting at eval.
6. **Per-axis marginal targets** → re-target the augmentation loop's order sheet.
7. **Language/cross-lingual** as supply lands (parallel track).

## Verification (rewritten to remove circularity)

- **Utility — gap-closure with a LANE-MATCHED control.** `router.py`
  headroom-captured exists, but objective-selection changes the lane mix, so an
  unmatched gain may be pure lane composition ("utility = avoid dead lanes").
  Match lane marginals, ablate the objective. NOT "decisive-yield up" (circular —
  the selector optimizes it). `leg_confidence` informativeness needs a
  PRE-REGISTERED disagreement map (e.g. "leg-3 disagreement concentrates in the
  5 single-answer lanes") or it's unfalsifiable.
- **Diversity — membership gate** (2 queries/cell admitted) + every per-axis
  marginal floor met.
- **Representation — census-rate match + inversion bound** (no archetype ≥K×
  under any plausible source weighting); traffic-weighted eval is production-honest.

## Open decisions (genuinely unresolved — do not pre-answer)

1. **K** for the inversion bound.
2. Whether utility warrants a query-side term at all — the **probes** decide.
3. External traffic anchor (page-search) availability, if a prior is ever wanted.
4. Skeleton (frame=cell, filler=corpus-stat) vs additive per-axis marginals — the
   skeleton probe decides; until then the stratum unit stays per-axis marginal.

## Why the product-form was dropped (provenance)

Two independent adversarial reviews (2026-08-19) converged: the machinery lens
proved `target = mass × diversity_floor × utility_multiplier` is unaddressable in
code (single-string floors, no compound-key order sheet, no corpus-band minter)
and reinstates the cut grid; the objective lens proved it's unidentifiable
(representation_mass and diversity are the same cell axis with opposite signs; a
floor is a constraint not a multiplicand; utility keyed to the 0.5% axis). Same
verdict from opposite directions → layer, don't multiply.
