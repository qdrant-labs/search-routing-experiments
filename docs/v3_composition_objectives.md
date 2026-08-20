# v3 composition: representation × diversity × utility (layered)

*The v3 target framework, and where each layer lives in code. Companion to
`~/.claude/plans/breezy-sleeping-star.md` (this doc is the DESIGN, the plan is
the execution tracker).*

Date: 2026-08-21. Status: **v3, rewritten after implementation.** The v2 draft
described a machinery-gap table that has since been built, and anchored its
central instruction on `src/scripts/select_v3_prototype.py`, a file deleted in
155178f when the prototype graduated into `src/composition/objectives.py`. Three
independent adversarial reviews each spent effort rediscovering that. Every
pointer below was checked against the tree on the date above.

---

## Purpose

Optimize three explicit objectives, expressed so they cannot collide.
Representation and diversity are the same cell axis with opposite signs, and a
floor is a constraint rather than a multiplicand, so the three are LAYERED and
never multiplied. That verdict came from two adversarial reviews reaching it
from opposite directions (machinery and objective-coherence).

- **Diversity = hard floors.** Constraints on WHAT must be covered.
- **Utility = the sole scalar objective.** What the draw maximizes.
- **Representation = bounds at composition, weighting at eval.** Never a
  selection-time mass prior.

## Where the layers live

| Layer | Class | File |
|---|---|---|
| Diversity floors | `DiversityFloors` | `src/composition/objectives.py` |
| Labelling demand | `SelectionOrder` | `src/composition/objectives.py` |
| Utility | `UtilityObjective` | `src/composition/objectives.py` |
| Representation | `InversionBound` | `src/composition/objectives.py` |
| Pool, classes, strata | `LabelledPool` | `src/composition/pool_v3.py` |
| Artifact owner | `V3Composition` | `src/composition/composer.py` |

`CellFill` (`src/composition/cellfill.py`) is the FROZEN v2 owner. It shares the
sheet schema and the augmentation loop with v3 and owns none of the v3
artifacts. `Recipe.n_per_route` is v2's flat quota and has exactly one consumer,
`CellFill`. v3 sizes off `target_total` and per-axis marginals.

---

## Layer 1: Diversity = hard floors

Per-axis MARGINAL floors, satisfied jointly. Not crossed boxes. One row credits
every stratum it touches, at selection and in the audit alike.

Four axes, four owners:

- **Cells.** `DiversityFloors.marginals` enumerates from the cell registry, so a
  cell holding two rows reports two rows and fails.
- **Corruption degree.** Bands declared as `STRATA` in `pool_v3.py`, beside the
  code that writes them.
- **Corpus bands** (`corpus_idf`, `corpus_oov`, `corpus_pmi`). Report-only:
  no operator mints a corpus statistic.
- **Lanes.** `V3Composition._per_dataset` alone. This is labelling demand, not a
  sheet floor.

Three rules the implementation enforces, each of which was violated at some
point and produced a green report over no measurement:

1. **Strata are declared, never observed.** `marginals` reindexes over the
   declared band tuple with `fill_value=0`. A band no row landed in reports 0
   and fails. Counting only the values that occurred is how three axes reported
   every floor met while measuring nothing.
2. **`unknown` is coverage, not a band.** It is a `dark` count per axis and is
   never floored. A pile of unmeasured rows must not pass as a covered stratum.
3. **A join that reaches no supply row raises.** `_attach_corpus_strata` fails
   loudly when `query_corpus_stats.parquet` covers zero v3-native rows. The
   graceful `unknown` fallback stays for partial misses, which are legitimately
   unknown. Note the guard is on NATIVE coverage, not on all rows: with 46K
   covered v2 rows in the pool, an all-rows check never fires while every
   selected row is still dark.

**One floor, one bar.** Lane floor credit, `class_debt` supply, and
`SelectionOrder.yields` all count tier-0 non-waste native rows: routes differ at
any margin, plus genuine hybrid ties, excluding only fake ties and all-zero.
Three bars for one concept is the two-structures design bug in CLAUDE.md. A gap
measured at one bar and divided by a yield measured at another over-orders
labels.

## Layer 2: Utility = the sole scalar objective

`UtilityObjective.select`. There is no single scalar expression: the objective is
a lexicographic order that falls out of control flow, in this sequence.

1. **Maximize artifact size under the lane cap.** `feasible_total` binary
   searches the largest total whose per-class targets fit lane-capped supply, so
   the binding class never silently under-fills while the others fill to
   uncapped targets.
2. **Per class, greedy coverage.** `gain = (ds not in lanes_drawn, len(diversity
   - covered))`, a tuple compared lexicographically, recomputed per pick. Lane
   strictly dominates: cells and the other strata only break ties. Coverage is
   keyed per (stratum, class), since each class draw carries its own `covered`
   set.
3. **Seeded lane-capped fill** once nothing uncovered remains.
4. **Dictated waste draw** at `waste_cap` of the tier-0 total, lane-capped,
   typed `route_class="waste"` and never certified.

Keyed on the axes that carry outcome. Lane explains ~4.9% and dataset 12-21%,
against ~0.5% for cell, so keying utility on the cell axis would be backwards.

**leg-1 labels only.** No `leg_confidence` term may enter any selection score
until three things are written down: (a) the referent, meaning confidence in
what, per leg; (b) the direction, meaning whether the selector prefers agreement
or disagreement; (c) a disagreement-to-number rule across legs with different
output types. leg-2 (stronger indexer) disagreement marks a label
stack-specific. leg-3 (LLM judge) disagreement means the qrels are shallow or
wrong. Those are different disagreement types and do not compose into one score
by default. The `--llm-coherence` path is a binary ADMISSION gate, not a score.

## Layer 3: Representation = bounds at composition, weighting at eval

Three mechanisms, none of them a selection-time mass prior.

- **Row realism.** Real-parent machinery plus census-rate dirt from
  `corruption_census.parquet`. `InversionBound.realism` checks the artifact's
  damaged share against the census's own pooled rate.
- **Inversion bound.** `InversionBound.report` computes each cell's selected
  share against the minimum share any candidate weighting implies, and flags
  which over-representation the stratum floor itself mandates. Report-only until
  K is frozen.
- **Traffic weighting at EVAL, not selection.** If a true prior is ever wanted,
  anchor it EXTERNALLY (Qdrant page-search production traffic). The 440K pool's
  mix is our own acquisition caps (orcas 100K, gooaq/webfaq/clerc 50K each);
  tracking it as a prior is proxy-circular.

Every pool-share quantity in the draw is a CAP, never a target. That is what
dissolves the conflict a prevalence prior would create: the single-answer lanes
are 70.7% of the pool and utility classifies their labels as waste, so matching
a prior and capping waste are near-jointly-infeasible without qrels deepening.

---

## The generation ring

Five edges. The list is what generation owes, and it closes only if every edge
exists.

```
order sheet -> generate -> admit -> label -> pool -> recompose
```

- **Sheet** is written by `DiversityFloors.order_sheet`: cell floor shortfalls
  net of both the selection and the unspent supply that already satisfies them,
  plus the corruption line. Nothing else.
- **Admit** (`V3Composition.admit`) re-measures generated text against the
  cell's own predicate over a mini catalog. A row minted FOR a floor still has
  to measurably serve it. It credits every hungry line a row serves, not only
  the line it was minted for, matching how selection credits every stratum.
- **Label** is `label_routes_v3.py --admitted`, writing
  `v3/augmented/labels.parquet`. Answer keys come from
  `augmentation/qrels.parquet`, inherited from each row's parent, through
  `RouteLabels.label(..., generation="cell_based", include_gated=True)`.
  Synthesize-operator rows are excluded here: they answer only against
  constructed docs in an isolated collection, and scoring them against the paid
  collection is contamination.
- **Pool** reads four label files. Re-scored v2 rows are `native=False` and
  measure yields and priors, never supply.

**Two owners, two numbers.** Generation owes the sheet. Labelling owes the class
shortfall its own order cannot buy (`SelectionOrder`'s `residual_debt`).
Conflating them is how a ~11.5K program once read as a 197,756-row one: the
class debt was being waterfilled onto cells by `cell.predicts`, which is the
prior under test and may never allocate. `cells.py` states that contract, and
allocating by it manufactures the correlation the cell exists to measure.

**Credit gates.** Two gates, two keys, both human-owned in different senses.
`coherence_gate` clears from `CoherenceJudge.passed()`. `declaration_audit`
clears from a hand-written newline file of query_ids. `None` for either keeps
its rows waiting. A gate with no key is a floor that can never be paid.

## Operational order

Any change to labels or corpus statistics has to walk the whole chain, because
each step is an input to the next.

```bash
# 1. corpus statistics for the v3-native rows (appends, resumable, no --force)
poetry run python src/scripts/collection_features.py --per-query --v3
# 2. label the admitted generated rows
poetry run python src/scripts/label_routes_v3.py --admitted
# 3. refresh the catalog so new rows get cells and a corruption degree
poetry run python src/scripts/build_v3_catalog.py --force
# 4. recompose
poetry run python src/scripts/compose_v3.py --force
```

Skipping step 1 raises at `LabelledPool.frame()`. Skipping step 3 lands the new
rows in the pool with zero cells and `corruption_degree == "unknown"`.

## Verification

- **Utility: gap closure against a LANE-MATCHED control.** `run_ablation.py`
  matches arm B on lane and route class and measures a paired objective on the
  eval reserve frozen BEFORE selection, with a lane-cluster bootstrap. NOT
  "decisive-yield up", which is circular because the selector optimizes it. Know
  the claim's scope: matching on route class means the ablation tests the
  within-(lane, class) coverage and tie-break choice, and never ablates the
  split, the lane cap, or the certified-first tiering. The reserve is
  certified-only at the same margin the selector uses, so utility is proven on
  rows leg-1 is already confident about.
- **Diversity: the membership gate** (`tests/test_cell_membership.py`, two
  positives and two negatives per cell, positives authored from `looks_like`
  alone, strict-xfail quarantine naming the inverted cells) plus every declared
  marginal met.
- **Representation: census-rate match and the inversion bound.** No archetype
  over-represented ≥K× under any plausible source weighting.

## Genuinely open

1. **K for the inversion bound.** The report prints a candidate. Nothing fails a
   build on breach yet, so Layer 3's bound does not bind.
2. **Floors as selector constraints.** `stratum_floor` still does not enter
   `select`. The greedy pass covers each stratum once and the rest is a
   lane-capped random fill, so floors are satisfied by accounting plus a
   generation order, not by the draw. Netting unspent supply out of the sheet
   makes `NeedsSelection` unreachable from a sheet the composer wrote, which
   closes the deadlock without making floors constraints. Two limits remain:
   `loop.demand`'s population is the CATALOG, so a cell whose only unspent
   members are unlabelled still raises (that is genuine labelling demand), and
   `unspent` counts eval-reserve rows and their cluster-mates, which the
   selector can never take.
3. **Whether utility warrants a query-side term at all.** The leg-3 spike and
   the skeleton minimal-pair probe decide it.
4. **External traffic anchor** (page-search) availability, if a prior is ever
   wanted.
5. **Skeleton (frame=cell, filler=corpus-stat) vs additive per-axis marginals.**
   The skeleton probe decides. Until then the stratum unit stays per-axis
   marginal.
6. **`WastedRecallObjective` as a selection driver.** Promoted out of the
   notebook to `hybrid_search_rrf_dataset/objective.py` with a test, and
   unreachable: nothing constructs it, and swapping it means relabelling the
   pool, for which no path exists.

## Language

- **Code-switch operator** is a diversity floor, manufacturable as a
  `CorruptOperator` sibling with deterministic apply and LANGID verification.
  `is_code_switched` exists in `features.py`; `lingua` is not in `poetry.lock`,
  so the dep must be locked first.
- **Multilingual share** is not a representation target until an external
  language anchor exists. A hand number here violates computed-not-assumed.
- **Cross-lingual** (translate query and corpus) creates a new LANE, so it
  belongs on the supply and utility track, parallel to new datasets.
