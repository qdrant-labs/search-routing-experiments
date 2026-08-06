"""Staged cell dispatch: classify each of a cell's requirements by what it
costs to satisfy, then serve them LEAST-HARMFUL FIRST (d53). Most-constraining
first is search logic; generation must add before it takes away, so the
destructive step runs last and cuts knowing what the row has to keep.
"""

from __future__ import annotations

from collections.abc import Iterable
from enum import StrEnum
from typing import NamedTuple

import pandas as pd

from augmentation.core import Operator, SurfaceOrigin
from composition.cell_targets import band_target
from composition.cells import ArchetypeCell, AxisBand
from query_taxonomy.taxonomy import RELEVANCE_CHANGING
from taxonomy_generators.verify import SpanTarget, StatTarget, Targets

class Stage(StrEnum):
    """Where a requirement is served. Declaration order IS the pipeline order,
    ascending in harm: what cannot be changed, then additions, then the cut."""

    SELECT = "select"
    """Nothing mints it, so it must already hold in the parent."""
    CORPUS = "corpus"
    """Minted from a surface copied out of the parent's own gold document."""
    QUERY_ONLY = "query_only"
    """Woven from the phrase list; needs no corpus and never runs short."""
    CONSTRAIN = "constrain"
    """Reached only by REMOVING words, so it runs last, once every addition is
    in place and the cut can see what must survive (d53)."""
    REBUILD = "rebuild"
    """Adding it changes which documents are relevant, so the answer key would
    have to be rebuilt rather than transferred (d40g). Nothing serves it."""


def requirements(cell: ArchetypeCell) -> tuple[tuple[AxisBand, ...], ...]:
    """The cell as independent requirements: each predicate band alone, then
    the `any_of` family as ONE requirement any alternative satisfies."""
    singles = tuple((band,) for band in cell.predicate)
    return singles + ((cell.any_of,) if cell.any_of else ())


def satisfied(requirement: tuple[AxisBand, ...], frame: pd.DataFrame) -> pd.Series:
    """Rows already meeting this requirement — any alternative will do."""
    holds = pd.Series(False, index=frame.index)
    for band in requirement:
        holds |= band.mask(frame)
    return holds


def reduces(requirement: tuple[AxisBand, ...]) -> bool:
    """Whether satisfying this requirement means having LESS of something — a
    scalar bounded only from above. Nothing can be added to reach it."""
    return all(
        not band.is_span and band.at_least is None for band in requirement
    )


def stage_of(
    requirement: tuple[AxisBand, ...], operators: tuple[Operator, ...]
) -> tuple[Stage, Operator | None]:
    """Which stage serves this requirement, and by whom. A relevance-changing
    alternative is struck out first: an `any_of` family stays servable through
    a safe sibling, and a requirement with no safe alternative is REBUILD."""
    mintable = [
        (band, operator)
        for band in requirement
        for operator in operators
        if operator.mints(band)
    ]
    if not mintable:
        return Stage.SELECT, None
    safe = [
        (band, operator)
        for band, operator in mintable
        if band.member not in RELEVANCE_CHANGING
    ]
    if not safe:
        return Stage.REBUILD, None
    operator = safe[0][1]
    if reduces(requirement):
        return Stage.CONSTRAIN, operator
    stage = (
        Stage.CORPUS
        if operator.declaration.surface_origin is SurfaceOrigin.DOC_COPIED
        else Stage.QUERY_ONLY
    )
    return stage, operator


class Step(NamedTuple):
    """One requirement plus how it gets served."""

    requirement: tuple[AxisBand, ...]
    stage: Stage
    operator: Operator | None


class CellPlan(NamedTuple):
    """How a cell's shortfall gets served: the parents that survived every
    filter, the mints they still need, and what nothing can do for them."""

    cell: str
    parents: pd.DataFrame
    mints: tuple[Step, ...]
    unsatisfied: tuple[Step, ...]

    @property
    def servable(self) -> bool:
        return not self.parents.empty

    @property
    def needs_selection(self) -> bool:
        """Parents survive and nothing is left to mint — they already satisfy
        the cell. The shortfall is the fill's constraint, not missing features,
        so augmenting them would pay an LLM to change nothing."""
        return self.servable and not self.mints

    @property
    def partial(self) -> bool:
        """Whether these parents reach the cell only in part — the row is worth
        keeping, but it lands in whatever cell it measures into, not this one."""
        return bool(self.unsatisfied)

    @property
    def operators(self) -> tuple[Operator, ...]:
        return tuple(step.operator for step in self.mints if step.operator)


def plan(
    cell: ArchetypeCell, pool: pd.DataFrame, operators: tuple[Operator, ...]
) -> CellPlan:
    """Run the stages over one cell in harm order (d53). Only SELECT filters the
    pool — a CONSTRAIN band is a rewrite the model performs, never an entry
    requirement, so a parent that is merely too long still qualifies. A stage
    that empties the set leaves its requirement unsatisfied and the survivors
    are still worth producing with fewer targets met."""
    steps = tuple(Step(r, *stage_of(r, operators)) for r in requirements(cell))
    by_stage = {
        stage: tuple(step for step in steps if step.stage is stage)
        for stage in Stage
    }

    candidates = pool
    for step in by_stage[Stage.SELECT]:
        candidates = candidates[satisfied(step.requirement, candidates)]

    # corpus before query-only: it is the scarce resource, so it must not be
    # narrowed by a step that never runs short. Both apply their operator's own
    # eligibility — the difference between them is the corpus, not the rule.
    unsatisfied = list(by_stage[Stage.REBUILD])
    served: list[Step] = []
    for step in by_stage[Stage.CORPUS] + by_stage[Stage.QUERY_ONLY]:
        joined = (
            step.operator.eligible(candidates, cell.name)
            if not candidates.empty
            else candidates
        )
        if joined.empty:
            unsatisfied.append(step)
            continue
        candidates = joined
        served.append(step)

    # the cut is last and filters nobody: every surviving parent can be
    # shortened, and whether the shortened text still holds is the local
    # re-measure's verdict, not an eligibility question
    served.extend(by_stage[Stage.CONSTRAIN] if not candidates.empty else ())
    if candidates.empty:
        unsatisfied.extend(by_stage[Stage.CONSTRAIN])
    return CellPlan(cell.name, candidates, tuple(served), tuple(unsatisfied))


class Call(NamedTuple):
    """One LLM call: the steps it performs, and every requirement its result is
    measured against — its own plus everything already achieved (d55c)."""

    steps: tuple[Step, ...]
    verified: tuple[tuple[AxisBand, ...], ...]

    @property
    def operators(self) -> tuple[Operator, ...]:
        return tuple(step.operator for step in self.steps if step.operator)

    @property
    def cuts(self) -> bool:
        return any(step.stage is Stage.CONSTRAIN for step in self.steps)


WORD_AXES = frozenset({"length_words"})
"""Bounded-above axes measured in WORDS — the unit `mandatory_words` counts
in. The only axis with a declared DOWN direction today (d53f)."""


def mandatory_words(parent: pd.Series) -> int:
    """Words this row is already committed to keep verbatim — literal mint
    surfaces that must survive any later cut, however short it goes."""
    surfaces = parent.get("surfaces")
    return sum(len(str(s).split()) for s in surfaces) if surfaces else 0


def already_holds(band: AxisBand, value: float, extra_words: int) -> bool:
    """Whether a scalar band holds once the words a prior mint already
    committed to add are counted — the free win a parent may not need any
    edit to reach (2026-08 follow-up: most parents don't need StatRewrite at
    all, only the ones that would still miss the band after minting do)."""
    total = value + extra_words
    if band.at_least is not None and total < band.at_least:
        return False
    if band.below is not None and total >= band.below:
        return False
    return True


def unreachable(call: Call, parent: pd.Series) -> bool:
    """Whether this call's own ceiling is already broken by tokens the row
    cannot drop, before any rewrite is attempted (d55-followup). Checked only
    for calls that cut: an additive call has no ceiling to break."""
    if not call.cuts:
        return False
    floor = mandatory_words(parent)
    return any(
        band.member in WORD_AXES and band.below is not None and floor >= band.below
        for requirement in call.verified
        for band in requirement
    )


def needs_a_move(step: Step, parent: pd.Series) -> bool:
    """Whether THIS parent needs an operator for this step at all. Minting a
    span is never free — the literal surface is not in the query yet. A
    scalar band may already be free: the words a CORPUS mint (Inject) already
    committed to add are known from `parent["surfaces"]` before any call runs,
    so a parent whose length already lands in-band once that surface is
    counted needs no rewrite for it (2026-08 follow-up)."""
    band = step.requirement[0]
    if len(step.requirement) != 1 or band.is_span:
        return True  # any_of families and spans: unchanged, always mint
    value = parent.get(band.column)
    if value is None:
        return True
    return not already_holds(band, float(value), mandatory_words(parent))


def calls_for(
    plan: CellPlan, cell: ArchetypeCell, parent: pd.Series
) -> tuple[Call, ...]:
    """The plan as a sequence of calls (d55b): every addition in one, the cut in
    a second. Additions cannot contradict each other — only add-versus-remove
    can — so one call per mint would buy nothing and cost a call. Checks
    accumulate, and the SELECT requirements ride along from the first call: the
    parent already satisfies them and the child must not break them.

    A step this SPECIFIC parent does not need (`needs_a_move` is False) is
    dropped from the call — no instruction asks for it — but its requirement
    stays in `verified`, so the local re-measure still catches a wrong guess."""
    spoken_for = {step.requirement for step in plan.mints + plan.unsatisfied}
    keep = tuple(r for r in requirements(cell) if r not in spoken_for)
    movable, free = [], []
    for step in plan.mints:
        if step.stage is not Stage.CONSTRAIN and not needs_a_move(step, parent):
            free.append(step)
        else:
            movable.append(step)
    keep = keep + tuple(step.requirement for step in free)
    additive = tuple(s for s in movable if s.stage is not Stage.CONSTRAIN)
    cut = tuple(s for s in movable if s.stage is Stage.CONSTRAIN)

    calls: list[Call] = []
    achieved = list(keep)
    for group in (additive, cut):
        if not group:
            continue
        achieved.extend(step.requirement for step in group)
        calls.append(Call(group, tuple(achieved)))
    return tuple(calls)


def resolved_band(
    requirement: tuple[AxisBand, ...], parent: pd.Series
) -> AxisBand:
    """The one band a requirement is measured on: the `any_of` alternative
    actually woven in, so a URI injection is not checked against the three
    siblings it did not use."""
    minted = str(parent.get("bank") or "")
    return next((b for b in requirement if b.member == minted), requirement[0])


def targets_of(
    reqs: Iterable[tuple[AxisBand, ...]], parent: pd.Series
) -> Targets:
    """These requirements as one postcondition, spans and stats kept apart."""
    spans: list[SpanTarget] = []
    stats: list[StatTarget] = []
    for requirement in reqs:
        target = band_target(resolved_band(requirement, parent))
        (spans if isinstance(target, SpanTarget) else stats).append(target)
    return Targets(spans=tuple(spans), stats=tuple(stats))


def planned_targets(
    plan: CellPlan, cell: ArchetypeCell, parent: pd.Series
) -> Targets:
    """The postcondition a produced row is measured against: every requirement
    except the ones nothing could serve. Keyed on the requirement's VALUE — a
    frozen band tuple compares by value, where object identity silently never
    matched, because `plan()` and this function each build their own."""
    dropped = {step.requirement for step in plan.unsatisfied}
    return targets_of(
        (r for r in requirements(cell) if r not in dropped), parent
    )


def plan_report(
    sheet: pd.DataFrame,
    pool: pd.DataFrame,
    cells: dict[str, ArchetypeCell],
    operators: tuple[Operator, ...],
) -> pd.DataFrame:
    """Every hungry cell priced before any spend: how many parents survive, who
    mints what, and which requirements nothing can serve."""
    rows = []
    for line in sheet[sheet["missing"] > 0].itertuples(index=False):
        cell = cells.get(line.floor)
        if cell is None:
            rows.append({"cell": line.floor, "missing": line.missing})
            continue
        result = plan(cell, pool, operators)
        rows.append({
            "cell": line.floor,
            "missing": line.missing,
            "parents": len(result.parents),
            "mints": ", ".join(
                step.operator.declaration.operator for step in result.mints
            ),
            "unsatisfied": ", ".join(
                f"{step.stage}:{step.requirement[0].member}"
                for step in result.unsatisfied
            ),
            "partial": result.partial,
            "action": (
                "select" if result.needs_selection
                else "augment" if result.servable
                else "construct"
            ),
        })
    return pd.DataFrame(rows)
