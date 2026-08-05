"""Derived dispatch (d51c/d/e): which operator serves a cell is a function of
the bands a given parent fails, not a field on the cell. A parent qualifies
only when every band it fails is minted by ONE operator, which is what makes
the cell predicate reachable in a single hop.
"""

from __future__ import annotations

from enum import StrEnum
from typing import NamedTuple

import pandas as pd

from augmentation.core import Operator
from composition.cell_targets import generation_branches
from composition.cells import ArchetypeCell, AxisBand
from taxonomy_generators.verify import Targets


class Unservable(StrEnum):
    """Why a cell has no parent rung — distinct failures, one report (d51d)."""

    UNMINTABLE = "unmintable"
    """No operator declares the bands this cell needs put into a query."""
    UNSUPPLIED = "unsupplied"
    """An operator declares them; no parent (or no gold doc) supplies one."""


def requirements(cell: ArchetypeCell) -> tuple[tuple[AxisBand, ...], ...]:
    """The cell as independent requirements: each predicate band alone, then
    the `any_of` family as ONE requirement any alternative satisfies."""
    singles = tuple((band,) for band in cell.predicate)
    return singles + ((cell.any_of,) if cell.any_of else ())


def short_of(band: AxisBand, frame: pd.DataFrame) -> pd.Series:
    """Rows an operator could move INTO this band, which is only ever the ones
    below its lower bound: every operator adds or raises, so a row above an
    upper bound needs removal and nothing declares that (d51f). A band with no
    lower bound is unreachable by construction."""
    if band.at_least is None:
        return pd.Series(False, index=frame.index)
    return frame[band.column] < band.at_least


def predicate_minus_one(
    cell: ArchetypeCell, frame: pd.DataFrame, operator: Operator
) -> pd.Series:
    """Parents this operator can carry into the cell in one hop: every
    requirement it cannot mint already holds, and at least one it can mint is
    short of its bound."""
    holds = pd.Series(True, index=frame.index)
    reachable = pd.Series(False, index=frame.index)
    for requirement in requirements(cell):
        satisfied = pd.Series(False, index=frame.index)
        for band in requirement:
            satisfied |= band.mask(frame)
        mintable = [band for band in requirement if operator.mints(band)]
        if mintable:
            for band in mintable:
                reachable |= ~satisfied & short_of(band, frame)
        else:
            holds &= satisfied
    return holds & reachable


class CellRung(NamedTuple):
    """How one cell's shortfall can be served, and by whom."""

    cell: str
    operator: Operator | None
    parents: pd.DataFrame
    reason: Unservable | None

    @property
    def servable(self) -> bool:
        return self.reason is None


def viable_rungs(
    cell: ArchetypeCell, pool: pd.DataFrame, operators: tuple[Operator, ...]
) -> tuple[CellRung, ...]:
    """Every operator with parents reaching this cell, richest supply first.
    A cell is served per (cell, parent), so two operators may each serve a
    different slice of it — one parent is short and holds the identifier, the
    next is long and lacks it. DOC_COPIED operators additionally need a gold
    doc carrying the surface, enforced by their own `eligible` join; NONE
    operators inherit the parent's qrels and consult no corpus (d51d)."""
    found = []
    for operator in operators:
        candidates = pool[predicate_minus_one(cell, pool, operator)]
        if candidates.empty:
            continue
        parents = operator.eligible(candidates, cell.name)
        if not parents.empty:
            found.append(CellRung(cell.name, operator, parents, None))
    return tuple(sorted(found, key=lambda rung: -len(rung.parents)))


def dispatch(
    cell: ArchetypeCell, pool: pd.DataFrame, operators: tuple[Operator, ...]
) -> CellRung:
    """The richest rung, or why there is none: nothing declares these bands
    (unmintable) versus something does but no parent qualifies (unsupplied)."""
    rungs = viable_rungs(cell, pool, operators)
    if rungs:
        return rungs[0]
    mintable = any(
        operator.mints(band)
        for operator in operators
        for requirement in requirements(cell)
        for band in requirement
    )
    reason = Unservable.UNSUPPLIED if mintable else Unservable.UNMINTABLE
    return CellRung(cell.name, None, pool.iloc[:0], reason)


def cell_targets(cell: ArchetypeCell, parent: pd.Series) -> Targets:
    """The whole cell predicate as the postcondition (d51b), on the branch
    matching the member this row mints — an `any_of` cell is satisfied by the
    alternative actually woven in, not by all of them."""
    branches = generation_branches(cell)
    minted = str(parent.get("bank") or "")
    for branch in branches:
        if branch.alternative is not None and branch.alternative.member == minted:
            return branch.targets
    return branches[0].targets


def rung_report(
    sheet: pd.DataFrame,
    pool: pd.DataFrame,
    cells: dict[str, ArchetypeCell],
    operators: tuple[Operator, ...],
) -> pd.DataFrame:
    """Every hungry cell priced before any LLM spend: who serves it, on how
    many parents, and why not when nobody does."""
    rows = []
    for line in sheet[sheet["missing"] > 0].itertuples(index=False):
        cell = cells.get(line.floor)
        if cell is None:
            rows.append({
                "cell": line.floor, "missing": line.missing, "operator": None,
                "surface_origin": None, "parents": 0, "rungs": 0,
                "reason": Unservable.UNMINTABLE,
            })
            continue
        rungs = viable_rungs(cell, pool, operators)
        best = rungs[0] if rungs else dispatch(cell, pool, operators)
        rows.append({
            "cell": line.floor,
            "missing": line.missing,
            "operator": (
                best.operator.declaration.operator if best.operator else None
            ),
            "surface_origin": (
                str(best.operator.declaration.surface_origin)
                if best.operator else None
            ),
            "parents": sum(len(rung.parents) for rung in rungs),
            "rungs": len(rungs),
            "reason": best.reason,
        })
    return pd.DataFrame(rows)
