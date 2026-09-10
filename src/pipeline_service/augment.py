"""Cell-driven augmentation planning — the half that makes an order sheet actionable.

`/augment/plan` serves the FLOOR path: one floor key, one operator. But
`/examine` emits CELLS, and a cell reaches an operator through its bands, not
through a floor (see `StatRewrite.serves`). This resolves a cell name into the
per-requirement steps `augmentation.dispatch` derives, so an order line can be
acted on directly.
"""

from __future__ import annotations

import pandas as pd
from pydantic import BaseModel, ConfigDict

from augmentation.core import Operator
from augmentation.dispatch import Stage, requirements, stage_of
from composition.cells import DECLARED_CELLS


class Band(BaseModel):
    """One band of a requirement, in the caller's terms rather than ours."""

    model_config = ConfigDict(frozen=True)

    member: str
    is_span: bool
    at_least: float | None = None
    at_most: float | None = None


class Step(BaseModel):
    """One of the cell's requirements and how it gets served.

    `operator` is null when nothing mints this requirement — `SELECT` means it
    must be found in existing supply, `REBUILD` that every alternative would
    change what the query asks for.
    """

    model_config = ConfigDict(frozen=True)

    requirement: tuple[Band, ...]
    stage: str
    operator: str | None
    instruction: str | None = None
    targets: dict | None = None


class CellPlan(BaseModel):
    model_config = ConfigDict(frozen=True)

    cell: str
    looks_like: str | None
    predicts: tuple[str, ...]
    steps: tuple[Step, ...]
    actionable: int


_BY_NAME = {cell.name: cell for cell in DECLARED_CELLS}


def _band(band) -> Band:
    return Band(
        member=str(band.member),
        is_span=bool(band.is_span),
        at_least=band.at_least,
        at_most=getattr(band, "at_most", None),
    )


def plan_cell(
    cell_name: str, text: str, parent: dict, operators: tuple[Operator, ...]
) -> CellPlan:
    """Resolve a cell into per-requirement steps, with a prompt where one exists.

    Operators take the cell NAME in their `floor` slot when a requirement is
    supplied — the convention `dispatch.plan` already uses.
    """
    cell = _BY_NAME.get(cell_name)
    if cell is None:
        raise ValueError(
            f"unknown cell {cell_name!r}; /examine reports the declared names"
        )

    row = pd.Series(parent)
    steps: list[Step] = []
    for requirement in requirements(cell):
        stage, operator = stage_of(requirement, operators)
        instruction = targets = None
        if operator is not None:
            instruction = operator.instruction(cell.name, row, requirement)
            targets = operator.targets(cell.name, row, requirement).model_dump()
        steps.append(
            Step(
                requirement=tuple(_band(b) for b in requirement),
                stage=str(Stage(stage).value),
                operator=operator.declaration.operator if operator else None,
                instruction=instruction,
                targets=targets,
            )
        )
    return CellPlan(
        cell=cell.name,
        looks_like=cell.looks_like,
        predicts=cell.predicts,
        steps=tuple(steps),
        actionable=sum(1 for s in steps if s.operator),
    )
