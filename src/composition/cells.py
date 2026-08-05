"""The archetype cells — `cells.json` as typed Python, the objective the
fill selects against. A cell claims the catalog rows where every `predicate`
band holds and at least one `any_of` band holds. Cells carry no number read
off current holdings: supply, lane share and cost are computed at fill time.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import yaml
from pydantic import BaseModel, ConfigDict, model_validator

from composition.floors import SPAN_PREFIXES

CELLS_PATH = Path(__file__).resolve().parent / "cells.yaml"


class AxisBand(BaseModel):
    """Half-open band on one catalog column: `at_least` is inclusive,
    `below` is exclusive. Either bound may be open, never both."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    column: str
    at_least: float | None = None
    below: float | None = None

    @model_validator(mode="after")
    def _bounded(self) -> AxisBand:
        if self.at_least is None and self.below is None:
            raise ValueError(f"{self.column}: unbounded band claims every row")
        if (
            self.at_least is not None
            and self.below is not None
            and self.at_least >= self.below
        ):
            raise ValueError(f"{self.column}: empty band")
        return self

    @property
    def is_span(self) -> bool:
        """Whether the column counts spans instead of measuring a scalar."""
        return self.column.startswith(SPAN_PREFIXES)

    @property
    def member(self) -> str:
        """The taxonomy member the column names — a bank, or a stat."""
        return self.column.rpartition(".")[2]

    @property
    def expressions(self) -> list[str]:
        """The band as `column<op>value` strings — one per finite bound."""
        out = []
        if self.at_least is not None:
            out.append(f"{self.column}>={self.at_least:g}")
        if self.below is not None:
            out.append(f"{self.column}<{self.below:g}")
        return out

    def mask(self, catalog: pd.DataFrame) -> pd.Series:
        """Rows whose value falls in the band."""
        values = catalog[self.column]
        keep = pd.Series(True, index=catalog.index)
        if self.at_least is not None:
            keep &= values >= self.at_least
        if self.below is not None:
            keep &= values < self.below
        return keep


class ArchetypeCell(BaseModel):
    """One query archetype: the `predicate` bands hold together, and where
    the cell spans a family at least one `any_of` band holds. `predicts` is
    a prior to be tested by labelling, never used to allocate."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str
    predicate: tuple[AxisBand, ...] = ()
    any_of: tuple[AxisBand, ...] = ()
    predicts: tuple[str, ...]
    rationale: str
    source: str
    operator: str | None = None
    """Augmentation operator that mints this cell when natural supply is
    thin (`Declaration.operator`); None where the cell fills from supply."""

    @model_validator(mode="after")
    def _banded(self) -> ArchetypeCell:
        if not self.bands:
            raise ValueError(f"{self.name}: no bands, claims every row")
        return self

    @property
    def bands(self) -> tuple[AxisBand, ...]:
        """Every band the cell reads, predicate and alternatives alike."""
        return self.predicate + self.any_of

    @property
    def required_banks(self) -> frozenset[str]:
        """Span banks the cell requires PRESENT. A `below`-only band forbids
        its feature, so minting it would push the row out of the cell."""
        return frozenset(
            band.member
            for band in self.bands
            if band.is_span and band.at_least is not None and band.at_least >= 1
        )

    def select(
        self, catalog: pd.DataFrame, *, alternative: AxisBand | None = None
    ) -> pd.Series:
        """Rows the cell claims — predicate ANDed, `any_of` ORed unless one
        alternative is named."""
        keep = pd.Series(True, index=catalog.index)
        for band in self.predicate:
            keep &= band.mask(catalog)
        branch = (alternative,) if alternative is not None else self.any_of
        if branch:
            reached = pd.Series(False, index=catalog.index)
            for band in branch:
                reached |= band.mask(catalog)
            keep &= reached
        return keep


def _load(path: Path = CELLS_PATH) -> tuple[ArchetypeCell, ...]:
    """Read the declaration file, ignoring its `_note` documentation key."""
    cells = tuple(
        ArchetypeCell.model_validate(entry)
        for entry in yaml.safe_load(path.read_text())["cells"]
    )
    duplicated = len(cells) - len({cell.name for cell in cells})
    if duplicated:
        raise ValueError(f"{duplicated} duplicate cell name(s)")
    return cells


CELLS: tuple[ArchetypeCell, ...] = _load()
"""The cell set, in file order. Routing keys off `name`, so names are the
stable identifier a shortfall, an order-sheet row and a label all share."""

CELL_TO_PREDICATE: dict[str, list[str]] = {
    cell.name: [expr for band in cell.predicate for expr in band.expressions]
    for cell in CELLS
}
"""Each cell's predicate bands as `column<op>value` strings — the flat view
consumers read instead of walking `AxisBand` objects."""

CELL_TO_BANKS: dict[str, frozenset[str]] = {
    cell.name: cell.required_banks for cell in CELLS
}
"""Each cell's required span banks — the unit a cell name resolves to when an
operator looks up supply, which is indexed per bank and knows no cell names."""

GENERATION_CELLS: dict[str, frozenset[str]] = {
    op: frozenset(cell.name for cell in CELLS if cell.operator == op)
    for op in dict.fromkeys(cell.operator for cell in CELLS if cell.operator)
}
"""Which cells each augmentation operator mints, derived from the cells'
own `operator` field — the operators read this instead of hardcoding names,
so a rename in cells.yaml moves the routing with it."""
