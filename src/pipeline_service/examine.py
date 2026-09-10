"""Profile a query set against the archetype cells and report what it lacks.

The deficit half of the augmentation loop, over queries a caller supplies rather
than a registered lane: extract features, band them, count cell membership, and
emit an order sheet whose entries feed `/augment/plan`.
"""

from __future__ import annotations

import io
import zipfile
from typing import TYPE_CHECKING

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from composition.catalog_axes import ID_COLUMNS, catalog_row
from composition.cells import DECLARED_CELLS, ArchetypeCell
from composition.floors import with_derived

if TYPE_CHECKING:
    from query_taxonomy.features import FeatureExtractor

MAX_UNCOMPRESSED_BYTES = 256 * 1024 * 1024
"""Zip-bomb ceiling, checked against the declared sizes before extracting."""

MAX_QUERIES = 50_000
"""Profiling is O(queries) regex work in-process; past this, use the CLI."""

QUERY_COLUMNS = ("query", "text")
"""Accepted spellings of the query-text column, in preference order."""


class CellCoverage(BaseModel):
    model_config = ConfigDict(frozen=True)

    cell: str
    rows: int
    share: float
    predicts: tuple[str, ...]
    operator: str | None
    looks_like: str | None


class OrderLine(BaseModel):
    """One generation order: a thin cell, how short it is, and the operator
    that mints it. `operator=None` means no family mints this cell — it fills
    from natural supply or not at all."""

    model_config = ConfigDict(frozen=True)

    cell: str
    have: int
    want: int
    deficit: int
    operator: str | None
    looks_like: str | None


class Examination(BaseModel):
    model_config = ConfigDict(frozen=True)

    queries: int
    cells_declared: int
    cells_covered: int
    uncovered: tuple[str, ...]
    coverage: tuple[CellCoverage, ...]
    order_sheet: tuple[OrderLine, ...]
    unclaimed_rows: int = Field(
        description="Rows no declared cell claims — supply the taxonomy sees "
        "but the archetypes do not describe."
    )


def queries_from_zip(blob: bytes) -> list[str]:
    """Query text from an uploaded lane archive.

    Accepts the `LanePaths` layout — any `queries.parquet`/`queries.csv` in the
    archive — so the upload contract is the one the repo already uses.
    """
    with zipfile.ZipFile(io.BytesIO(blob)) as archive:
        members = [
            info for info in archive.infolist()
            if not info.is_dir()
            and info.filename.rsplit("/", 1)[-1] in
            ("queries.parquet", "queries.csv")
        ]
        if not members:
            raise ValueError(
                "no queries.parquet or queries.csv in the archive "
                "(expected the lane layout)"
            )
        declared = sum(info.file_size for info in archive.infolist())
        if declared > MAX_UNCOMPRESSED_BYTES:
            raise ValueError(
                f"archive declares {declared} uncompressed bytes, "
                f"over the {MAX_UNCOMPRESSED_BYTES} limit"
            )
        member = members[0]
        # Read the member, never extract to disk — nothing here writes a path
        # the archive controls, so zip-slip has no surface.
        with archive.open(member) as handle:
            payload = handle.read(MAX_UNCOMPRESSED_BYTES + 1)
    if len(payload) > MAX_UNCOMPRESSED_BYTES:
        raise ValueError("archive member exceeds the uncompressed limit")

    buffer = io.BytesIO(payload)
    frame = (
        pd.read_parquet(buffer)
        if member.filename.endswith(".parquet")
        else pd.read_csv(buffer)
    )
    column = next((c for c in QUERY_COLUMNS if c in frame.columns), None)
    if column is None:
        raise ValueError(
            f"{member.filename} has no {' or '.join(QUERY_COLUMNS)} column; "
            f"found {sorted(frame.columns)[:8]}"
        )
    return [str(text) for text in frame[column].dropna()]


def build_catalog(queries: list[str], extractor: FeatureExtractor) -> pd.DataFrame:
    """The banded catalog the cells select against."""
    rows = [
        catalog_row("uploaded", str(index), True, extractor.resolve(text))
        for index, text in enumerate(queries)
    ]
    frame = pd.DataFrame(rows).fillna(0.0)
    ordered = [*ID_COLUMNS, *sorted(c for c in frame.columns if c not in ID_COLUMNS)]
    return with_derived(frame[ordered])


def examine(
    queries: list[str],
    extractor: FeatureExtractor,
    *,
    floor: int = 0,
    cells: tuple[ArchetypeCell, ...] = DECLARED_CELLS,
) -> Examination:
    """Cell coverage over the queries, plus an order sheet for thin cells.

    `floor` is the per-cell target a caller wants; 0 reports coverage only.
    """
    if not queries:
        raise ValueError("no queries to examine")
    if len(queries) > MAX_QUERIES:
        raise ValueError(f"{len(queries)} queries exceeds the {MAX_QUERIES} cap")

    catalog = build_catalog(queries, extractor)
    total = len(catalog)
    claimed = pd.Series(False, index=catalog.index)
    coverage: list[CellCoverage] = []
    orders: list[OrderLine] = []

    for cell in cells:
        try:
            mask = cell.select(catalog)
        except (KeyError, ValueError):
            # A cell banding on a column no bank emitted for THIS upload claims
            # nothing here — a coverage gap, not an error.
            mask = pd.Series(False, index=catalog.index)
        claimed |= mask
        rows = int(mask.sum())
        coverage.append(
            CellCoverage(
                cell=cell.name, rows=rows, share=round(rows / total, 4),
                predicts=cell.predicts, operator=cell.operator,
                looks_like=cell.looks_like,
            )
        )
        if floor and rows < floor:
            orders.append(
                OrderLine(
                    cell=cell.name, have=rows, want=floor, deficit=floor - rows,
                    operator=cell.operator, looks_like=cell.looks_like,
                )
            )

    covered = [c for c in coverage if c.rows > 0]
    return Examination(
        queries=total,
        cells_declared=len(cells),
        cells_covered=len(covered),
        uncovered=tuple(c.cell for c in coverage if c.rows == 0),
        coverage=tuple(sorted(coverage, key=lambda c: -c.rows)),
        order_sheet=tuple(sorted(orders, key=lambda o: -o.deficit)),
        unclaimed_rows=int((~claimed).sum()),
    )
