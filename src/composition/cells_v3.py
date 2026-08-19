"""v3 archetype cells — additive to the frozen v2 `CELLS`. Loads
`cells_v3.yaml` through the same `ArchetypeCell` model (imported read-only), so
v2's cells.py is untouched. These band on the new taxonomy columns present only
in the v3 catalog; use them only with `catalog_v3.parquet`."""

from __future__ import annotations

from pathlib import Path

import yaml

from composition.cells import ArchetypeCell

V3_CELLS_PATH = Path(__file__).resolve().parent / "cells_v3.yaml"


def load_v3_cells() -> tuple[ArchetypeCell, ...]:
    if not V3_CELLS_PATH.exists():
        return ()
    entries = yaml.safe_load(V3_CELLS_PATH.read_text())["cells"]
    return tuple(ArchetypeCell.model_validate(entry) for entry in entries)


CELLS_V3: tuple[ArchetypeCell, ...] = load_v3_cells()
