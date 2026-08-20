"""v3 archetype cells — re-exported from `cells.py`, which loads both files
through one validator. Kept as a module so existing importers keep working; the
cells themselves band on columns only `catalog_v3.parquet` carries."""

from __future__ import annotations

from composition.cells import CELLS_V3, V3_CELLS_PATH

__all__ = ["CELLS_V3", "V3_CELLS_PATH"]
