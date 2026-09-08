"""Catalog content identity includes every field consumed after composition."""

from __future__ import annotations

import pandas as pd

from rungs.catalog import CatalogConfig, _catalog_fingerprint, _row_fingerprints
from tests.rungs_fixtures import make_row


def test_debt_attribution_changes_catalog_fingerprint():
    first = pd.DataFrame([make_row("A:1", "A", "query", ["cell_x"], debt_id="cell:x")])
    second = first.assign(debt_id="cell:y")
    first["content_fp"] = _row_fingerprints(first)
    second["content_fp"] = _row_fingerprints(second)

    assert _catalog_fingerprint(first, CatalogConfig()) != _catalog_fingerprint(
        second, CatalogConfig()
    )
