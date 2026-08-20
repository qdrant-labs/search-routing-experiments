"""v3 cells load, band only on columns a v3 build can produce, and are
DISPATCHABLE — the v2 quota set stays 44 cells wide while the name lookup covers
both, which is what lets the loop serve a v3 cell and keeps its children on the
audited path. The count is deliberately NOT asserted: the set is additive."""

from pathlib import Path

import pandas as pd
import pytest

from composition.cells import (
    CELL_TO_BANKS,
    CELLS,
    CELLS_BY_NAME,
    CELLS_V3,
    ArchetypeCell,
)
from composition.floors import with_derived

V3_CATALOG = (
    Path(__file__).resolve().parent.parent
    / "src" / "data" / "v3" / "catalog_v3.parquet"
)
NEEDS_CATALOG = pytest.mark.skipif(
    not V3_CATALOG.exists(), reason="v3 catalog not built"
)


def test_v3_cells_load_and_are_named_uniquely():
    names = [c.name for c in CELLS_V3]
    assert names, "cells_v3.yaml loaded nothing"
    assert len(set(names)) == len(names)
    assert set(names) >= {"rare_term_query", "typo_bearing_query"}


@NEEDS_CATALOG
def test_every_v3_band_column_is_built_or_backfilled():
    """A band on a column no build emits is a KeyError inside `cell.select`,
    which is how a cell takes the selector down mid-run."""
    from scripts.build_v3_catalog import _absent_identifier_columns

    columns = set(with_derived(pd.read_parquet(V3_CATALOG)).columns)
    buildable = columns | set(_absent_identifier_columns(columns))
    unbuildable = sorted(
        band.column
        for cell in CELLS_V3
        for band in cell.bands
        if band.column not in buildable
    )
    assert not unbuildable, unbuildable


@NEEDS_CATALOG
def test_active_cells_never_band_on_a_column_the_catalog_lacks():
    """The selector's own guard: a cell whose columns are not yet backfilled is
    BLOCKED, not silently empty and not a crash."""
    from composition.pool_v3 import LabelledPool

    catalog = with_derived(pd.read_parquet(V3_CATALOG)).head(1)
    for cell in LabelledPool().active_cells(catalog):
        cell.select(catalog)


def test_v3_names_do_not_collide_with_v2():
    v2 = {c.name for c in CELLS}
    assert not (v2 & {c.name for c in CELLS_V3})


def test_v2_quota_set_stays_frozen_at_44():
    """`cellfill` quotas per cell in CELLS and asserts its length, so widening
    that tuple would change every v2 quota."""
    assert len(CELLS) == 44
    assert not ({c.name for c in CELLS_V3} & {c.name for c in CELLS})


def test_v3_cells_are_dispatchable():
    for cell in CELLS_V3:
        assert cell.name in CELLS_BY_NAME, f"{cell.name} unreachable by the loop"
        assert cell.name in CELL_TO_BANKS
    assert len(CELLS_BY_NAME) == len(CELLS) + len(CELLS_V3)


def test_a_v3_cell_child_is_not_filed_as_floor_based():
    """The audit-gate regression: `_augmented_rows`' floor_based branch admits
    any row whose floor is not a cell name, so a v3 cell missing from the lookup
    let its children skip the audit."""
    from hybrid_search_rrf_dataset.labels import _augmented_rows

    pool = pd.DataFrame({
        "query_id": ["a", "b"],
        "home_lane": ["lane", "lane"],
        "credit_gate": ["none", "none"],
        "floor": [CELLS_V3[0].name, "id:uuid"],
    })
    admitted = _augmented_rows(pool, pd.DataFrame({"query_id": []}), "lane",
                               "floor_based")
    assert list(admitted["floor"]) == ["id:uuid"]


def test_predicts_accepts_either_vocabulary_and_normalises():
    assert CELLS_BY_NAME["rare_term_query"].predicts_class == frozenset({"sparse"})
    assert CELLS_BY_NAME["bare_concept_token"].predicts_class == frozenset({"dense"})


def test_predicts_rejects_a_name_that_is_neither_route_nor_class():
    band = {"column": "length.length_words", "at_least": 1}
    try:
        ArchetypeCell.model_validate({
            "name": "x", "predicate": [band], "predicts": ["dense_ish"],
            "rationale": "r", "source": "s",
        })
    except ValueError as error:
        assert "not routes" in str(error)
    else:
        raise AssertionError("an unknown predicts value was accepted")


@NEEDS_CATALOG
def test_catalog_build_covers_the_additive_v3_labels():
    """The 342 in-loop labelled rows must be catalog rows, or they carry zero
    cell membership silently while still counting toward per-dataset floors."""
    from scripts.build_v3_catalog import _pool_labels

    v3_labels = V3_CATALOG.parent / "labels.parquet"
    if not v3_labels.exists():
        pytest.skip("no additive v3 labels yet")
    keys = set(map(tuple, pd.read_parquet(
        v3_labels, columns=["dataset", "query_id"]
    ).astype({"query_id": str}).itertuples(index=False)))
    pooled = set(map(tuple, _pool_labels()[["dataset", "query_id"]]
                     .astype({"query_id": str}).itertuples(index=False)))
    missing = keys - pooled
    assert not missing, f"{len(missing)} v3-labelled rows absent from the build"
