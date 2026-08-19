"""v3 cells load and band only on columns the v3 catalog provides — additive
to the frozen v2 cells (which stay loadable and unchanged)."""

from composition.cells import CELLS
from composition.cells_v3 import CELLS_V3

# the columns the v3 catalog adds beyond v2 (build_v3_catalog NEW_PREFIXES)
V3_PREFIXES = (
    "corruption.", "unknown_token_rate.", "term_rarity.",
    "subword_fragmentation.", "query_corpus.",
)


def test_v3_cells_load_and_are_named_uniquely():
    assert len(CELLS_V3) == 5
    names = [c.name for c in CELLS_V3]
    assert len(set(names)) == len(names)
    assert set(names) >= {"rare_term_query", "typo_bearing_query"}


def test_v3_cells_band_only_on_new_taxonomy_columns():
    for cell in CELLS_V3:
        for band in cell.bands:
            assert band.column.startswith(V3_PREFIXES), (
                f"{cell.name} bands {band.column}, not a v3 taxonomy column"
            )


def test_v3_names_do_not_collide_with_v2():
    v2 = {c.name for c in CELLS}
    assert not (v2 & {c.name for c in CELLS_V3})
