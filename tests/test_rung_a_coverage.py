"""spec verification (5): structural gain uses normalized max deficit, not the
number of underfilled memberships. And (7): floors have priority over novelty
and provenance."""

from __future__ import annotations

from rungs.rung_a import RungA, _structural_gain
from tests.rungs_fixtures import default_config, make_catalog, make_row


def test_gain_is_normalized_max_deficit_not_membership_count():
    """A row covering ONE stratum at 0% fill must equal a row covering TWO
    strata at 0% fill (both = 1.0 max deficit); a count would give the pair
    a bonus. Then, a row covering TWO strata already at 80% must LOSE to a
    single-stratum row at 0% — a membership count would flip this (spec:206-208)."""
    import pandas as pd
    from dataclasses import replace

    catalog = make_catalog([
        make_row("A:1", "A", "one", ["huge_empty"]),
        make_row("A:2", "A", "two", ["small_fill_a", "small_fill_b"]),
    ])
    config = default_config(catalog, ceiling=2, floor_by_cell=1, floor_by_lane=1)
    floors = {
        ("cell", "huge_empty"): 10,
        ("cell", "small_fill_a"): 5,
        ("cell", "small_fill_b"): 5,
    }
    cfg = replace(config, floors=floors)

    # pre-seed 4 rows into small_fill_a and small_fill_b -> both at 4/5 = 80%
    seed = pd.DataFrame([
        {"_strata": frozenset([("cell", "small_fill_a"), ("cell", "small_fill_b")])}
        for _ in range(4)
    ])
    row_alone = pd.Series({"_strata": frozenset([("cell", "huge_empty")])})
    row_pair = pd.Series({
        "_strata": frozenset([("cell", "small_fill_a"), ("cell", "small_fill_b")])
    })

    gain_alone = _structural_gain(row_alone, seed, cfg)
    gain_pair = _structural_gain(row_pair, seed, cfg)
    assert gain_alone == 1.0, f"empty-huge cell must give gain 1.0, got {gain_alone}"
    assert abs(gain_pair - 0.2) < 1e-9, (
        f"nearly-filled pair must give gain 0.2 (=1-4/5), got {gain_pair}; "
        "a membership count would give 2.0 or 1.0 by picking the SUM/MAX of "
        "underfilled memberships"
    )
    assert gain_alone > gain_pair


def test_floors_beat_novelty_and_provenance():
    """A floor-underfilled candidate must be selected even if a novelty- and
    provenance-friendlier candidate exists (spec:222-227 + spec:242-252)."""
    catalog = make_catalog([
        # only carrier of cell_z — floor forces selection
        make_row("A:1", "A", "unique query one", ["cell_z"]),
        # very different tokens (higher novelty) and rarer family, but covers
        # a cell whose floor is already met
        make_row("A:2", "A", "totally distinct wording", ["cell_x"], family="fam_rare"),
    ])
    config = default_config(catalog, ceiling=2, floor_by_cell=1, floor_by_lane=1)
    plan = RungA().compose(catalog, config, catalog_fp="fp")
    assert "A:1" in plan.planned_set["row_id"].tolist(), (
        "A:1 must be picked because cell_z's floor demands it, regardless of "
        "A:2's higher novelty or rarer family"
    )
