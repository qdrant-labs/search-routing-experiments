"""spec verification (8): unreachable and exhausted floors produce exact
generation debt. The floor is never lowered, renamed, or reported as met."""

from __future__ import annotations

from rungs.rung_a import RungA
from tests.rungs_fixtures import default_config, make_catalog, make_row


def test_unreachable_floor_becomes_debt():
    """One row for `cell_scarce` but the floor demands three — debt = 2,
    reason = unreachable_supply."""
    catalog = make_catalog([
        make_row("A:1", "A", "scarce cell", ["cell_scarce"]),
        make_row("A:2", "A", "other cell", ["cell_common"]),
        make_row("A:3", "A", "other cell two", ["cell_common"]),
        make_row("A:4", "A", "other cell three", ["cell_common"]),
    ])
    from dataclasses import replace
    config = default_config(catalog, ceiling=10, floor_by_cell=3, floor_by_lane=1)
    floors = dict(config.floors)
    floors[("cell", "cell_scarce")] = 3  # supply is 1 -> debt = 2
    floors[("cell", "cell_common")] = 3
    cfg = replace(config, floors=floors)

    plan = RungA().compose(catalog, cfg, catalog_fp="fp")
    debt = plan.generation_debt
    scarce_line = debt[debt["name"] == "cell_scarce"]
    assert len(scarce_line) == 1
    row = scarce_line.iloc[0]
    assert int(row["floor"]) == 3
    assert int(row["admissible_supply"]) == 1
    assert int(row["missing"]) == 2
    assert row["reason"] == "unreachable_supply"
    # the reachable floor must not appear in debt
    assert "cell_common" not in set(debt["name"])


def test_floor_is_never_lowered_or_reported_as_met():
    """A stratum whose supply is short remains in the coverage report at its
    original floor value; only selected fills the shortage."""
    catalog = make_catalog([
        make_row("A:1", "A", "one and done", ["cell_scarce"]),
        make_row("A:2", "A", "filler", ["cell_common"]),
    ])
    from dataclasses import replace
    config = default_config(catalog, ceiling=5, floor_by_cell=1, floor_by_lane=1)
    floors = dict(config.floors)
    floors[("cell", "cell_scarce")] = 5
    cfg = replace(config, floors=floors)
    plan = RungA().compose(catalog, cfg, catalog_fp="fp")
    lines = {line.name: line for line in plan.coverage_report.strata}
    assert lines["cell_scarce"].floor == 5, "floor lowered to supply; contract broken"
    assert lines["cell_scarce"].selected <= 1
