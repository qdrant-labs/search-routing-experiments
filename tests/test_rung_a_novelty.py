"""spec verification (6): novelty relaxation reaches zero and recovers the
unfiltered feasible set — novelty never converts feasible supply into apparent
generation debt (spec:225-227)."""

from __future__ import annotations

from rungs.rung_a import RungA
from tests.rungs_fixtures import default_config, make_catalog, make_row


def test_novelty_relaxes_to_zero_when_all_candidates_are_duplicates():
    """Every candidate for a floor is a near-duplicate; theta0 blocks them all,
    but relaxation drops the bar and the floor still fills."""
    # cell_only carrier: three near-identical queries in lane A. High theta
    # would block picks 2 and 3, but the floor requires 3, so relaxation
    # must reach far enough to admit them all.
    catalog = make_catalog([
        make_row("A:1", "A", "alpha beta gamma delta", ["cell_only"]),
        make_row("A:2", "A", "alpha beta gamma delta epsilon", ["cell_only"]),
        make_row("A:3", "A", "alpha beta gamma delta epsilon zeta", ["cell_only"]),
        make_row("B:1", "B", "totally different stuff here", ["cell_x"]),
    ])
    config = default_config(
        catalog,
        ceiling=4,
        floor_by_cell=1,
        floor_by_lane=1,
        theta0=0.99,
        theta_step=0.1,
    )
    # override cell_only floor to 3 to force relaxation
    from dataclasses import replace
    floors = dict(config.floors)
    floors[("cell", "cell_only")] = 3
    cfg = replace(config, floors=floors)
    plan = RungA().compose(catalog, cfg, catalog_fp="fp")
    picked = plan.planned_set["row_id"].tolist()
    for row_id in ("A:1", "A:2", "A:3"):
        assert row_id in picked, (
            f"{row_id} was gated out by novelty despite a floor demanding it; "
            "relaxation to zero did not run"
        )


def test_effective_theta_falls_when_needed():
    """The selection trace records the effective threshold each pick used —
    subsequent picks against near-duplicates should show theta below theta0."""
    catalog = make_catalog([
        make_row("A:1", "A", "same tokens here", ["cell_only"]),
        make_row("A:2", "A", "same tokens here plus", ["cell_only"]),
    ])
    from dataclasses import replace
    config = default_config(catalog, ceiling=2, floor_by_cell=2, floor_by_lane=2, theta0=0.99)
    floors = dict(config.floors)
    floors[("cell", "cell_only")] = 2
    cfg = replace(config, floors=floors)
    plan = RungA().compose(catalog, cfg, catalog_fp="fp")
    thetas = plan.selection_trace["theta_effective"].tolist()
    assert min(thetas) < 0.99, f"no pick relaxed theta below theta0; trace={thetas}"
