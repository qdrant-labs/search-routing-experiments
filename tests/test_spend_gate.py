"""Pre-label spend gate verdict, reporting, and immutability contract."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from dataclasses import replace

import pandas as pd
import pytest

from rungs.rung_a import RungA
from rungs.spend_gate import SpendGate
from tests.rungs_fixtures import default_config, make_catalog, make_row


def _plan(catalog, config):
    return RungA().compose(catalog, config, catalog_fp="fp")


def _assess(plan, catalog, *, theta0=0.7):
    return SpendGate().assess(
        plan.planned_set,
        plan.selection_trace,
        plan.coverage_report,
        catalog=catalog,
        theta0=theta0,
    )


def test_healthy_plan_passes():
    catalog = make_catalog([
        make_row("A:1", "A", "alpha query one", ["cell_x"]),
        make_row("A:2", "A", "beta query two", ["cell_x"]),
        make_row("B:1", "B", "gamma query three", ["cell_y"]),
        make_row("B:2", "B", "delta query four", ["cell_y"]),
    ])
    config = default_config(catalog, floor_by_cell=2, floor_by_lane=1)
    plan = _plan(catalog, config)

    report = _assess(plan, catalog)

    assert report.verdict == "PASS"
    assert report.failed_conditions == ()
    assert report.reachable_floors_unmet == ()
    assert report.assessment_complete
    assert report.predicted_yield.status == "not_activated"


def test_missing_catalog_fails_closed():
    catalog = make_catalog([
        make_row("A:1", "A", "alpha query", ["cell_x"]),
        make_row("A:2", "A", "beta query", ["cell_x"]),
    ])
    config = default_config(catalog, floor_by_cell=2, floor_by_lane=1)
    plan = _plan(catalog, config)

    report = SpendGate().assess(
        plan.planned_set,
        plan.selection_trace,
        plan.coverage_report,
        theta0=config.theta0,
    )

    assert report.verdict == "FAIL"
    assert not report.assessment_complete
    assert "catalog" in " ".join(report.failed_conditions)


def test_reachable_floor_left_unmet_fails():
    """Supply exists for the floor (achievable >= floor) but the ceiling stops
    selection short — a wasted-supply spend failure, derived from coverage
    alone."""
    catalog = make_catalog([
        make_row(f"A:{i}", "A", f"query number {i}", ["cell_x"]) for i in range(6)
    ])
    config = replace(
        default_config(catalog, ceiling=2, floor_by_cell=5, floor_by_lane=1),
        floors={("cell", "cell_x"): 5, ("lane", "A"): 1},
    )
    plan = _plan(catalog, config)

    report = _assess(plan, catalog, theta0=config.theta0)

    assert report.verdict == "FAIL"
    assert any("cell_x" in c for c in report.failed_conditions)


def test_unreachable_floor_does_not_fail():
    """One row for a floor of three: unreachable, so debt — never a gate FAIL."""
    catalog = make_catalog([
        make_row("A:1", "A", "lonely scarce row", ["cell_scarce"]),
        make_row("A:2", "A", "common one", ["cell_common"]),
        make_row("A:3", "A", "common two", ["cell_common"]),
    ])
    config = replace(
        default_config(catalog, floor_by_lane=1),
        floors={("cell", "cell_scarce"): 3, ("cell", "cell_common"): 2, ("lane", "A"): 1},
    )
    plan = _plan(catalog, config)

    report = _assess(plan, catalog, theta0=config.theta0)

    assert report.verdict == "PASS"
    assert any(f.name == "cell_scarce" for f in report.unreachable_floors)


def test_missing_answer_coverage_fails():
    catalog = make_catalog([
        make_row("A:1", "A", "covered row", ["cell_x"]),
        make_row("A:2", "A", "uncovered row", ["cell_x"]),
    ])
    config = default_config(catalog, floor_by_cell=2, floor_by_lane=1)
    plan = _plan(catalog, config)
    planned = plan.planned_set.copy()
    planned.loc[planned["query_id"] == "2", "answer_covered"] = False

    report = SpendGate().assess(
        planned,
        plan.selection_trace,
        plan.coverage_report,
        catalog=catalog,
        theta0=config.theta0,
    )

    assert report.verdict == "FAIL"
    assert report.rows_answer_uncovered == 1
    assert any("answer coverage" in c for c in report.failed_conditions)


def test_family_concentration_is_reported_without_failing():
    catalog = make_catalog([
        make_row(f"A:{i}", "A", f"generated variant {i}", ["cell_x"],
                 provenance="synthetic", operator="paraphrase", family="one_template")
        for i in range(4)
    ])
    config = default_config(catalog, floor_by_cell=2, floor_by_lane=1)
    plan = _plan(catalog, config)

    report = _assess(plan, catalog, theta0=config.theta0)

    assert report.verdict == "PASS"
    assert report.largest_family_share == 1.0
    assert report.effective_families == 1.0


def test_unselected_generated_rows_are_informational_when_floor_is_met():
    """Unused generated supply is campaign evidence, not a reason to reject a
    currently healthy labeling plan whose floor is already satisfied."""
    catalog = make_catalog([
        make_row("A:1", "A", "natural covers the floor already", ["cell_x"]),
        make_row("A:2", "A", "natural two also covers it", ["cell_x"]),
        make_row("B:g1", "B", "generated for the shortage", ["cell_x"],
                 provenance="synthetic", operator="para", debt_id="cell:cell_x"),
        make_row("B:g2", "B", "generated for the shortage too", ["cell_x"],
                 provenance="synthetic", operator="para", debt_id="cell:cell_x"),
    ])
    config = replace(
        default_config(catalog, ceiling=10, floor_by_lane=1),
        floors={("cell", "cell_x"): 2, ("lane", "A"): 1},
        lane_budgets={"A": 10, "B": 0},  # lane B admitted but unspendable
    )
    plan = _plan(catalog, config)

    report = _assess(plan, catalog, theta0=config.theta0)

    funnel = {row.debt_id: row for row in report.augmentation_funnel}
    assert funnel["cell:cell_x"].admitted == 2
    assert funnel["cell:cell_x"].selected == 0
    assert report.verdict == "PASS"
    assert report.failed_conditions == ()


def test_funnel_passes_when_shortage_is_repaired():
    catalog = make_catalog([
        make_row("A:g1", "A", "generated shortage row one", ["cell_x"],
                 provenance="synthetic", operator="para", debt_id="cell:cell_x"),
        make_row("A:g2", "A", "generated shortage row two", ["cell_x"],
                 provenance="synthetic", operator="para", debt_id="cell:cell_x"),
    ])
    config = replace(
        default_config(catalog, ceiling=10, floor_by_lane=1),
        floors={("cell", "cell_x"): 2, ("lane", "A"): 1},
    )
    plan = _plan(catalog, config)

    report = _assess(plan, catalog, theta0=config.theta0)

    funnel = {row.debt_id: row for row in report.augmentation_funnel}
    assert funnel["cell:cell_x"].selected == 2
    assert report.verdict == "PASS"


def test_novelty_reports_share_below_initial_threshold():
    catalog = make_catalog([
        make_row("A:1", "A", "first query", ["cell_x"]),
        make_row("A:2", "A", "second query", ["cell_x"]),
        make_row("A:3", "A", "third query", ["cell_x"]),
    ])
    config = default_config(catalog, floor_by_cell=2, floor_by_lane=1)
    plan = _plan(catalog, config)
    trace = plan.selection_trace.copy()
    trace["novelty"] = [0.2, 0.8]

    report = SpendGate().assess(
        plan.planned_set,
        trace,
        plan.coverage_report,
        catalog=catalog,
        theta0=0.7,
    )

    assert report.novelty is not None
    assert report.novelty.theta0 == 0.7
    assert report.novelty.share_below_theta0 == pytest.approx(0.5)


def test_uncalibrated_concentration_has_no_threshold_warning():
    catalog = make_catalog([
        make_row(
            f"A:{i}",
            "A",
            f"generated variant {i}",
            ["cell_x"],
            provenance="synthetic",
            operator="paraphrase",
            family="one_template",
        )
        for i in range(4)
    ])
    config = default_config(catalog, floor_by_cell=2, floor_by_lane=1)
    plan = _plan(catalog, config)

    report = _assess(plan, catalog, theta0=config.theta0)

    assert report.largest_family_share == 1.0
    assert report.warnings == ()


def test_gate_is_read_only_deterministic_and_report_is_immutable():
    catalog = make_catalog([
        make_row("A:1", "A", "first query", ["cell_x"]),
        make_row("A:2", "A", "second query", ["cell_x"]),
    ])
    config = default_config(catalog, floor_by_cell=2, floor_by_lane=1)
    plan = _plan(catalog, config)
    before = plan.planned_set.copy()

    gate = SpendGate()
    r1 = gate.assess(
        plan.planned_set,
        plan.selection_trace,
        plan.coverage_report,
        catalog=catalog,
        theta0=config.theta0,
    )
    r2 = gate.assess(
        plan.planned_set,
        plan.selection_trace,
        plan.coverage_report,
        catalog=catalog,
        theta0=config.theta0,
    )

    pd.testing.assert_frame_equal(plan.planned_set, before)
    assert r1.fingerprint() == r2.fingerprint()
    with pytest.raises(FrozenInstanceError):
        r1.verdict = "FAIL"
    with pytest.raises(AttributeError):
        r1.warnings.append("mutated")
