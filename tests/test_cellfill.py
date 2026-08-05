"""The lane-share cap arithmetic, and the per-cell readout when the
artifact has been built.
"""

import pandas as pd
import pytest

from composition.cellfill import CellFill, LaneCap
from composition.cells import CELLS


def _counts(*values: int) -> pd.Series:
    return pd.Series(values, index=[f"lane{i}" for i, _ in enumerate(values)])


def test_the_target_holds_when_the_lanes_can_satisfy_it():
    cap = LaneCap(_counts(100, 100, 100, 100, 100, 100), draw=50, target=0.2)
    assert cap.share == 0.2
    assert not cap.infeasible
    assert cap.capacity == 50


def test_few_lanes_record_the_achievable_floor():
    cap = LaneCap(_counts(100, 100, 100), draw=50, target=0.2)
    assert cap.floor == pytest.approx(0.34)
    assert cap.share == pytest.approx(0.34)
    assert cap.infeasible
    assert cap.capacity == 50


def test_a_single_lane_cell_fills_at_share_one():
    cap = LaneCap(_counts(80), draw=50, target=0.2)
    assert cap.share == 1.0
    assert cap.infeasible
    assert cap.capacity == 50


def test_thin_supply_caps_capacity_below_the_draw():
    cap = LaneCap(_counts(3, 2), draw=50, target=0.2)
    assert cap.floor == 1.0
    assert cap.capacity == 5


def test_an_empty_cell_has_no_capacity():
    cap = LaneCap(_counts(), draw=50, target=0.2)
    assert cap.capacity == 0


@pytest.mark.skipif(
    not CellFill().report_path.exists(), reason="cell fill not built"
)
def test_the_readout_covers_every_cell_and_respects_its_cap():
    report = pd.read_parquet(CellFill().report_path)
    assert list(report["cell"]) == [cell.name for cell in CELLS]
    assert (report["lane_share_cap"] >= report["lane_share_floor"]).all()
    assert (report["natural_capacity"] <= report["natural_rows"]).all()
