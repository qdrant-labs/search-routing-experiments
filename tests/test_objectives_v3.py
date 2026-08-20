"""The diversity-floor order sheet at target_total scale, the inversion
bound's arithmetic, and the sheet's round-trip through the augmentation
loop's own readers."""

from types import SimpleNamespace

import pandas as pd
import pytest

from augmentation.config import AugmentationConfig
from augmentation.loop import AugmentationLoop
from composition.objectives import DiversityFloors, InversionBound
from composition.recipe import Recipe


def _cell(name, predicts=()):
    return SimpleNamespace(
        name=name, bands=(), predicts_class=frozenset(predicts)
    )


def _frame(rows):
    frame = pd.DataFrame(rows)
    frame["cells"] = frame["cells"].map(frozenset)
    return frame


POOL = _frame(
    [{"dataset": "a", "query_id": str(i), "cells": {"X"},
      "corruption_degree": "clean", "route_class_any": "dense",
      "is_waste": False} for i in range(80)]
    + [{"dataset": "b", "query_id": f"b{i}", "cells": set(),
        "corruption_degree": "clean", "route_class_any": "sparse",
        "is_waste": False} for i in range(20)]
)

RECIPE = Recipe.v3(stratum_floor=25, target_total=1_000)


def test_class_debt_nets_the_pools_own_tier0_supply():
    floors = DiversityFloors(RECIPE, ())
    debt = floors.class_debt(
        pd.concat([POOL, POOL.head(5).assign(is_waste=True)])
    ).set_index("route_class")
    # targets 450/450/100; waste rows never count as supply
    assert debt.at["dense", "debt"] == 450 - 80
    assert debt.at["sparse", "debt"] == 450 - 20
    assert debt.at["hybrid", "debt"] == 100
    assert debt.at["hybrid", "x_under"] == float("inf")


def test_order_sheet_carries_the_class_debt_not_the_floor():
    cells = (_cell("X", predicts=("dense",)), _cell("Y", predicts=("sparse",)))
    floors = DiversityFloors(RECIPE, cells)
    sheet = floors.order_sheet(POOL.head(40), POOL)
    lines = sheet.set_index("floor")
    # X hosts all of dense's debt (cap 5 * 0.8 * 1000 = 4000 doesn't bind);
    # the sheet is a debt ledger: credit 0 at build, missing == amount
    assert lines.at["X", "amount"] == 450 - 80
    assert lines.at["X", "credit"] == 0.0
    assert lines.at["X", "missing"] == 450 - 80
    # Y has zero natural supply -> zero K_cap capacity -> the floor is its
    # minimum and the line is declared generation_only
    assert lines.at["Y", "missing"] == 25.0
    assert lines.at["Y", "reason"] == "generation_only"
    # sparse's debt (430) fits no host; hybrid (100) has no host cell at all
    assert lines.at["uncovered:sparse", "missing"] == 430.0
    assert lines.at["uncovered:hybrid", "missing"] == 100.0
    assert lines.at["uncovered:hybrid", "reason"] == "no_cell_hosts"


def test_corruption_line_scales_to_target_total():
    floors = DiversityFloors(RECIPE, ())
    sheet = floors.order_sheet(POOL.head(40), POOL).set_index("floor")
    census = pd.read_parquet(AugmentationConfig().paths.corruption_census)
    rate = (census["n_sampled"] * census["any_span"]).sum() / census["n_sampled"].sum()
    # the target is rate x target_total net of the pool's damaged rows (0),
    # never rate x the selection
    assert sheet.at["corruption:light", "missing"] == round(rate * 1_000)


def test_waterfill_splits_equally_and_spills_from_full_hosts():
    fill = DiversityFloors._waterfill
    even = fill(430.0, {"s1": 500.0, "s2": 2000.0})
    assert even["s1"] == pytest.approx(215.0)
    assert even["s2"] == pytest.approx(215.0)
    clamped = fill(430.0, {"s1": 50.0, "s2": 200.0})
    assert clamped["s1"] == pytest.approx(50.0)
    assert clamped["s2"] == pytest.approx(200.0)  # leftover 180 stays unplaced


def test_inversion_bound_ratio_and_floor_forced_arithmetic():
    # X sits only in lane a (80% of pool): p_uniform = .5*1 + .5*0 = .5 is NOT
    # the minimum; pool_share gives .8*1 = .8; capped(0.5) gives .5 -> min .5
    selected = POOL.head(50).assign(corruption_degree="clean")
    bound = InversionBound(Recipe.v3(stratum_floor=5), (_cell("X"),))
    row = bound.report(selected, POOL).iloc[0]
    p_lane_a = 1.0  # every lane-a row holds X
    assert row["min_weighted_share"] == pytest.approx(0.5 * p_lane_a, abs=1e-6)
    assert row["max_ratio"] == pytest.approx(1.0 / 0.5, abs=0.01)
    assert row["floor_ratio"] == pytest.approx((5 / 50) / 0.5, abs=0.01)
    # observed ratio (2.0) far exceeds what the floor mandates (0.2)
    assert not row["floor_forced"]


def test_sheet_roundtrips_through_the_loop(tmp_path):
    floors = DiversityFloors(RECIPE, (_cell("Y", predicts=("sparse",)),))
    sheet = floors.order_sheet(POOL.head(40), POOL)
    path = tmp_path / "order_sheet.parquet"
    sheet.to_parquet(path, index=False)
    loop = AugmentationLoop(POOL.head(3), sheet_path=path)
    hungry = loop.order_sheet()
    assert set(hungry["floor"]) == {
        "Y", "corruption:light", "uncovered:dense", "uncovered:sparse",
        "uncovered:hybrid",
    }
    readout = loop.hungry()
    # corruption:light must be SERVABLE by a registered operator today — the
    # "no writer emits corruption floors" gap closing — while staying behind
    # its declared audit gate (d42h), which is what runnable=False means here
    corr = readout[readout["floor"] == "corruption:light"].iloc[0]
    assert corr["operator"] == "corrupt"
    assert corr["gate"] == "declaration_audit" and not corr["runnable"]
    # an uncovered accounting line is visible but nothing serves it
    unc = readout[readout["floor"] == "uncovered:dense"].iloc[0]
    assert pd.isna(unc["operator"]) and not unc["runnable"]
