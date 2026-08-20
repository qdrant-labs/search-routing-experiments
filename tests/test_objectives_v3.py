"""The diversity-floor order sheet, the inversion bound's arithmetic, and the
sheet's round-trip through the augmentation loop's own readers."""

from types import SimpleNamespace

import pandas as pd
import pytest

from augmentation.config import AugmentationConfig
from augmentation.loop import AugmentationLoop
from composition.objectives import DiversityFloors, InversionBound
from composition.recipe import Recipe


def _cell(name):
    return SimpleNamespace(name=name, bands=())


def _frame(rows):
    frame = pd.DataFrame(rows)
    frame["cells"] = frame["cells"].map(frozenset)
    return frame


POOL = _frame(
    [{"dataset": "a", "query_id": str(i), "cells": {"X"},
      "corruption_degree": "clean"} for i in range(80)]
    + [{"dataset": "b", "query_id": f"b{i}", "cells": set(),
        "corruption_degree": "clean"} for i in range(20)]
)


def test_order_sheet_carries_cell_and_corruption_lines():
    selected = POOL.head(40)
    floors = DiversityFloors(Recipe.v3(stratum_floor=25), (_cell("X"), _cell("Y")))
    sheet = floors.order_sheet(selected, POOL, certified_total=100)
    lines = sheet.set_index("floor")
    # Y has zero natural supply -> K_cap admits 0 -> declared generation_only
    assert lines.at["Y", "reason"] == "generation_only"
    assert lines.at["Y", "missing"] == 25.0
    # X is fully covered by the selected slice (40 rows hold it) -> no line
    assert "X" not in lines.index
    # the corruption target is the census's own pooled rate, never a hand rate
    census = pd.read_parquet(AugmentationConfig().paths.corruption_census)
    rate = (census["n_sampled"] * census["any_span"]).sum() / census["n_sampled"].sum()
    assert lines.at["corruption:light", "amount"] == round(rate * len(selected))


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
    floors = DiversityFloors(Recipe.v3(stratum_floor=25), (_cell("Y"),))
    sheet = floors.order_sheet(POOL.head(40), POOL, certified_total=100)
    path = tmp_path / "order_sheet.parquet"
    sheet.to_parquet(path, index=False)
    loop = AugmentationLoop(POOL.head(3), sheet_path=path)
    hungry = loop.order_sheet()
    assert set(hungry["floor"]) == {"Y", "corruption:light"}
    readout = loop.hungry()
    # corruption:light must be SERVABLE by a registered operator today — the
    # "no writer emits corruption floors" gap closing — while staying behind
    # its declared audit gate (d42h), which is what runnable=False means here
    corr = readout[readout["floor"] == "corruption:light"].iloc[0]
    assert corr["operator"] == "corrupt"
    assert corr["gate"] == "declaration_audit" and not corr["runnable"]
