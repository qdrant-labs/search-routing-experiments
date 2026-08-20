"""The diversity floors — declared-band marginals, the floor-only order sheet,
the class-debt ledger — plus the inversion bound's arithmetic and the sheet's
round-trip through the augmentation loop's own readers."""

from types import SimpleNamespace

import pandas as pd
import pytest

from augmentation.config import AugmentationConfig
from augmentation.loop import AugmentationLoop
from composition.objectives import DiversityFloors, InversionBound
from composition.pool_v3 import UNKNOWN
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
    pool = pd.concat([
        POOL,
        POOL.head(5).assign(is_waste=True),
        POOL.head(3).assign(
            query_id=["u0", "u1", "u2"], route_class="", certified=False
        ),
    ])
    debt = floors.class_debt(pool).set_index("route_class")
    # targets 450/450/100. Tier-0 non-waste is the supply bar: the three
    # uncertified rows count, the five waste rows never do
    assert debt.at["dense", "debt"] == 450 - 83
    assert debt.at["sparse", "debt"] == 450 - 20
    assert debt.at["hybrid", "debt"] == 100
    assert debt.at["hybrid", "x_under"] == float("inf")


def test_order_sheet_carries_floors_only_never_the_class_debt():
    cells = (_cell("X", predicts=("dense",)), _cell("Y", predicts=("sparse",)))
    floors = DiversityFloors(RECIPE, cells)
    sheet = floors.order_sheet(POOL.head(40), POOL)
    lines = sheet.set_index("floor")
    # 40 selected rows hold X, so its floor is met and it gets no line at all;
    # dense's 370-row class debt is the labelling rung's, not the sheet's
    assert "X" not in lines.index
    assert set(sheet["slice"]) == {"cell", "corruption"}
    # Y has no supply anywhere: the floor is the whole demand, credit 0 at build
    assert lines.at["Y", "amount"] == 25.0
    assert lines.at["Y", "credit"] == 0.0
    assert lines.at["Y", "missing"] == 25.0
    assert lines.at["Y", "reason"] == "exhausted"


def test_cell_floor_nets_the_unspent_supply_that_already_satisfies_it():
    pool = _frame([
        {"dataset": "a", "query_id": str(i), "cells": {"Z"},
         "corruption_degree": "clean", "route_class_any": "dense",
         "is_waste": False}
        for i in range(30)
    ])
    floors = DiversityFloors(
        Recipe.v3(stratum_floor=25, target_total=0), (_cell("Z"),)
    )
    # 10 selected + 20 unspent rows already satisfy Z, so generation owes
    # nothing — a line here is the NeedsSelection the loop cannot pay
    assert "Z" not in set(floors.order_sheet(pool.head(10), pool)["floor"])
    # the same rows as waste are not supply, so the shortfall is real again
    wasted = pool.assign(is_waste=pool.index >= 10)
    sheet = floors.order_sheet(pool.head(10), wasted).set_index("floor")
    assert sheet.at["Z", "missing"] == 15.0


def test_marginals_floor_declared_bands_and_count_the_dark_apart():
    selected = _frame([
        {"dataset": "a", "query_id": str(i), "cells": {"X"},
         "corruption_degree": "clean" if i < 30 else UNKNOWN,
         "corpus_idf": "low_idf", "corpus_oov": UNKNOWN,
         "corpus_pmi": "co_occurring"}
        for i in range(40)
    ])
    out = DiversityFloors(RECIPE, (_cell("X"),)).marginals(selected)
    axes = out.set_index(["axis", "stratum"])
    # a band no row landed in is a line that FAILS, not a missing line
    assert axes.at[("corruption_degree", "heavy"), "rows"] == 0
    assert not axes.at[("corruption_degree", "heavy"), "met"]
    assert axes.at[("corruption_degree", "clean"), "met"]
    # 'unknown' is coverage, not a band: counted per axis, never floored
    assert set(out.loc[out["axis"] == "corpus_oov", "stratum"]) == {
        "in_vocab", "has_oov",
    }
    assert not axes.at[("corpus_oov", "in_vocab"), "met"]
    assert axes.at[("corpus_oov", "in_vocab"), "dark"] == 40
    assert axes.at[("corruption_degree", "clean"), "dark"] == 10
    # the lane floor lives in per_dataset alone — one axis, one structure
    assert "dataset" not in set(out["axis"])


def test_corruption_line_scales_to_target_total():
    floors = DiversityFloors(RECIPE, ())
    sheet = floors.order_sheet(POOL.head(40), POOL).set_index("floor")
    census = pd.read_parquet(AugmentationConfig().paths.corruption_census)
    rate = (census["n_sampled"] * census["any_span"]).sum() / census["n_sampled"].sum()
    # the target is rate x target_total net of the pool's damaged rows (0),
    # never rate x the selection
    assert sheet.at["corruption:light", "missing"] == round(rate * 1_000)


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
    # every line the sheet writes is payable: a cell demand and the corruption
    # demand, nothing decorative
    assert set(hungry["floor"]) == {"Y", "corruption:light"}
    readout = loop.hungry()
    # corruption:light must be SERVABLE by a registered operator today — the
    # "no writer emits corruption floors" gap closing — while staying behind
    # its declared audit gate (d42h), which is what runnable=False means here
    corr = readout[readout["floor"] == "corruption:light"].iloc[0]
    assert corr["operator"] == "corrupt"
    assert corr["gate"] == "declaration_audit" and not corr["runnable"]
