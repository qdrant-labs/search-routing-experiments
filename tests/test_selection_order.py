"""The SelectionOrder rung: measured yields, debt allocation under caps,
answer-coverage-gated picks, and the order's round-trip into the labelling
sweep's selection builder."""

from types import SimpleNamespace

import pandas as pd
import pytest

from composition.objectives import CLASS_ALLOCATION, SelectionOrder
from composition.pool_v3 import REUSED
from composition.recipe import Recipe

RECIPE = Recipe.v3(target_total=1_000)


def _debt(rows):
    return pd.DataFrame(
        rows, columns=["route_class", "target", "supply", "debt", "x_under"]
    )


def test_yields_read_blind_nonwaste_rows_only():
    pool = pd.DataFrame({
        "dataset": ["a"] * 10 + ["a"] * 5,
        "stage": ["candidate"] * 10 + [REUSED] * 5,
        "route_class_any": ["sparse"] * 4 + ["dense"] * 2 + [""] * 4
        + ["sparse"] * 5,
        "is_waste": [False, False, False, True] + [False] * 6 + [False] * 5,
    })
    yields = SelectionOrder.yields(pool)
    assert yields.at["a", "blind_labelled"] == 10
    # one sparse hit is waste and five more are reused — neither counts
    assert yields.at["a", "yield_sparse"] == pytest.approx(0.3)
    assert yields.at["a", "yield_dense"] == pytest.approx(0.2)


def test_allocation_is_yield_greedy_and_cap_capped():
    order = SelectionOrder(RECIPE, (), data_dir=None)
    debt = _debt([
        ("dense", 450, 80, 370, 5.6), ("sparse", 450, 20, 430, 22.5),
        ("hybrid", 100, 0, 100, float("inf")),
    ])
    yields = pd.DataFrame({
        "blind_labelled": [100, 100],
        "yield_dense": [0.1, 0.0],
        "yield_sparse": [0.5, 0.1],
        "yield_hybrid": [0.0, 0.0],
    }, index=pd.Index(["hi", "lo"], name="dataset"))
    per_ds = pd.DataFrame({"fresh_left": [400, 10_000]}, index=yields.index)
    pool = pd.DataFrame({
        "dataset": pd.Series(dtype=str),
        "route_class_any": pd.Series(dtype=str),
        "is_waste": pd.Series(dtype=bool),
    })

    allocation = order.allocate(debt, yields, per_ds, pool)

    # sparse first (biggest debt), best yield first; each lane stops at its
    # 20% share cap (90 sparse rows), the second pass buys dense from hi's
    # remaining fresh room
    assert allocation.at["hi", "labels_ordered"] == 400
    assert allocation.at["lo", "labels_ordered"] == 900
    assert allocation.attrs["residual_debt"] == {
        "dense": 330, "sparse": 250, "hybrid": 100,
    }


def test_picks_gate_on_labelled_checkable_and_answer_coverage(tmp_path):
    (tmp_path / "a").mkdir()
    pd.DataFrame({
        "query_id": ["q1", "q2", "q3", "q4"],
        "doc_id": ["d1", "d2", "d3", "gone"],
        "relevance": [1, 1, 1, 1],
    }).to_parquet(tmp_path / "a" / "qrels.parquet", index=False)
    pd.DataFrame({"doc_id": ["d1", "d2", "d3"]}).to_parquet(
        tmp_path / "a" / "corpus.parquet", index=False
    )
    pd.DataFrame({"query_id": [f"q{i}" for i in range(1, 6)]}).to_parquet(
        tmp_path / "a" / "queries.parquet", index=False
    )
    cell = SimpleNamespace(name="hx", select=lambda frame: frame["x"] >= 5)
    order = SelectionOrder(RECIPE, (cell,), data_dir=tmp_path)
    # the catalog carries features for q1 only — cell targeting reaches
    # exactly the rows the catalog knows, the rest fill at natural rates
    catalog = pd.DataFrame({"dataset": ["a"], "query_id": ["q1"], "x": [5]})
    pool = pd.DataFrame({
        "dataset": ["a"], "query_id": ["q5"], "min_relevance": [1],
    })
    allocation = pd.DataFrame(
        {"labels_ordered": [3]}, index=pd.Index(["a"], name="dataset")
    )
    sheet = pd.DataFrame({"slice": ["cell"], "floor": ["hx"], "missing": [9.0]})

    picks = order.picks(allocation, pool, catalog, sheet)

    # q1 leads as the hungry cell's member; q4 (answer doc missing from the
    # corpus) and q5 (already labelled) never appear
    assert picks["query_id"].iloc[0] == "q1"
    assert set(picks["query_id"]) == {"q1", "q2", "q3"}
    assert picks["reason"].iloc[0] == "cell:hx"
    assert set(picks["reason"].iloc[1:]) == {CLASS_ALLOCATION}


def test_build_appends_the_residual_footer(tmp_path):
    order = SelectionOrder(RECIPE, (), data_dir=tmp_path)
    pool = pd.DataFrame({
        "dataset": ["a"], "query_id": ["q0"], "stage": ["candidate"],
        "route_class_any": ["dense"], "is_waste": [False],
        "min_relevance": [1],
    })
    debt = _debt([
        ("dense", 450, 1, 449, 450.0), ("sparse", 450, 0, 450, float("inf")),
        ("hybrid", 100, 0, 100, float("inf")),
    ])
    per_ds = pd.DataFrame({"fresh_left": [0]}, index=pd.Index(["a"], name="dataset"))
    sheet = pd.DataFrame({"slice": [], "floor": [], "missing": []})
    catalog = pd.DataFrame({"dataset": [], "query_id": [], "checkable": []})

    picks, summary = order.build(pool, per_ds, debt, sheet, catalog)

    assert picks.empty
    footer = summary.loc["(residual after labelling)"]
    assert footer["expected_sparse"] == 450  # nothing labellable -> full debt


def test_v2_origin_rows_are_measurement_never_supply(tmp_path):
    from composition.objectives import DiversityFloors
    from composition.pool_v3 import LabelledPool

    (tmp_path / "v3").mkdir()
    (tmp_path / "v3" / "augmented").mkdir()
    pd.DataFrame({
        "dataset": ["a"], "query_id": ["old"], "checkable": [True],
    }).to_parquet(tmp_path / "v3" / "labels_rederived.parquet", index=False)
    pd.DataFrame({
        "dataset": ["a"], "query_id": ["new"], "checkable": [True],
    }).to_parquet(tmp_path / "v3" / "labels.parquet", index=False)
    pd.DataFrame({
        "dataset": ["a"], "query_id": ["aug"], "checkable": [True],
    }).to_parquet(tmp_path / "v3" / "augmented" / "labels.parquet", index=False)
    labels = LabelledPool(data_dir=tmp_path).labels().set_index("query_id")
    assert labels["native"].to_dict() == {"old": False, "new": True, "aug": True}
    # the augmentation rung's rows are scored against a supplemented corpus
    assert labels.at["aug", "scored_against"] == "supplemented"

    pool = pd.DataFrame({
        "dataset": ["a"] * 6,
        "route_class_any": ["dense"] * 6,
        "is_waste": [False] * 6,
        "native": [True] * 2 + [False] * 4,
    })
    debt = DiversityFloors(RECIPE, ()).class_debt(pool).set_index("route_class")
    assert debt.at["dense", "supply"] == 2  # the four v2-origin rows measure nothing


def test_order_selection_joins_text_dedups_and_caps(tmp_path, monkeypatch):
    import scripts.label_routes_v3 as mod

    (tmp_path / "v3").mkdir()
    (tmp_path / "a").mkdir()
    pd.DataFrame({
        "dataset": ["a"] * 4, "query_id": ["q1", "q2", "q3", "q4"],
        "reason": [CLASS_ALLOCATION] * 4,
    }).to_parquet(tmp_path / "v3" / "selection_order.parquet", index=False)
    pd.DataFrame({
        "query_id": ["q1", "q2", "q3", "q4"],
        "text": ["one", "two", "three", "four"],
    }).to_parquet(tmp_path / "a" / "queries.parquet", index=False)
    pd.DataFrame({"dataset": ["a"], "query_id": ["q1"]}).to_parquet(
        tmp_path / "labels_v2.parquet", index=False
    )
    monkeypatch.setattr(mod, "DATA_DIR", tmp_path)
    monkeypatch.setattr(mod, "V3_DIR", tmp_path / "v3")
    monkeypatch.setattr(mod, "V2_LABELS", tmp_path / "labels_v2.parquet")

    selection = mod.order_selection(cap=2)

    assert list(selection["query_id"]) == ["q2", "q3"]  # q1 already labelled
    assert list(selection["query"]) == ["two", "three"]
    assert set(selection["home_lane"]) == {"a"}
