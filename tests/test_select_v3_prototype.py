"""Core-logic checks for the v3 selector prototype: the pure class-assignment
policy, the closed-form feasibility bound, and the corpus stratum axes."""

import numpy as np
import pandas as pd

from scripts import select_v3_prototype as sel
from scripts.select_v3_prototype import (
    CLASSES,
    SelectorRecipe,
    assign_classes,
    feasible_total,
)

R = SelectorRecipe()


def _classes(scores, depth):
    """assign_classes over a list of 3-score rows (dense, rrf, sparse order)."""
    arr = np.array(scores, dtype=float)
    ordered = np.sort(arr, axis=1)
    winner = np.array(["dense_only", "pure_rrf", "sparse_only"])[arr.argmax(axis=1)]
    return assign_classes(
        ordered[:, -1], ordered[:, -2], ordered[:, 0],
        winner, np.array(depth, dtype=float), R,
    )


def test_decisive_routes_map_to_classes():
    # dense clear win, sparse clear win, rrf clear win (margin >= 0.4)
    kind, cls = _classes(
        [[1.0, 0.3, 0.2], [0.2, 0.3, 1.0], [0.3, 1.0, 0.2]],
        depth=[5, 5, 5],
    )
    assert list(kind) == ["decisive", "decisive", "decisive"]
    assert list(cls) == ["dense", "sparse", "hybrid"]


def test_genuine_tie_is_hybrid_fake_tie_is_waste():
    # both tie at ceiling; deep qrels -> genuine (hybrid), single doc -> fake (waste)
    kind, cls = _classes([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]], depth=[3, 1])
    assert list(kind) == ["genuine_tie", "fake_tie"]
    assert list(cls) == ["hybrid", ""]


def test_all_zero_and_undecisive():
    # nothing retrieved; and a routes_differ below the decisive margin
    kind, cls = _classes([[0.0, 0.0, 0.0], [0.5, 0.3, 0.4]], depth=[0, 5])
    assert kind[0] == "all_zero" and cls[0] == ""
    assert kind[1] == "undecisive" and cls[1] == ""


def _corpus_strata(monkeypatch, tmp_path, qcs: pd.DataFrame, query_ids):
    """_attach_corpus_strata over a synthetic query_corpus_stats file."""
    (tmp_path / "route_labels").mkdir(exist_ok=True)
    qcs.to_parquet(tmp_path / "route_labels" / "query_corpus_stats.parquet")
    monkeypatch.setattr(sel, "DATA", tmp_path)
    pool = pd.DataFrame({"dataset": ["d"] * len(query_ids), "query_id": query_ids})
    return sel._attach_corpus_strata(pool)


def test_pmi_sentinel_nan_and_absent_stay_separate(monkeypatch, tmp_path):
    qcs = pd.DataFrame({
        "dataset": ["d"] * 5,
        "query_id": ["1", "2", "3", "4", "5"],
        "avg_idf": [0.1, 0.2, 0.5, 0.8, 0.9],
        "oov_share": [0.0, 0.5, 0.0, 0.0, 0.0],
        # -1.0 is the "never co-occurs" sentinel; NaN is unmeasurable
        "min_pmi": [-1.0, 0.3, np.nan, -0.5, -1.0],
    })
    out = _corpus_strata(monkeypatch, tmp_path, qcs, ["1", "2", "3", "4", "5", "99"])
    assert list(out["corpus_pmi"]) == [
        "never_co_occurs", "co_occurring", "unmeasured",
        "co_occurring", "never_co_occurs", "unknown",
    ]
    assert list(out["corpus_oov"]) == [
        "in_vocab", "has_oov", "in_vocab", "in_vocab", "in_vocab", "unknown",
    ]
    # quartiles of the file: p25 0.2, p75 0.8
    assert list(out["corpus_idf"]) == [
        "low_idf", "mid_idf", "mid_idf", "high_idf", "high_idf", "unknown",
    ]


def test_idf_edges_are_measured_not_hardcoded(monkeypatch, tmp_path):
    # every value sits above the production p75 (~0.50), so hardcoded edges
    # would collapse the axis to one band; the file's own quartiles split it
    qcs = pd.DataFrame({
        "dataset": ["d"] * 4,
        "query_id": ["1", "2", "3", "4"],
        "avg_idf": [5.0, 6.0, 7.0, 8.0],
        "oov_share": [0.0] * 4,
        "min_pmi": [0.0] * 4,
    })
    out = _corpus_strata(monkeypatch, tmp_path, qcs, ["1", "2", "3", "4"])
    assert list(out["corpus_idf"]) == ["low_idf", "mid_idf", "mid_idf", "high_idf"]


def _counts(**lanes_per_class):
    return {c: pd.Series(v) for c, v in lanes_per_class.items()}


def test_feasible_total_uncapped_matches_the_closed_form():
    # sparse is scarcest relative to its share -> it binds; cap=1 = closed form
    lane_counts = _counts(
        dense={"a": 3149}, sparse={"a": 1731}, hybrid={"a": 1702},
    )
    total = feasible_total(lane_counts, (0.45, 0.45, 0.10), lane_share_cap=1.0)
    assert total == int(1731 / 0.45)
    assert all(round(total * s) <= int(lane_counts[c].sum())
               for c, s in zip(CLASSES, (0.45, 0.45, 0.10)))


def test_feasible_total_shrinks_when_one_lane_holds_the_supply():
    # sparse supply is huge but 90% of it sits in one lane; a 0.2 cap must
    # bind the total far below the closed form
    lane_counts = _counts(
        dense={"a": 500, "b": 500}, hybrid={"a": 200, "b": 200},
        sparse={"big": 900, "s1": 25, "s2": 25, "s3": 25, "s4": 25},
    )
    uncapped = feasible_total(lane_counts, (0.45, 0.45, 0.10), 1.0)
    capped = feasible_total(lane_counts, (0.45, 0.45, 0.10), 0.2)
    assert capped < uncapped
    # and the capped total is actually drawable: no class needs more than its
    # lanes can give under the cap
    for c, s in zip(CLASSES, (0.45, 0.45, 0.10)):
        target = round(capped * s)
        per_lane = max(1, int(np.ceil(0.2 * target)))
        assert np.minimum(lane_counts[c], per_lane).sum() >= target


def _pool(rows):
    frame = pd.DataFrame(rows)
    frame["cells"] = frame["cells"].map(frozenset)
    defaults = {
        "corpus_idf": "unknown", "corpus_oov": "unknown", "corpus_pmi": "unknown",
        "corruption_degree": "clean", "is_waste": False,
    }
    for column, value in defaults.items():
        if column not in frame.columns:
            frame[column] = value
    frame["is_waste"] = frame["is_waste"].fillna(False).astype(bool)
    if "route_class_any" not in frame.columns:
        frame["route_class_any"] = frame["route_class"]
    frame["route_class_any"] = frame["route_class_any"].fillna(frame["route_class"])
    if "certified" not in frame.columns:
        frame["certified"] = frame["route_class"] != ""
    return frame


def test_greedy_recomputes_gain_per_pick():
    # rows 1 and 2 both cover cell X (2-cell rows); row 3 covers only Y.
    # A frozen ordering ranks {1, 2} first and never reaches Y within a
    # 2-row target; a true greedy takes one X-row then Y.
    pool = _pool([
        {"query_id": "1", "dataset": "a", "route_class": "dense", "cells": {"X", "Z"}},
        {"query_id": "2", "dataset": "b", "route_class": "dense", "cells": {"X", "Z"}},
        {"query_id": "3", "dataset": "c", "route_class": "dense", "cells": {"Y"}},
    ])
    recipe = SelectorRecipe(target_split=(1.0, 0.0, 0.0), lane_share_cap=1.0)
    picked = sel.greedy_select(pool, recipe)
    # target = feasible total = 3 here, so instead pin the coverage ORDER:
    covered_first_two = set().union(
        *pool.set_index("query_id").loc[picked["query_id"].iloc[:2], "cells"]
    )
    assert {"X", "Y"} <= covered_first_two


def test_lane_cap_is_exact_no_escape_row():
    # 10 rows in one lane, cap 0.5 over a target of 4 -> exactly 2 from it
    pool = _pool(
        [{"query_id": str(i), "dataset": "solo", "route_class": "dense",
          "cells": set()} for i in range(10)]
        + [{"query_id": f"o{i}", "dataset": f"lane{i}", "route_class": "dense",
            "cells": set()} for i in range(2)]
    )
    recipe = SelectorRecipe(target_split=(1.0, 0.0, 0.0), lane_share_cap=0.5)
    picked = sel.greedy_select(pool, recipe)
    counts = picked["dataset"].value_counts()
    target = len(picked)
    assert counts.max() <= max(1, int(np.ceil(0.5 * target)))


def test_waste_draw_is_budgeted_and_typed():
    pool = _pool(
        [{"query_id": str(i), "dataset": f"l{i % 3}", "route_class": "dense",
          "cells": set()} for i in range(9)]
        + [{"query_id": f"w{i}", "dataset": f"l{i % 3}", "route_class": "",
            "cells": set(), "is_waste": True} for i in range(9)]
    )
    recipe = SelectorRecipe(
        target_split=(1.0, 0.0, 0.0), lane_share_cap=1.0, waste_cap=0.5,
    )
    picked = sel.greedy_select(pool, recipe)
    waste = picked[picked["route_class"] == "waste"]
    assert len(waste) == picked.attrs["waste_budget"] > 0
    assert set(waste["query_id"]).issubset({f"w{i}" for i in range(9)})


def test_zero_waste_cap_draws_no_waste():
    pool = _pool(
        [{"query_id": "1", "dataset": "a", "route_class": "dense", "cells": set()},
         {"query_id": "w", "dataset": "a", "route_class": "", "cells": set(),
          "is_waste": True}]
    )
    picked = sel.greedy_select(
        pool, SelectorRecipe(target_split=(1.0, 0.0, 0.0), lane_share_cap=1.0)
    )
    assert not (picked["route_class"] == "waste").any()


def test_shallow_tie_is_fake_at_any_score():
    # sub-ceiling depth-1 tie: all routes missed the one judged doc equally —
    # the same qrels artifact as the ceiling tie, and now filed the same way
    kind, cls = _classes([[0.3, 0.3, 0.3], [0.3, 0.3, 0.3]], depth=[1, 3])
    assert list(kind) == ["fake_tie", "genuine_tie"]
    assert list(cls) == ["", "hybrid"]


def test_two_tier_nesting_and_per_tier_splits():
    # 6 certified dense + 6 uncertified (margin-0 only) dense, one lane each;
    # sparse/hybrid shares zero so the tier totals are driven by dense alone
    rows = (
        [{"query_id": f"c{i}", "dataset": f"l{i}", "route_class": "dense",
          "cells": set()} for i in range(6)]
        + [{"query_id": f"u{i}", "dataset": f"m{i}", "route_class": "",
            "route_class_any": "dense", "cells": set()} for i in range(6)]
    )
    pool = _pool(rows)
    pool.loc[pool["route_class"] == "", "certified"] = False
    picked = sel.greedy_select(
        pool, SelectorRecipe(target_split=(1.0, 0.0, 0.0), lane_share_cap=1.0,
                             waste_cap=0.0),
    )
    cert = picked[picked["certified"]]
    top_up = picked[~picked["certified"]]
    assert len(cert) == 6 and set(cert["query_id"]) == {f"c{i}" for i in range(6)}
    assert len(top_up) == 6  # tier-0 total 12, minus the certified 6
    assert (picked["route_class"] == "dense").all()


def test_top_up_never_leaks_an_undrawn_certified_row():
    # dense: 10 certified rows in one lane, but sparse's thin certified supply
    # (2) caps total_c at 4 -> dense's certified target is only 2, leaving 8
    # certified dense rows undrawn. dense's tier-0 target (8) then exceeds
    # those 2, so top-up goes looking for 6 more dense rows; only 3 truly
    # uncertified ones exist. The pre-fix top-up filter
    # (~index.isin(certified_taken)) would have happily grafted 3 of the 8
    # undrawn CERTIFIED rows in to make up the shortfall.
    rows = (
        [{"query_id": f"cd{i}", "dataset": "d0", "route_class": "dense",
          "cells": set()} for i in range(10)]
        + [{"query_id": f"ud{i}", "dataset": "d0", "route_class": "",
            "route_class_any": "dense", "cells": set()} for i in range(3)]
        + [{"query_id": f"cs{i}", "dataset": "s0", "route_class": "sparse",
            "cells": set()} for i in range(2)]
        + [{"query_id": f"us{i}", "dataset": "s0", "route_class": "",
            "route_class_any": "sparse", "cells": set()} for i in range(6)]
    )
    pool = _pool(rows)
    picked = sel.greedy_select(
        pool, SelectorRecipe(target_split=(0.5, 0.5, 0.0), lane_share_cap=1.0,
                             waste_cap=0.0),
    )
    top_up_dense = picked[(~picked["certified"]) & (picked["route_class"] == "dense")]
    assert len(top_up_dense) == 3  # true uncertified supply, not the target of 6
    assert set(top_up_dense["query_id"]) == {f"ud{i}" for i in range(3)}


def test_eval_reserve_is_certified_only_stratified_and_disjoint():
    rows = (
        [{"query_id": f"a{i}", "dataset": "laneA", "route_class": "dense",
          "cells": set()} for i in range(10)]
        + [{"query_id": f"b{i}", "dataset": "laneB", "route_class": "sparse",
            "cells": set()} for i in range(10)]
        + [{"query_id": f"w{i}", "dataset": "laneA", "route_class": "",
            "route_class_any": "", "cells": set()} for i in range(5)]
    )
    pool = _pool(rows)
    reserve = sel.eval_reserve(pool, SelectorRecipe(eval_reserve_frac=0.2))
    assert len(reserve) == 4  # 20% of each 10-row (lane, class) stratum
    assert reserve["certified"].all()
    remaining = pool.drop(index=reserve.index)
    picked = sel.greedy_select(
        remaining,
        SelectorRecipe(target_split=(0.5, 0.5, 0.0), lane_share_cap=1.0,
                       waste_cap=0.0),
    )
    assert not set(picked.index) & set(reserve.index)


def test_waste_default_is_five_percent():
    assert SelectorRecipe().waste_cap == 0.05
