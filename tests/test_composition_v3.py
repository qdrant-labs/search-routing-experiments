"""Core-logic checks for the v3 composition: the pure class-assignment policy,
the capped feasibility bound, the two-tier utility draw, and the corpus
stratum axes."""

import numpy as np
import pandas as pd
import pytest

from composition.objectives import UtilityObjective
from composition.pool_v3 import (
    CLASSES,
    STRATA,
    UNKNOWN,
    LabelledPool,
    assign_classes,
)
from composition.recipe import Recipe

R = Recipe()


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
    # 0.03 margin: clearly under the 0.1 certification bar, not float-luck
    kind, cls = _classes([[0.0, 0.0, 0.0], [0.5, 0.3, 0.47]], depth=[0, 5])
    assert kind[0] == "all_zero" and cls[0] == ""
    assert kind[1] == "undecisive" and cls[1] == ""


def test_shallow_tie_is_fake_at_any_score():
    # sub-ceiling depth-1 tie: all routes missed the one judged doc equally —
    # the same qrels artifact as the ceiling tie, and now filed the same way
    kind, cls = _classes([[0.3, 0.3, 0.3], [0.3, 0.3, 0.3]], depth=[1, 3])
    assert list(kind) == ["fake_tie", "genuine_tie"]
    assert list(cls) == ["", "hybrid"]


def _corpus_strata(tmp_path, qcs: pd.DataFrame, query_ids):
    """_attach_corpus_strata over a synthetic query_corpus_stats file."""
    (tmp_path / "route_labels").mkdir(exist_ok=True)
    qcs.to_parquet(tmp_path / "route_labels" / "query_corpus_stats.parquet")
    pool = pd.DataFrame({"dataset": ["d"] * len(query_ids), "query_id": query_ids})
    return LabelledPool(data_dir=tmp_path)._attach_corpus_strata(pool)


def test_pmi_sentinel_nan_and_absent_stay_separate(tmp_path):
    qcs = pd.DataFrame({
        "dataset": ["d"] * 5,
        "query_id": ["1", "2", "3", "4", "5"],
        "avg_idf": [0.1, 0.2, 0.5, 0.8, 0.9],
        "oov_share": [0.0, 0.5, 0.0, 0.0, 0.0],
        # -1.0 is the "never co-occurs" sentinel; NaN is unmeasurable
        "min_pmi": [-1.0, 0.3, np.nan, -0.5, -1.0],
    })
    out = _corpus_strata(tmp_path, qcs, ["1", "2", "3", "4", "5", "99"])
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
    # this fixture exercises every band each axis declares, so equality (not
    # subset) is what keeps the vocabulary the floors read from drifting
    for axis, bands in STRATA.items():
        if axis in out.columns:
            assert set(out[axis]) == {*bands, UNKNOWN}


def test_corpus_join_that_reaches_no_supply_row_raises(tmp_path):
    qcs = pd.DataFrame({
        "dataset": ["d"], "query_id": ["1"], "avg_idf": [0.5],
        "oov_share": [0.0], "min_pmi": [0.0],
    })
    # a partial miss is legitimately unknown (the test above); a total miss is
    # a broken join that used to write 'unknown' over every v3 row in silence
    with pytest.raises(ValueError, match="covers 0 of"):
        _corpus_strata(tmp_path, qcs, ["98", "99"])


def test_idf_edges_are_measured_not_hardcoded(tmp_path):
    # every value sits above the production p75 (~0.50), so hardcoded edges
    # would collapse the axis to one band; the file's own quartiles split it
    qcs = pd.DataFrame({
        "dataset": ["d"] * 4,
        "query_id": ["1", "2", "3", "4"],
        "avg_idf": [5.0, 6.0, 7.0, 8.0],
        "oov_share": [0.0] * 4,
        "min_pmi": [0.0] * 4,
    })
    out = _corpus_strata(tmp_path, qcs, ["1", "2", "3", "4"])
    assert list(out["corpus_idf"]) == ["low_idf", "mid_idf", "mid_idf", "high_idf"]


def _counts(**lanes_per_class):
    return {c: pd.Series(v) for c, v in lanes_per_class.items()}


def test_feasible_total_uncapped_matches_the_closed_form():
    # sparse is scarcest relative to its share -> it binds; cap=1 = closed form
    lane_counts = _counts(
        dense={"a": 3149}, sparse={"a": 1731}, hybrid={"a": 1702},
    )
    total = UtilityObjective.feasible_total(
        lane_counts, (0.45, 0.45, 0.10), lane_share_cap=1.0
    )
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
    uncapped = UtilityObjective.feasible_total(lane_counts, (0.45, 0.45, 0.10), 1.0)
    capped = UtilityObjective.feasible_total(lane_counts, (0.45, 0.45, 0.10), 0.2)
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


def _select(pool, **recipe_kw):
    return UtilityObjective(Recipe(**recipe_kw)).select(pool)


def test_greedy_recomputes_gain_per_pick():
    # rows 1 and 2 both cover cell X (2-cell rows); row 3 covers only Y.
    # A frozen ordering ranks {1, 2} first and never reaches Y within a
    # 2-row target; a true greedy takes one X-row then Y.
    pool = _pool([
        {"query_id": "1", "dataset": "a", "route_class": "dense", "cells": {"X", "Z"}},
        {"query_id": "2", "dataset": "b", "route_class": "dense", "cells": {"X", "Z"}},
        {"query_id": "3", "dataset": "c", "route_class": "dense", "cells": {"Y"}},
    ])
    picked = _select(pool, target_split=(1.0, 0.0, 0.0), target_lane_share=1.0)
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
    picked = _select(pool, target_split=(1.0, 0.0, 0.0), target_lane_share=0.5)
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
    picked = _select(
        pool, target_split=(1.0, 0.0, 0.0), target_lane_share=1.0, waste_cap=0.5,
    )
    waste = picked[picked["route_class"] == "waste"]
    assert len(waste) == picked.attrs["waste_budget"] > 0
    assert set(waste["query_id"]).issubset({f"w{i}" for i in range(9)})


def test_zero_waste_cap_draws_no_waste():
    pool = _pool(
        [{"query_id": "1", "dataset": "a", "route_class": "dense", "cells": set()},
         {"query_id": "w", "dataset": "a", "route_class": "", "cells": set(),
          "is_waste": True}]
    )
    picked = _select(
        pool, target_split=(1.0, 0.0, 0.0), target_lane_share=1.0, waste_cap=0.0,
    )
    assert not (picked["route_class"] == "waste").any()


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
    picked = _select(
        pool, target_split=(1.0, 0.0, 0.0), target_lane_share=1.0, waste_cap=0.0,
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
    picked = _select(
        pool, target_split=(0.5, 0.5, 0.0), target_lane_share=1.0, waste_cap=0.0,
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
    reserve = LabelledPool(Recipe(eval_reserve_frac=0.2)).eval_reserve(pool)
    assert len(reserve) == 4  # 20% of each 10-row (lane, class) stratum
    assert reserve["certified"].all()
    remaining = pool.drop(index=reserve.index)
    picked = _select(
        remaining, target_split=(0.5, 0.5, 0.0), target_lane_share=1.0,
        waste_cap=0.0,
    )
    assert not set(picked.index) & set(reserve.index)


def test_waste_default_is_five_percent():
    assert Recipe().waste_cap == 0.05


# --------------------------------------------------------------- admit door ---
def _gate_pool(gates):
    """One generated pool row per credit_gate value, all on one hungry floor."""
    return pd.DataFrame({
        "query_id": [f"g{i}" for i in range(len(gates))],
        "floor": ["cellA"] * len(gates),
        "home_lane": ["a"] * len(gates),
        "credit_gate": list(gates),
    })


def _hungry_sheet(**missing_per_floor):
    floors = list(missing_per_floor)
    return pd.DataFrame({
        "slice": ["cell"] * len(floors),
        "floor": floors,
        "amount": [float(v) for v in missing_per_floor.values()],
        "credit": [0.0] * len(floors),
        "missing": [float(v) for v in missing_per_floor.values()],
        "reason": ["exhausted"] * len(floors),
    })


def test_each_gate_clears_only_from_its_own_list():
    from composition.composer import V3Composition

    pool = _gate_pool(["none", "coherence_gate", "declaration_audit"])
    sheet = _hungry_sheet(cellA=10)
    empty = pd.DataFrame({"query_id": []})
    waiting = V3Composition._admissible(pool, empty, sheet)
    assert list(waiting["query_id"]) == ["g0"]  # both gates wait by default
    audited = V3Composition._admissible(pool, empty, sheet, None, {"g2"})
    assert list(audited["query_id"]) == ["g0", "g2"]
    # a coherence verdict may never clear the human's audit, or the reverse
    crossed = V3Composition._admissible(pool, empty, sheet, {"g2"}, {"g1"})
    assert list(crossed["query_id"]) == ["g0"]
    both = V3Composition._admissible(pool, empty, sheet, {"g1"}, {"g2"})
    assert list(both["query_id"]) == ["g0", "g1", "g2"]


class _Claims:
    """Cell double: claims the query_ids it was built with."""

    def __init__(self, *query_ids):
        self._ids = set(query_ids)

    def select(self, frame):
        return frame["query_id"].isin(self._ids)


def test_extra_credit_bills_every_line_a_row_serves():
    from composition.composer import V3Composition

    # both rows were minted for A; q1 also satisfies B (2 owed) and C (1 owed)
    mini = pd.DataFrame({"query_id": ["q0", "q1"]}, index=[0, 1])
    rows = pd.DataFrame(
        {"query_id": ["q0", "q1"], "credited_floor": ["A", "A"]}, index=[0, 1]
    )
    cells = {
        "A": _Claims("q0", "q1"), "B": _Claims("q1"), "C": _Claims("q0", "q1"),
    }
    extra = V3Composition._extra_credit(
        _hungry_sheet(A=2, B=2, C=1), rows, {"A": 2}, cells, mini
    )
    # A is already fully credited by the per-line pass; C is capped at its need
    assert extra == {"B": 1, "C": 1}


def test_natural_share_is_cumulative_and_counts_no_row_twice():
    from composition.composer import V3Composition

    selection = pd.DataFrame({
        "query_id": [f"n{i}" for i in range(9)] + ["g0"],
        "provenance": ["natural"] * 9 + ["augmented"],
    })
    # g0 already sits in the selection AND on the admission record: 9/10, not
    # 9/11, or the report would double-charge the row it already counted
    share = V3Composition._natural_share(
        selection, pd.DataFrame({"query_id": ["g0"]})
    )
    assert share == pytest.approx(9 / 10)
    # one more generated row dilutes to 9/11 — reported, never asserted:
    # the v3 dataset is generation-fed by decision
    diluted = V3Composition._natural_share(
        selection, pd.DataFrame({"query_id": ["g0", "g1"]})
    )
    assert diluted == pytest.approx(9 / 11)


def test_admission_record_appends_and_keeps_the_newest_verdict(tmp_path):
    from composition.composer import V3Composition

    path = tmp_path / "admitted.parquet"
    first = pd.DataFrame({"query_id": ["a", "b"], "credited_floor": ["X", "X"]})
    V3Composition._admission_record(path, first).to_parquet(path, index=False)
    second = pd.DataFrame({"query_id": ["b", "c"], "credited_floor": ["Y", "Y"]})
    merged = V3Composition._admission_record(path, second)
    assert list(merged["query_id"]) == ["a", "b", "c"]
    assert merged.set_index("query_id").at["b", "credited_floor"] == "Y"
