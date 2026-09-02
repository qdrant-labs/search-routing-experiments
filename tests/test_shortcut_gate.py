"""Checks for the gate rung: the lineage key a split groups on, the reserve
fence that now honours it, and the nuisance matrix's refusal to see anything
retrieval measured."""

from itertools import cycle, islice

import pandas as pd
import pytest

from composition.pool_v3 import LabelledPool
from composition.recipe import Recipe
from scripts.shortcut_gate import BANDS, MEASURED, NuisanceMatrix, ShortcutGate


def _lineage(tmp_path, rows):
    """The augmentation pool where `family_ids` looks for `generated_from`."""
    path = tmp_path / "augmentation" / "pool.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        rows,
        columns=["query_id", "generated_from", "parent_dataset", "home_lane"],
    ).to_parquet(path)
    return path


def _pool(tmp_path) -> LabelledPool:
    return LabelledPool(Recipe.v3(), data_dir=tmp_path)


def test_child_and_parent_share_one_family(tmp_path):
    path = _lineage(tmp_path, [("aug-1", "7", "quest", "quest")])
    selected = pd.DataFrame(
        {"dataset": ["quest", "quest"], "query_id": ["7", "aug-1"]}
    )
    fam = _pool(tmp_path).family_ids(selected, pool_path=path)
    assert fam.iloc[0] == fam.iloc[1] == "quest:7"


def test_a_child_minted_into_another_lane_still_keys_on_its_parent(tmp_path):
    """`generated_from` is `7`, not `quest:7` — unqualified it never matches the
    parent's own key and the pair straddles a split anyway. The child here lives
    in `clerc` while its parent is in `quest`, so both keys must be qualified."""
    path = _lineage(tmp_path, [("aug-1", "7", "quest", "clerc")])
    selected = pd.DataFrame({"dataset": ["clerc"], "query_id": ["aug-1"]})
    assert _pool(tmp_path).family_ids(selected, pool_path=path).iloc[0] == "quest:7"


def test_unparented_rows_key_on_their_own_lane_and_id(tmp_path):
    path = _lineage(tmp_path, [("aug-1", "7", "quest", "quest")])
    selected = pd.DataFrame({"dataset": ["quest", "clerc"], "query_id": ["9", "9"]})
    assert list(_pool(tmp_path).family_ids(selected, pool_path=path)) == [
        "quest:9", "clerc:9"
    ]


def test_multi_generation_lineage_fails_loudly(tmp_path):
    """`family_ids` links one hop. Single-generation is enforced upstream by
    `first_generation_only` (0 grandchildren in 112,263 rows), so a grandchild
    means that guard was switched off and the fence would leak silently."""
    path = _lineage(tmp_path, [
        ("child", "7", "quest", "quest"),
        ("grandchild", "child", "quest", "quest"),
    ])
    selected = pd.DataFrame({"dataset": ["quest"], "query_id": ["grandchild"]})
    with pytest.raises(AssertionError, match="one hop only"):
        _pool(tmp_path).family_ids(selected, pool_path=path)


def test_missing_lineage_file_falls_back_to_own_id(tmp_path):
    selected = pd.DataFrame({"dataset": ["quest"], "query_id": ["9"]})
    fam = _pool(tmp_path).family_ids(selected, pool_path=tmp_path / "absent.parquet")
    assert list(fam) == ["quest:9"]


def test_reserve_fence_excludes_family_mates(tmp_path):
    """A child whose parent sits in the eval reserve must not stay selectable —
    379 such rows were measured live before the fence grouped on family."""
    _lineage(tmp_path, [("aug-1", "7", "quest", "quest")])
    pool = pd.DataFrame(
        {"dataset": ["quest", "quest", "clerc"], "query_id": ["7", "aug-1", "9"]}
    )
    excluded = _pool(tmp_path).reserve_exclusion_keys(pool, pool.loc[[0]])
    assert set(excluded) == {0, 1}


def test_fence_rejects_a_reserve_that_lost_pools_index(tmp_path):
    """`run_ablation` reads the reserve back from disk; rebuilt with merge() it
    carries a fresh RangeIndex and the fence would quietly exclude wrong rows."""
    _lineage(tmp_path, [("aug-1", "7", "quest", "quest")])
    pool = pd.DataFrame(
        {"dataset": ["quest", "quest", "clerc"], "query_id": ["7", "aug-1", "9"]},
        index=[10, 11, 12],
    )
    reindexed = pool.loc[[10]].reset_index(drop=True)
    with pytest.raises(AssertionError, match="pool's index"):
        _pool(tmp_path).reserve_exclusion_keys(pool, reindexed)


def _cyc(values, n) -> list:
    return list(islice(cycle(values), n))


def _rows(routes, certified=True) -> pd.DataFrame:
    """A gateable frame of any size carrying every column the matrix reads — a
    population short of them dies on a KeyError instead of being scored."""
    n = len(routes)
    return pd.DataFrame({
        "dataset": _cyc(("quest", "clerc"), n),
        "query_id": [str(i) for i in range(n)],
        "route_class": list(routes),
        "certified": (
            list(certified) if isinstance(certified, list) else [certified] * n
        ),
        "cells": _cyc((("rare_term_query",), ()), n),
        "query": _cyc(("one two", "three", "four five six", "seven"), n),
        **{band: _cyc(("low", "high"), n) for band in BANDS},
    })


def _gate(tmp_path, population, certified=True) -> ShortcutGate:
    gate = ShortcutGate(out_dir=tmp_path)
    _rows(population, certified).to_parquet(gate.population_path)
    return gate


def _target(routes, certified=True) -> pd.DataFrame:
    return pd.DataFrame({"route_class": routes, "certified": certified})


def test_comparator_matches_the_target_class_counts(tmp_path):
    """Gaps measured on different class mixes are not comparable, so the
    comparator draws the target's own counts, never a fixed frozen draw."""
    gate = _gate(tmp_path, ["dense"] * 10 + ["sparse"] * 10)
    got = gate.comparator(_target(["dense"] * 3 + ["sparse"] * 5))
    assert got["route_class"].value_counts().to_dict() == {"sparse": 5, "dense": 3}


def test_comparator_matches_certification_not_only_class(tmp_path):
    """An uncertified label is a bare argmax — arms with different certified
    shares are scored on labels of different reliability."""
    gate = _gate(tmp_path, ["dense"] * 10, certified=[True] * 5 + [False] * 5)
    got = gate.comparator(_target(["dense"] * 4, [True] * 3 + [False]))
    assert got["certified"].value_counts().to_dict() == {True: 3, False: 1}


def test_comparator_reports_a_cell_it_cannot_supply(tmp_path):
    """Short cells are reported, not raised on: an artifact built before the
    current reserve fence is still worth gating, just not fully matched."""
    gate = _gate(tmp_path, ["dense"] * 2)
    drawn = gate.comparator(_target(["dense"] * 5))
    assert len(drawn) == 2
    assert drawn.attrs["match_deficit"] == {"dense/certified=True": 3}


def _frame() -> pd.DataFrame:
    return pd.DataFrame({
        "dataset": ["quest", "clerc", "quest", "clerc"],
        "query_id": list("abcd"),
        "route_class": ["dense", "sparse", "dense", "sparse"],
        "certified": [True, True, False, False],
        "cells": [("rare_term_query",), (), ("rare_term_query",), ()],
        "query": ["one two", "three", "four five six", "seven"],
        **{band: ["low", "high", "low", "high"] for band in BANDS},
    })


def test_empty_and_all_waste_artifacts_are_rejected_not_crashed(tmp_path):
    """A gate that dies inside StratifiedKFold reads as a broken pipeline rather
    than an ungateable draw, so both cases raise a named refusal instead."""
    gate = ShortcutGate(out_dir=tmp_path)
    empty = _frame().iloc[:0]
    with pytest.raises(ValueError, match="not gateable"):
        gate.score(empty, "empty")
    all_waste = _frame().assign(route_class="waste")
    with pytest.raises(ValueError, match="not gateable"):
        gate.score(all_waste, "all_waste")


def test_run_refuses_an_ungateable_artifact_before_the_comparator(tmp_path):
    """The guard in score() is reached only AFTER run() builds the comparator,
    which dies in concat() on an empty draw — so run() must check first."""
    gate = _gate(tmp_path, ["dense"] * 4 + ["sparse"] * 4)
    artifact = tmp_path / "all_waste.parquet"
    _frame().assign(route_class="waste").to_parquet(artifact)
    with pytest.raises(ValueError, match="not gateable"):
        gate.run(artifact)


def test_an_unbuildable_control_downgrades_instead_of_aborting(tmp_path):
    """A deficit is contracted to REPORT — a control the population cannot build
    must not also destroy the artifact's own numbers."""
    gate = _gate(tmp_path, ["dense"] * 4)          # population holds no sparse
    artifact = tmp_path / "art.parquet"
    _frame().to_parquet(artifact)                  # artifact holds both classes
    out = gate.run(artifact)
    assert list(out["name"]) == ["art"]
    report = gate.report_path.read_text()
    assert "INCONCLUSIVE" in report and "Match deficit" in report


def test_a_small_artifact_shrinks_the_fold_count_instead_of_failing(tmp_path):
    """Four rows cannot make five folds; the gate adapts rather than crashing."""
    score = ShortcutGate(out_dir=tmp_path).score(_frame(), "tiny")
    assert score.rows == 4


def test_context_excludes_everything_retrieval_measured():
    """The gate is meaningless if a retrieval-produced column reaches `X`. The
    frame is NARROWED to CONTEXT at construction, so widening CONTEXT is the one
    edit that could break it — which is what this pins."""
    assert not (MEASURED & set(NuisanceMatrix.CONTEXT))


def test_extra_columns_are_dropped_not_featurised():
    base = _frame()
    leaked = base.assign(margin=1.0, oracle=0.9, depth=3, certified=True)
    assert NuisanceMatrix(base).X.shape == NuisanceMatrix(leaked).X.shape


def test_matrix_rows_match_the_frame():
    matrix = NuisanceMatrix(_frame())
    assert matrix.X.shape[0] == len(matrix.y) == 4
