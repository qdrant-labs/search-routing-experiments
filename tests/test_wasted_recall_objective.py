"""The promoted WastedRecallObjective: hand-vector arithmetic, the
ordered-list ndcg regression, and the two golden-generation invariants
(depth cross-check, no confidence term on the selection surface)."""

import inspect

import pytest

from composition.recipe import Recipe
from hybrid_search_rrf_dataset.objective import WastedRecallObjective


def test_perfect_top_and_full_recall_pays_no_penalty():
    obj = WastedRecallObjective()
    score, top = obj.assess({"d1": 3.0, "d2": 2.0}, {"d1": 1})
    assert top[0] == "d1"
    # hit 0.7 + 0.3 * ndcg^2 with ndcg == 1, minus 0.3 * recall * (1 - 1) = 0
    assert score == pytest.approx(1.0)


def test_no_judged_docs_scores_zero():
    obj = WastedRecallObjective()
    score, _ = obj.assess({"d1": 3.0}, {"d2": 0})
    assert score == 0.0


def test_margin_never_certifies_a_top1_separation():
    assert WastedRecallObjective().decisive_margin == float("inf")


def test_ndcg_reads_the_stored_top_k_not_the_raw_dict():
    # both docs tie on score; ordered() keeps insertion order stably, so the
    # stored top-1 is the irrelevant "a". Scoring from the raw dict lets
    # ranx's unstable sort count "b" — the parquet round-trip bug the
    # notebook's version still carries.
    obj = WastedRecallObjective(top_k=1)
    score, top = obj.assess({"a": 1.0, "b": 1.0}, {"b": 1})
    assert top == ["a"]
    assert score == 0.0


def test_docs_per_query_default_is_the_genuine_tie_depth():
    from augmentation.loop import AugmentationLoop

    default = inspect.signature(
        AugmentationLoop.synthesize
    ).parameters["docs_per_query"].default
    # the 2 is computed, not chosen: it is exactly assign_classes' fake-tie
    # boundary, where one more doc buys nothing and one fewer is waste
    assert default == Recipe().genuine_tie_depth


def test_no_confidence_term_reaches_the_selection_surface():
    # leg_confidence may not exist anywhere on the recipe or pool surface
    # until its referent, direction and mechanism are written down (Layer 2)
    from composition import objectives, pool_v3

    assert "leg_confidence" not in Recipe.model_fields
    assert "leg_confidence" not in inspect.getsource(pool_v3)
    assert "leg_confidence" not in inspect.getsource(objectives)
