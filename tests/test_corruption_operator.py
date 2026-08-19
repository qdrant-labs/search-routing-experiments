"""The corruptor damages what the detectors see, reproduces from its seed,
and never reaches the model."""

import pandas as pd
import pytest

from augmentation.config import AugmentationConfig
from augmentation.corruption import (
    CorruptionDegree,
    CorruptOperator,
    LaneBaseline,
    Mojibake,
    QueryCorruptor,
    QwertyTypo,
    Truncate,
)
from query_taxonomy.corruption import (
    EncodingArtifactBank,
    TruncationBank,
    TypoBank,
)

QUERY = "how to configure nginx reverse proxy for a docker container"


@pytest.fixture(scope="module")
def corruptor() -> QueryCorruptor:
    return QueryCorruptor(LaneBaseline(AugmentationConfig().paths.corruption_census))


@pytest.mark.parametrize(
    "perturbation, bank",
    [
        (QwertyTypo(), TypoBank()),
        (Mojibake(), EncodingArtifactBank()),
        (Truncate(), TruncationBank()),
    ],
)
def test_each_perturbation_fires_its_own_detector(perturbation, bank):
    from random import Random

    damaged = perturbation.apply(QUERY, Random(0))
    assert damaged != QUERY
    assert bank.compute(damaged), f"{perturbation.kind} left {bank.name} silent"


def test_degree_adds_spans_over_the_parent_count(corruptor):
    base = corruptor.spans(QUERY)
    for degree in (CorruptionDegree.LIGHT, CorruptionDegree.HEAVY):
        damaged = corruptor.corrupt(QUERY, "miracl-en-dev", degree, "q1")
        assert corruptor.spans(damaged) >= base + degree.added_spans


def test_clean_degree_is_a_no_op(corruptor):
    assert corruptor.corrupt(QUERY, "miracl-en-dev", CorruptionDegree.CLEAN, "q1") == QUERY


def test_same_seed_same_damage(corruptor):
    args = (QUERY, "miracl-en-dev", CorruptionDegree.HEAVY, "q1")
    assert corruptor.corrupt(*args) == corruptor.corrupt(*args)


def test_headroom_is_lane_sensitive(corruptor):
    """orcas is the only lane with any natural mojibake (0.0010), so it is
    the only one that must not spend its first span there."""
    baseline = LaneBaseline(AugmentationConfig().paths.corruption_census)
    assert baseline.by_headroom("orcas")[0].value == "truncation"
    assert baseline.by_headroom("miracl-en-dev")[0].value == "encoding_artifact"
    assert corruptor.corrupt(QUERY, "orcas", CorruptionDegree.LIGHT, "q1") != (
        corruptor.corrupt(QUERY, "miracl-en-dev", CorruptionDegree.LIGHT, "q1")
    )


def test_typo_saturated_lane_spends_typo_last():
    """A lane already firing typo on 47% of queries learns nothing from one
    more, which is the whole reason the census feeds this."""
    baseline = LaneBaseline(AugmentationConfig().paths.corruption_census)
    assert baseline.by_headroom("bright-robotics")[-1].value == "typo"


def test_heavy_never_nests_mojibake(corruptor):
    """Re-encoding already-mojibaked text yields unreadable garbage, so a
    non-repeatable move must be spent at most once."""
    damaged = corruptor.corrupt(QUERY, "miracl-en-dev", CorruptionDegree.HEAVY, "q1")
    assert "Ã\x83" not in damaged and "Â\x83" not in damaged


def test_operator_is_deterministic_not_generative():
    operator = CorruptOperator()
    parent = pd.Series(
        {"query": QUERY, "dataset": "miracl-en-dev", "query_id": "q1"}
    )
    assert operator.serves("corruption:heavy")
    assert not operator.serves("corruption:nonsense")
    assert operator.apply(parent, "corruption:heavy", QUERY) != QUERY
    with pytest.raises(NotImplementedError):
        operator.instruction("corruption:heavy", parent)


def test_structural_rejects_a_no_op():
    operator = CorruptOperator()
    parent = pd.Series({"query": QUERY, "dataset": "miracl-en-dev", "query_id": "q1"})
    assert operator.structural(parent, QUERY, operator.targets("corruption:light", parent))
    assert not operator.structural(parent, QUERY + "x...", operator.targets("corruption:light", parent))
