"""Unit tests for the features.py machinery, isolated from the real bank
inventories: fake banks with canned outputs pin claim resolution, group
layering, profile math, and extractor validation. Real-bank integration is
covered by features_summary_test.py / regex_bank_test.py.
"""

from enum import StrEnum

import pytest
from edify import RegexBuilder

from query_taxonomy.core import (
    AmbiguityTier,
    FeatureSpan,
    FeatureStat,
    RegexBank,
    StatBank,
)
from query_taxonomy.features import (
    FeatureExtractor,
    QueryFeatures,
    SpanProfile,
    StatProfile,
    normalized_idf,
)
from query_taxonomy.taxonomy import FeatureGroup


class FakeType(StrEnum):
    ALPHA = "alpha"
    BETA = "beta"
    STAT = "stat"


def span_bank(
    name: FakeType,
    group: FeatureGroup,
    tier: AmbiguityTier,
    spans_by_text: dict[str, list[FeatureSpan]],
) -> type[RegexBank]:
    class _SpanBank(RegexBank):
        @property
        def name(self) -> FakeType:
            return name

        @property
        def group(self) -> FeatureGroup:
            return group

        @property
        def ambiguity(self) -> AmbiguityTier:
            return tier

        def define(self, builder: RegexBuilder) -> RegexBuilder:
            # Fake bank — regex is unused, compute() below returns canned spans.
            return builder

        def compute(self, text: str) -> list[FeatureSpan]:
            return list(spans_by_text.get(text, []))

    return _SpanBank


def stat_bank(
    name: FakeType,
    group: FeatureGroup,
    stats_by_text: dict[str, list[FeatureStat]],
) -> type[StatBank[FeatureStat]]:
    class _StatBank(StatBank[FeatureStat]):
        @property
        def name(self) -> FakeType:
            return name

        @property
        def group(self) -> FeatureGroup:
            return group

        def define(self, builder: FeatureStat) -> FeatureStat:
            return builder

        def compute(self, text: str) -> list[FeatureStat]:
            return list(stats_by_text.get(text, []))

    return _StatBank


IDS = FeatureGroup.STRUCTURED_IDENTIFIERS
MARKERS = FeatureGroup.SENTENCE_MARKERS
METRICS = FeatureGroup.STATISTICAL_METRICS


# --- normalized_idf ---


def test_normalized_idf_bounds():
    assert normalized_idf(0, 100) == 1.0
    assert normalized_idf(100, 100) == 0.0
    assert normalized_idf(5, 0) == 0.0


def test_normalized_idf_decreases_with_df():
    values = [normalized_idf(df, 100) for df in (0, 1, 10, 50, 100)]
    assert values == sorted(values, reverse=True)


# --- profile models ---


def test_span_profile_diversity_and_dfs():
    profile = SpanProfile(
        type="alpha",
        spans={
            "0": [FeatureSpan("foo", 0, 3), FeatureSpan("foo", 10, 13)],
            "1": [FeatureSpan("foo", 0, 3), FeatureSpan("bar", 4, 7)],
        },
    )
    assert profile.diversity == 2
    # repeated form within one doc counts that doc once
    assert profile.dfs == {"foo": 2, "bar": 1}


def test_stat_profile_pools_aggregates_per_name():
    profile = StatProfile(
        type="stat",
        values={
            "0": [FeatureStat("length", 4.0), FeatureStat("ratio", 0.5)],
            "1": [FeatureStat("length", 8.0)],
        },
    )
    assert profile.aggregates["length"] == {
        "count": 2.0,
        "mean": 6.0,
        "min": 4.0,
        "max": 8.0,
    }
    assert profile.aggregates["ratio"]["count"] == 1.0


def test_query_features_tfs_counts_per_group_and_type():
    features = QueryFeatures(
        query_text="q",
        spans={
            IDS: {
                "alpha": [FeatureSpan("a", 0, 1), FeatureSpan("b", 2, 3)],
                "beta": [FeatureSpan("c", 4, 5)],
            }
        },
        stats={},
    )
    assert features.tfs == {IDS: {"alpha": 2, "beta": 1}}


# --- extractor validation ---


def test_extractor_rejects_bank_registered_under_wrong_group():
    stray = span_bank(FakeType.ALPHA, MARKERS, AmbiguityTier.RIGID, {})
    with pytest.raises(ValueError, match="declares group"):
        FeatureExtractor({IDS: [stray]})


def test_extractor_rejects_unknown_group_selection():
    extractor = FeatureExtractor(
        {IDS: [span_bank(FakeType.ALPHA, IDS, AmbiguityTier.RIGID, {})]}
    )
    with pytest.raises(ValueError, match="no banks registered"):
        extractor.resolve("text", groups=[MARKERS])


def test_repeated_group_selection_is_deduped():
    text = "hello"
    bank = span_bank(
        FakeType.ALPHA, IDS, AmbiguityTier.RIGID,
        {text: [FeatureSpan("hello", 0, 5)]},
    )
    extractor = FeatureExtractor({IDS: [bank]})
    features = extractor.resolve(text, groups=[IDS, IDS])
    assert features.tfs[IDS]["alpha"] == 1


# --- claim resolution ---


def test_rigid_tier_claims_before_ambiguous():
    text = "v1.0.0"
    rigid = span_bank(
        FakeType.ALPHA, IDS, AmbiguityTier.RIGID,
        {text: [FeatureSpan("v1.0.0", 0, 6)]},
    )
    ambiguous = span_bank(
        FakeType.BETA, IDS, AmbiguityTier.AMBIGUOUS,
        {text: [FeatureSpan("1.0", 1, 4), FeatureSpan("0", 5, 6)]},
    )
    # registration order says ambiguous first; tier order must still win
    extractor = FeatureExtractor({IDS: [ambiguous, rigid]})
    spans = extractor.resolve(text).spans[IDS]
    assert [s.text for s in spans["alpha"]] == ["v1.0.0"]
    assert "beta" not in spans


def test_registration_order_breaks_ties_within_tier():
    text = "overlap"
    alpha = span_bank(FakeType.ALPHA, IDS, AmbiguityTier.RIGID,
                      {text: [FeatureSpan("over", 0, 4)]})
    beta = span_bank(FakeType.BETA, IDS, AmbiguityTier.RIGID,
                     {text: [FeatureSpan("verla", 1, 6)]})

    alpha_first = FeatureExtractor({IDS: [alpha, beta]})
    assert list(alpha_first.resolve(text).spans[IDS]) == ["alpha"]

    beta_first = FeatureExtractor({IDS: [beta, alpha]})
    assert list(beta_first.resolve(text).spans[IDS]) == ["beta"]


def test_groups_are_independent_claim_layers():
    text = "no"
    identifier = span_bank(
        FakeType.ALPHA, IDS, AmbiguityTier.RIGID,
        {text: [FeatureSpan("no", 0, 2)]},
    )
    marker = span_bank(
        FakeType.BETA, MARKERS, AmbiguityTier.RIGID,
        {text: [FeatureSpan("no", 0, 2)]},
    )
    extractor = FeatureExtractor({IDS: [identifier], MARKERS: [marker]})
    features = extractor.resolve(text)
    assert features.tfs == {IDS: {"alpha": 1}, MARKERS: {"beta": 1}}


def test_stats_never_enter_the_claim_registry():
    text = "abc"
    claimer = span_bank(
        FakeType.ALPHA, METRICS, AmbiguityTier.RIGID,
        {text: [FeatureSpan("abc", 0, 3)]},
    )
    stats = stat_bank(
        FakeType.STAT, METRICS, {text: [FeatureStat("length", 3.0)]}
    )
    extractor = FeatureExtractor({METRICS: [claimer, stats]})
    features = extractor.resolve(text)
    assert features.tfs[METRICS]["alpha"] == 1
    assert features.stats[METRICS]["stat"] == [FeatureStat("length", 3.0)]


# --- corpus extraction ---


def test_extract_builds_doc_keyed_profiles():
    bank = span_bank(
        FakeType.ALPHA, IDS, AmbiguityTier.RIGID,
        {"one foo": [FeatureSpan("foo", 4, 7)],
         "two foo": [FeatureSpan("foo", 4, 7)]},
    )
    stats = stat_bank(
        FakeType.STAT, METRICS,
        {"one foo": [FeatureStat("length", 7.0)],
         "two foo": [FeatureStat("length", 7.0)],
         "bare": [FeatureStat("length", 4.0)]},
    )
    extractor = FeatureExtractor({IDS: [bank], METRICS: [stats]})
    corpus = extractor.extract(["one foo", "two foo", "bare"])

    profile = corpus.span_profiles[IDS]["alpha"]
    assert set(profile.spans) == {"0", "1"}
    assert profile.dfs == {"foo": 2}
    assert corpus.stat_profiles[METRICS]["stat"].aggregates["length"]["count"] == 3.0
    assert len(corpus.queries) == 3


def test_extract_empty_corpus():
    extractor = FeatureExtractor(
        {IDS: [span_bank(FakeType.ALPHA, IDS, AmbiguityTier.RIGID, {})]}
    )
    corpus = extractor.extract([])
    assert corpus.queries == []
    assert corpus.summary() == "queries: 0 tagged: 0 (0.0%)"
