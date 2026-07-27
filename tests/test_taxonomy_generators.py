from random import Random

import pytest

from query_taxonomy import FEATURE_BANKS
from query_taxonomy.core import AmbiguityTier, RegexBank
from query_taxonomy.taxonomy import FeatureGroup

from taxonomy_generators import (
    FEATURE_GENERATORS,
    SpanTarget,
    StatTarget,
    SurfaceGenerator,
    Targets,
    build_registry,
    build_tools,
    verify,
)

BANKS_BY_FEATURE = {
    str(bank.name): bank
    for classes in FEATURE_BANKS.values()
    for cls in classes
    if issubclass(cls, RegexBank)
    for bank in (cls(),)
}

GENERATORS = [
    generator
    for members in FEATURE_GENERATORS.values()
    for generator in members
]


@pytest.mark.parametrize("generator", GENERATORS, ids=lambda g: g.feature)
def test_round_trip(generator: SurfaceGenerator) -> None:
    """d34b lock-step invariant: every sampled surface is claimed in full
    by its twin detection bank."""
    bank = BANKS_BY_FEATURE[generator.feature]
    for surface in generator.sample(Random(0), 20):
        spans = bank.compute(surface)
        assert any(span.text == surface for span in spans), (
            f"{generator.feature}: {surface!r} not fully claimed"
            f" — got {spans}"
        )


def test_every_regex_span_bank_has_a_generator() -> None:
    assert set(BANKS_BY_FEATURE) == {g.feature for g in GENERATORS}


class _UuidOverride(SurfaceGenerator):
    @property
    def feature(self) -> str:
        return "uuid"

    @property
    def group(self) -> FeatureGroup:
        return FeatureGroup.STRUCTURED_IDENTIFIERS

    @property
    def tier(self) -> AmbiguityTier:
        return AmbiguityTier.RIGID

    @property
    def description(self) -> str:
        return "fixed uuid for the shadow test"

    def sample(self, rng: Random, n: int = 1) -> list[str]:
        return ["550e8400-e29b-41d4-a716-446655440000"] * n


class _InventedOverride(_UuidOverride):
    @property
    def feature(self) -> str:
        return "not-a-taxonomy-feature"


def test_override_shadows_default_by_feature_name() -> None:
    registry = build_registry(overrides=(_UuidOverride,))
    generators = registry[FeatureGroup.STRUCTURED_IDENTIFIERS]
    shadowed = next(g for g in generators if g.feature == "uuid")
    assert isinstance(shadowed, _UuidOverride)


def test_override_without_twin_bank_fails_loudly() -> None:
    with pytest.raises(ValueError, match="without a twin bank"):
        build_registry(overrides=(_InventedOverride,))


def test_verify_span_targets() -> None:
    targets = Targets(spans=(SpanTarget(feature="uuid"),))
    hit = verify(
        "reset router 550e8400-e29b-41d4-a716-446655440000", targets
    )
    miss = verify("reset router", targets)
    assert hit.passed
    assert not miss.passed
    assert miss.checks[0].measured == 0.0


def test_verify_absent_stat_fails_with_none() -> None:
    report = verify(
        "anything", Targets(stats=(StatTarget(stat="nesting_depth"),))
    )
    assert not report.passed
    assert report.checks[0].measured is None


def test_generate_surface_tool_is_seed_deterministic() -> None:
    tools = {spec.name: spec for spec in build_tools(seed=1)}
    first = tools["generate_surface"].run(feature="uuid", n=3, seed=7)
    second = tools["generate_surface"].run(feature="uuid", n=3, seed=7)
    assert first == second
    assert len(first["surfaces"]) == 3
