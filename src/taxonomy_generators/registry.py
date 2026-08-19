"""Group-keyed generator registry, the FEATURE_BANKS mirror (SPEC d34f).

Defaults are auto-built at import by iterating the detection registry:
every regex span bank gets a PatternGenerator twin. Stat banks are skipped
by doctrine — the five signals are verify-only acceptance filters (d26),
never generated. Overrides shadow defaults by feature name.
"""

from query_taxonomy import FEATURE_BANKS, split_bank
from query_taxonomy.core import RegexBank
from query_taxonomy.taxonomy import FeatureGroup

from taxonomy_generators.core import PatternGenerator, SurfaceGenerator
from taxonomy_generators.overrides import OVERRIDES


def build_registry(
    overrides: tuple[type[SurfaceGenerator], ...] = OVERRIDES,
) -> dict[FeatureGroup, tuple[SurfaceGenerator, ...]]:
    shadows: dict[str, SurfaceGenerator] = {}
    for override_cls in overrides:
        override = override_cls()
        if override.feature in shadows:
            raise ValueError(f"duplicate override for {override.feature!r}")
        shadows[override.feature] = override

    generators: dict[FeatureGroup, tuple[SurfaceGenerator, ...]] = {}
    for group, bank_specs in FEATURE_BANKS.items():
        members: list[SurfaceGenerator] = []
        for spec in bank_specs:
            # a registry entry is a bank type or a (type, kwargs) pair since
            # calibrated values inject at construction; the kwargs belong to
            # stat banks, which this loop skips by doctrine anyway
            bank_cls, _kwargs = split_bank(spec)
            if not issubclass(bank_cls, RegexBank):
                continue
            default = PatternGenerator(bank_cls)
            member = shadows.pop(default.feature, default)
            if member.group is not group:
                raise ValueError(
                    f"override for {member.feature!r} declares group "
                    f"{member.group}, but its twin bank lives in {group}"
                )
            members.append(member)
        if members:
            generators[group] = tuple(members)

    if shadows:
        raise ValueError(
            f"overrides without a twin bank: {sorted(shadows)} — an override"
            " must shadow an existing feature, never invent one"
        )
    return generators


FEATURE_GENERATORS: dict[FeatureGroup, tuple[SurfaceGenerator, ...]] = (
    build_registry()
)

_BY_FEATURE: dict[str, SurfaceGenerator] = {
    generator.feature: generator
    for members in FEATURE_GENERATORS.values()
    for generator in members
}


def generator_for(feature: str) -> SurfaceGenerator:
    try:
        return _BY_FEATURE[feature]
    except KeyError:
        raise KeyError(
            f"no generator for feature {feature!r} — valid names come from"
            " list_features()/catalog()"
        ) from None


def catalog() -> list[dict[str, str]]:
    """LLM-facing feature catalog: the list_features() tool payload."""
    return [
        {
            "feature": generator.feature,
            "group": str(generator.group),
            "tier": generator.tier.name.lower(),
            "description": generator.description,
        }
        for members in FEATURE_GENERATORS.values()
        for generator in members
    ]
