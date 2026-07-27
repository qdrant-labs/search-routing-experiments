"""verify(text, targets) — d10's library-first verb (SPEC d34a).

Measures a text with the detection banks and checks it against feature
targets: span counts and stat ranges (d2 — targets are quantities, features
are always re-measured on final text, never trusted from a generator).
"""

from pydantic import BaseModel, ConfigDict

from query_taxonomy.features import FeatureExtractor


class SpanTarget(BaseModel):
    """Count range for one span feature, e.g. uuid >= 1."""

    model_config = ConfigDict(frozen=True)

    feature: str
    min_count: int = 1
    max_count: int | None = None


class StatTarget(BaseModel):
    """Value range for one measured scalar, e.g. 3 <= length_words <= 6."""

    model_config = ConfigDict(frozen=True)

    stat: str
    min_value: float | None = None
    max_value: float | None = None


class Targets(BaseModel):
    """Per-feature quantities a text must exhibit (the d2 feature_targets
    shape, verification-side)."""

    model_config = ConfigDict(frozen=True)

    spans: tuple[SpanTarget, ...] = ()
    stats: tuple[StatTarget, ...] = ()


class TargetCheck(BaseModel):
    model_config = ConfigDict(frozen=True)

    target: str
    measured: float | None
    passed: bool


class VerifyReport(BaseModel):
    model_config = ConfigDict(frozen=True)

    passed: bool
    checks: tuple[TargetCheck, ...]


def _span_label(target: SpanTarget) -> str:
    upper = "" if target.max_count is None else f", at most {target.max_count}"
    return f"{target.feature}: at least {target.min_count} span(s){upper}"


def _stat_label(target: StatTarget) -> str:
    bounds = [
        f">= {target.min_value}" if target.min_value is not None else "",
        f"<= {target.max_value}" if target.max_value is not None else "",
    ]
    return f"{target.stat} {' and '.join(part for part in bounds if part)}"


def verify(
    text: str,
    targets: Targets,
    *,
    extractor: FeatureExtractor | None = None,
) -> VerifyReport:
    """PASS/FAIL per target plus measured values.

    The default extractor is regex-only, so span targets are always
    checkable; stat targets over the spaCy signals (natural_language_share,
    nesting_depth, ...) need a caller-supplied
    ``FeatureExtractor(engines=None)``. An absent stat fails its check with
    ``measured=None`` — loudly distinguishable from a wrong value.
    """
    extractor = extractor or FeatureExtractor()
    features = extractor.resolve(text)

    span_counts: dict[str, int] = {}
    for types in features.spans.values():
        for name, matches in types.items():
            span_counts[name] = span_counts.get(name, 0) + len(matches)
    stat_values: dict[str, float] = {}
    for types in features.stats.values():
        for values in types.values():
            for stat in values:
                stat_values[stat.name] = stat.value

    checks: list[TargetCheck] = []
    for span_target in targets.spans:
        count = span_counts.get(span_target.feature, 0)
        passed = count >= span_target.min_count and (
            span_target.max_count is None or count <= span_target.max_count
        )
        checks.append(
            TargetCheck(
                target=_span_label(span_target),
                measured=float(count),
                passed=passed,
            )
        )
    for stat_target in targets.stats:
        value = stat_values.get(stat_target.stat)
        passed = (
            value is not None
            and (stat_target.min_value is None or value >= stat_target.min_value)
            and (stat_target.max_value is None or value <= stat_target.max_value)
        )
        checks.append(
            TargetCheck(
                target=_stat_label(stat_target), measured=value, passed=passed
            )
        )
    return VerifyReport(
        passed=all(check.passed for check in checks), checks=tuple(checks)
    )
