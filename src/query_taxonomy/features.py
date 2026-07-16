import math
from collections.abc import Iterable, Mapping
from functools import cached_property

from pydantic import BaseModel, ConfigDict, computed_field

from query_taxonomy import FEATURE_BANKS, Bank, BankTypes
from query_taxonomy.banks import domain_by_type
from query_taxonomy.core import FeatureSpan, FeatureStat
from query_taxonomy.taxonomy import Domain, FeatureGroup


def normalized_idf(df: int, n_docs: int) -> float:
    """
    Smoothed idf in [0, 1]: log((N+1)/(df+1)) / log(N+1).
    df=0 (absent from corpus) maps to 1.0, df=N to 0.0.
    """
    if n_docs == 0:
        return 0.0

    return math.log((n_docs + 1) / (df + 1)) / math.log(n_docs + 1)


class SpanProfile(BaseModel):
    """Corpus profile of one span-emitting feature type."""

    model_config = ConfigDict(frozen=True)

    type: str
    # doc_ids -> list of claimed spans
    spans: dict[str, list[FeatureSpan]]

    @computed_field
    @cached_property
    def diversity(self) -> int:
        """Corpus-wide: distinct surface forms of this type."""
        return len({
            m.text for matches in self.spans.values() for m in matches
        })

    @computed_field
    @cached_property
    def dfs(self) -> dict[str, int]:
        """Per surface form: number of docs containing it."""
        counts: dict[str, int] = {}
        for matches in self.spans.values():
            for text in {m.text for m in matches}:
                counts[text] = counts.get(text, 0) + 1
        return counts


class StatProfile(BaseModel):
    """Corpus profile of one stat-emitting feature type. Per-doc values are
    kept — recipe strata stratify on them — with aggregates as views."""

    model_config = ConfigDict(frozen=True)

    type: str
    # doc_ids -> list of named stats
    values: dict[str, list[FeatureStat]]

    @computed_field
    @cached_property
    def aggregates(self) -> dict[str, dict[str, float]]:
        """Per stat name: count / mean / min / max over the corpus."""
        pooled: dict[str, list[float]] = {}
        for stats in self.values.values():
            for stat in stats:
                pooled.setdefault(stat.name, []).append(stat.value)
        return {
            name: {
                "count": float(len(values)),
                "mean": sum(values) / len(values),
                "min": min(values),
                "max": max(values),
            }
            for name, values in pooled.items()
        }


class QueryFeatures(BaseModel):
    """Feature outputs of one query, spans and stats sectioned by group."""

    model_config = ConfigDict(frozen=True)

    query_text: str

    spans: dict[FeatureGroup, dict[str, list[FeatureSpan]]]
    stats: dict[FeatureGroup, dict[str, list[FeatureStat]]]

    @computed_field
    @property
    def tfs(self) -> dict[FeatureGroup, dict[str, int]]:
        return {
            group: {type_: len(matches) for type_, matches in types.items()}
            for group, types in self.spans.items()
        }


class CorpusFeatures(BaseModel):
    model_config = ConfigDict(frozen=True)

    span_profiles: dict[FeatureGroup, dict[str, SpanProfile]]
    """Per group, per type present in the corpus: doc-keyed spans profile."""
    stat_profiles: dict[FeatureGroup, dict[str, StatProfile]]
    """Per group, per type: doc-keyed stat values with corpus aggregates."""
    queries: list[QueryFeatures]
    """One entry per input query, in input order."""

    def summary(self) -> str:
        """Group-sectioned feature report. Tagged = at least one span in any
        group; stats never tag a query. Inside structured_identifiers the
        per-domain view doubles as the FP smell-test: hits from an off-topic
        domain (finance types on a QA corpus) are a priori suspect."""
        tagged_total = sum(1 for query in self.queries if query.spans)
        total = len(self.queries)
        share = 100 * tagged_total / total if total else 0.0
        lines = [f"queries: {total} tagged: {tagged_total} ({share:.1f}%)"]

        def tagged(profiles: Iterable[SpanProfile]) -> int:
            return len({
                doc_id for profile in profiles for doc_id in profile.spans
            })

        def type_lines(profiles: list[SpanProfile]) -> list[str]:
            pad = max((len(profile.type) for profile in profiles), default=0)
            out: list[str] = []
            for profile in sorted(profiles, key=lambda p: -len(p.spans)):
                matches = sum(len(spans) for spans in profile.spans.values())
                out.append(
                    f"{profile.type:<{pad}}  queries={len(profile.spans):3d} "
                    f"matches={matches:3d} diversity={profile.diversity:3d}"
                )
                top = sorted(profile.dfs.items(), key=lambda item: -item[1])[:3]
                forms = ", ".join(f"{text} ({df})" for text, df in top)
                out.append(f"{'':<{pad}}  top: {forms}")
            return out

        for group, by_type in sorted(
            self.span_profiles.items(),
            key=lambda item: -tagged(item[1].values()),
        ):
            profiles = list(by_type.values())
            lines.append("")
            lines.append(
                f"== {group.value} "
                f"({len(profiles)} types, {tagged(profiles)} tagged)"
            )
            if group is FeatureGroup.STRUCTURED_IDENTIFIERS:
                by_domain: dict[Domain, list[SpanProfile]] = {}
                for profile in profiles:
                    domain = domain_by_type()[profile.type]
                    by_domain.setdefault(domain, []).append(profile)
                for domain, docs in sorted(
                    by_domain.items(), key=lambda item: -tagged(item[1])
                ):
                    lines.append(
                        f"-- {domain.value} ({len(docs)} types, "
                        f"{tagged(docs)} tagged)"
                    )
                    lines.extend(type_lines(docs))
            else:
                lines.extend(type_lines(profiles))

        for group, by_type in sorted(self.stat_profiles.items()):
            lines.append("")
            lines.append(f"== {group.value} ({len(by_type)} types)")
            for profile in sorted(by_type.values(), key=lambda p: p.type):
                for name, agg in sorted(profile.aggregates.items()):
                    lines.append(
                        f"{profile.type}.{name}  docs={agg['count']:.0f} "
                        f"mean={agg['mean']:.3f} min={agg['min']:.3f} "
                        f"max={agg['max']:.3f}"
                    )
        return "\n".join(lines)

    def __str__(self) -> str:
        return self.summary()

class FeatureExtractor:
    """
    Runs registered banks group by group over a corpus of queries. Within a
    group, banks run in AmbiguityTier order (RIGID first; registration order
    within a tier) against a per-text registry of claimed char ranges, so a
    lower-priority bank never re-claims overlapping text — NUMBER can't
    steal `1.0` from a claimed `v1.0.0`. Groups are independent layers:
    spans of different groups may overlap; stats never claim.
    """

    def __init__(
        self,
        banks: Mapping[FeatureGroup, Iterable[BankTypes]] = FEATURE_BANKS,
    ) -> None:
        self._by_group: dict[FeatureGroup, list[Bank]] = {}
        for group, classes in banks.items():
            # stable sort -> registration order breaks ties within a tier
            instances = sorted(
                (cls() for cls in classes), key=lambda bank: bank.ambiguity
            )
            for bank in instances:
                if bank.group is not group:
                    raise ValueError(
                        f"bank {bank.name!r} declares group {bank.group}, "
                        f"but is registered under {group}"
                    )
            self._by_group[group] = instances

    def _selected(
        self, groups: Iterable[FeatureGroup] | None
    ) -> list[FeatureGroup]:
        if groups is None:
            return list(self._by_group)
        # dedup, order-preserving: a repeated group would re-run its banks
        # against a fresh claim registry and emit duplicate spans
        selected = list(dict.fromkeys(groups))
        unknown = [group for group in selected if group not in self._by_group]
        if unknown:
            raise ValueError(f"no banks registered for groups: {unknown}")
        return selected

    def resolve(
        self, text: str, *, groups: Iterable[FeatureGroup] | None = None
    ) -> QueryFeatures:
        """Claim-resolved outputs for one text, sectioned by group."""
        spans: dict[FeatureGroup, dict[str, list[FeatureSpan]]] = {}
        stats: dict[FeatureGroup, dict[str, list[FeatureStat]]] = {}
        for group in self._selected(groups):
            registry: list[FeatureSpan] = []
            for bank in self._by_group[group]:
                name = str(bank.name)
                for out in bank.compute(text):
                    if isinstance(out, FeatureSpan):
                        if any(
                            out.start < c.end and c.start < out.end
                            for c in registry
                        ):
                            continue
                        registry.append(out)
                        spans.setdefault(group, {}).setdefault(name, []).append(out)
                    else:
                        stats.setdefault(group, {}).setdefault(name, []).append(out)
        return QueryFeatures(query_text=text, spans=spans, stats=stats)

    def extract(
        self,
        queries: Iterable[str],
        *,
        groups: Iterable[FeatureGroup] | None = None,
    ) -> CorpusFeatures:
        """Single pass: fills per-type profiles and QueryFeatures per query."""
        selected = None if groups is None else tuple(groups)
        span_docs: dict[FeatureGroup, dict[str, dict[str, list[FeatureSpan]]]] = {}
        stat_docs: dict[FeatureGroup, dict[str, dict[str, list[FeatureStat]]]] = {}
        query_models: list[QueryFeatures] = []
        for index, query in enumerate(queries):
            features = self.resolve(query, groups=selected)
            query_models.append(features)
            doc_id = str(index)
            for group, types in features.spans.items():
                for type_, matches in types.items():
                    span_docs.setdefault(group, {}).setdefault(type_, {})[
                        doc_id
                    ] = matches
            for group, types in features.stats.items():
                for type_, values in types.items():
                    stat_docs.setdefault(group, {}).setdefault(type_, {})[
                        doc_id
                    ] = values
        span_profiles = {
            group: {
                type_: SpanProfile(type=type_, spans=docs)
                for type_, docs in types.items()
            }
            for group, types in span_docs.items()
        }
        stat_profiles = {
            group: {
                type_: StatProfile(type=type_, values=docs)
                for type_, docs in types.items()
            }
            for group, types in stat_docs.items()
        }
        return CorpusFeatures(
            span_profiles=span_profiles,
            stat_profiles=stat_profiles,
            queries=query_models,
        )
