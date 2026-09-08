"""Canonical axes over the d29 feature-table catalog — shared knowledge,
housed with the composition floors that select against it. The cells and the
composition floors (SPEC d33b) both read their band definitions from here, so
the coordinate system a human reads and the one the fill selects against can
never drift apart. Bands are RAW scalar units by decision d33b: stable,
generation-targetable, unmoved by catalog growth.

NOTE: the production router's classify() score 0..9 is a sparse-vs-dense
GRADE, not a word count — never draw length bands as router bands."""

from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd

from query_taxonomy.taxonomy import FeatureGroup


def stat_column(catalog: pd.DataFrame, stat: str) -> str:
    """Map a bare stat name to its `<bank>.<stat>` catalog column."""
    matches = [c for c in catalog.columns if c.endswith(f".{stat}")]
    if len(matches) != 1:
        raise ValueError(
            f"stat {stat!r} maps to {matches or 'no catalog column'}"
        )
    return matches[0]


class CoverageAxis(ABC):
    """One axis candidate: how to read a per-query value off the catalog
    and how to band it into cells."""

    def __init__(
        self,
        title: str,
        edges: tuple[float, ...],
        labels: tuple[str, ...],
    ) -> None:
        if len(labels) != len(edges) - 1:
            raise ValueError(
                f"{title}: {len(edges)} edges need {len(edges) - 1} "
                f"labels, got {len(labels)}"
            )
        self.title = title
        self.edges = edges
        self.labels = labels

    @property
    def y_excludes(self) -> frozenset[FeatureGroup]:
        """Span groups a chart's y-axis must not count under this x — an
        axis that is itself a span count would correlate with y by
        construction."""
        return frozenset()

    @abstractmethod
    def values(self, catalog: pd.DataFrame) -> pd.Series: ...

    def bands(self, catalog: pd.DataFrame) -> pd.Series:
        return pd.cut(
            self.values(catalog), self.edges, labels=self.labels,
            right=False,
        )


class StatAxis(CoverageAxis):
    """Axis over one scalar stat (a `<bank>.<stat>` catalog column)."""

    def __init__(
        self,
        stat: str,
        edges: tuple[float, ...],
        labels: tuple[str, ...],
        title: str | None = None,
    ) -> None:
        super().__init__(title or stat, edges, labels)
        self.stat = stat

    def values(self, catalog: pd.DataFrame) -> pd.Series:
        return catalog[stat_column(catalog, self.stat)]


class SpanCountAxis(CoverageAxis):
    """Axis over the span count of one feature group per query."""

    def __init__(
        self,
        group: FeatureGroup,
        edges: tuple[float, ...],
        labels: tuple[str, ...],
        title: str,
    ) -> None:
        super().__init__(title, edges, labels)
        self.group = group

    @property
    def y_excludes(self) -> frozenset[FeatureGroup]:
        return frozenset((self.group,))

    def values(self, catalog: pd.DataFrame) -> pd.Series:
        prefix = f"{self.group.value}."
        columns = [c for c in catalog.columns if c.startswith(prefix)]
        if not columns:
            raise ValueError(
                f"catalog has no {self.group.value} span columns"
            )
        return catalog[columns].sum(axis=1)


LENGTH_AXIS = StatAxis(
    "length_words",
    edges=(0, 3, 7, 10, 20, 60, float("inf")),
    labels=("0-2", "3-6", "7-9", "10-19", "20-59", "60+"),
)
IDENTIFIER_AXIS = SpanCountAxis(
    FeatureGroup.STRUCTURED_IDENTIFIERS,
    edges=(-0.5, 0.5, 1.5, 2.5, float("inf")),
    labels=("0", "1", "2", "3+"),
    title="identifier spans per query",
)
# parser hallucinates structure on non-sentences — read jointly with the
# natural_language_share axis (see SyntacticDepthBank)
DEPTH_AXIS = StatAxis(
    "nesting_depth",
    edges=(0, 2, 4, 6, float("inf")),
    labels=("0-1", "2-3", "4-5", "6+"),
)
NL_SHAPE_AXIS = StatAxis(
    "natural_language_share",
    edges=(0.0, 0.1, 0.25, 0.4, float("inf")),
    labels=("<0.1\ntelegram", "0.1-0.25", "0.25-0.4", "0.4+\nsentence"),
)
DEFAULT_AXES: tuple[CoverageAxis, ...] = (
    LENGTH_AXIS, IDENTIFIER_AXIS, DEPTH_AXIS, NL_SHAPE_AXIS,
)
