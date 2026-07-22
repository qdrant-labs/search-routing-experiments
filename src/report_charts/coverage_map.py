"""SPEC d31 coverage view — the third fingerprint chart, over the d29
feature-table catalog (per-query rows, not per-dataset means).

Every query lands in one (x band × spans-per-query) cell; the chart maps
what to expect from the collected data before any routing algorithm
exists. x is a pluggable `CoverageAxis` — length_words by default, with
identifier count / nesting_depth / natural_language_share as the other
stock axes (`DEFAULT_AXES`, drawn together by coverage_map_grid). Cell
color = log-scaled raw row count — "can this cell reach a floor?"; cell
text = raw count + the equal-weight dominant source (which dataset
characterizes the cell once size is debiased). Empty cells get a red
border: the holes are the generation lane's order sheet.

NOTE: the production router's classify() score 0..9 is a sparse-vs-dense
GRADE, not a word count — never draw length bands as router bands (an
earlier revision did)."""

from __future__ import annotations

from abc import ABC, abstractmethod

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle

from query_taxonomy.taxonomy import FeatureGroup

from ._base import blank, make_axis
from .equal_weight import stat_column

_SPAN_EDGES: tuple[float, ...] = (-0.5, 0.5, 1.5, 2.5, float("inf"))
_SPAN_LABELS: tuple[str, ...] = ("0 spans", "1 span", "2 spans", "3+ spans")


class CoverageAxis(ABC):
    """One x-axis candidate: how to read a per-query value off the
    catalog and how to band it into cells."""

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
        """Span groups the y-axis must not count under this x — an axis
        that is itself a span count would correlate with y by
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
    """x = one scalar stat (a `<bank>.<stat>` catalog column)."""

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
    """x = span count of one feature group per query; y then counts only
    the remaining groups."""

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
# natural_language_share panel (see SyntacticDepthBank)
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


class CoverageMap:
    """Per-query cell counts: rows of the catalog binned into
    (x band × span bucket) cells, with equal-weight dominance."""

    def __init__(
        self, catalog: pd.DataFrame, x_axis: CoverageAxis = LENGTH_AXIS,
    ) -> None:
        if catalog.empty:
            raise ValueError("CoverageMap needs a non-empty catalog")
        self.x_axis = x_axis
        prefixes = tuple(
            f"{group.value}."
            for group in FeatureGroup
            if group is not FeatureGroup.STATISTICAL_METRICS
            and group not in x_axis.y_excludes
        )
        span_columns = [
            c for c in catalog.columns if c.startswith(prefixes)
        ]
        sizes = catalog["dataset"].value_counts()
        cells = pd.DataFrame({
            "x": x_axis.bands(catalog),
            "y": pd.cut(
                catalog[span_columns].sum(axis=1),
                _SPAN_EDGES, labels=_SPAN_LABELS,
            ),
            "dataset": catalog["dataset"],
            "weight": 1.0 / catalog["dataset"].map(sizes),
        })
        self.counts = (
            cells.groupby(["y", "x"], observed=False)
            .size()
            .unstack()
            .reindex(index=_SPAN_LABELS, columns=x_axis.labels)
            .fillna(0)
            .astype(int)
        )
        dominant_weight = (
            cells.groupby(["y", "x", "dataset"], observed=False)["weight"]
            .sum()
            .unstack()
        )
        self.dominant = dominant_weight.idxmax(axis=1).unstack().reindex(
            index=_SPAN_LABELS, columns=x_axis.labels
        )

    def render(self, ax: plt.Axes | None = None) -> Figure:
        fig, axis = make_axis(ax, size=(11.0, 5.0))
        values = self.counts.to_numpy(dtype=float)
        axis.imshow(
            np.log10(values + 1), aspect="auto", cmap="Blues",
            origin="lower",
        )
        self._annotate(axis, values)
        self._configure_axes(axis)
        return fig

    def _annotate(self, axis: plt.Axes, values: np.ndarray) -> None:
        rows, cols = values.shape
        top = np.log10(values + 1).max() or 1.0
        for i in range(rows):
            for j in range(cols):
                count = int(values[i, j])
                if count == 0:
                    axis.add_patch(Rectangle(
                        (j - 0.5, i - 0.5), 1, 1,
                        fill=False, edgecolor="#D55E00", linewidth=1.8,
                    ))
                    axis.text(j, i, "empty", ha="center", va="center",
                              fontsize=8, color="#D55E00")
                    continue
                dominant = _short(str(self.dominant.iloc[i, j]))
                axis.text(
                    j, i, f"{count:,}\n{dominant}",
                    ha="center", va="center", fontsize=7.5,
                    color="white"
                    if np.log10(count + 1) / top > 0.6 else "black",
                )

    def _configure_axes(self, axis: plt.Axes) -> None:
        axis.set_xticks(range(len(self.x_axis.labels)))
        axis.set_xticklabels(self.x_axis.labels, fontsize=9)
        axis.set_yticks(range(len(_SPAN_LABELS)))
        axis.set_yticklabels(_SPAN_LABELS, fontsize=9)
        axis.set_xlabel(self.x_axis.title, fontsize=9)
        axis.set_ylabel(self._ylabel(), fontsize=9)
        axis.set_title(
            f"Coverage by {self.x_axis.title} — catalog rows per cell "
            "(log color), equal-weight dominant source in cell (SPEC d31)"
        )

    def _ylabel(self) -> str:
        if not self.x_axis.y_excludes:
            return "feature evidence per query"
        excluded = ", ".join(
            sorted(g.value for g in self.x_axis.y_excludes)
        )
        return f"feature evidence per query (excl. {excluded})"


def _short(name: str, limit: int = 16) -> str:
    """Cell-width guard: long dataset names bleed across columns."""
    return name if len(name) <= limit else name[: limit - 1] + "…"


def coverage_map(
    catalog: pd.DataFrame,
    ax: plt.Axes | None = None,
    x_axis: CoverageAxis = LENGTH_AXIS,
) -> Figure:
    """Functional shim — blank chart on an empty catalog."""
    if catalog.empty:
        return blank(ax, "no catalog rows to map")[0]
    return CoverageMap(catalog, x_axis).render(ax)
