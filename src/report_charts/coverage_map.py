"""SPEC d31 coverage view — the third fingerprint chart, over the d29
feature-table catalog (per-query rows, not per-dataset means).

Every query lands in one (x band × spans-per-query) cell; the chart maps
what to expect from the collected data before any routing algorithm
exists. x is a pluggable `CoverageAxis` — the canonical axes live in
`catalog_axes` (shared with the composition floors, d33b) so chart and
fill can never disagree on band edges. Cell color = log-scaled raw row
count — "can this cell reach a floor?"; cell text = raw count + the
equal-weight dominant source (which dataset characterizes the cell once
size is debiased). Empty cells get a red border: the holes are the
generation lane's order sheet."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle

from catalog_axes import LENGTH_AXIS, CoverageAxis
from query_taxonomy.taxonomy import FeatureGroup

from ._base import blank, make_axis

_SPAN_EDGES: tuple[float, ...] = (-0.5, 0.5, 1.5, 2.5, float("inf"))
_SPAN_LABELS: tuple[str, ...] = ("0 spans", "1 span", "2 spans", "3+ spans")


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
