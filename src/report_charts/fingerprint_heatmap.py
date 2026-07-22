"""SPEC d28 catalog view — dataset × axis heatmap with per-column colors."""

from __future__ import annotations

from collections.abc import Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from query_taxonomy.reporting import CorpusReport
from query_taxonomy.taxonomy import Domain

from ._base import blank, make_axis
from ._matrix import build_matrix, domain_value, per_column_normalize, stat_value
from .equal_weight import EqualWeightScale

_DOMAIN_AXES: tuple[str, ...] = tuple(d.value for d in Domain)
_STAT_AXES: tuple[str, ...] = (
    "length_words", "length_chars", "stopword_ratio",
    "natural_language_share", "word_variation_share",
    "nesting_depth", "statement_count",
)


class FingerprintHeatmap:
    """Rows = datasets, cols = 8 domain query-shares + 7 stat means;
    per-column min-max colors so ratios and counts stay legible on the
    same chart."""

    def __init__(
        self,
        reports: dict[str, CorpusReport],
        domain_axes: Sequence[str] = _DOMAIN_AXES,
        stat_axes: Sequence[str] = _STAT_AXES,
        scale: EqualWeightScale | None = None,
    ) -> None:
        if not reports:
            raise ValueError("FingerprintHeatmap needs at least one dataset")
        self.rows = list(reports)
        self.domain_axes, self.stat_axes = tuple(domain_axes), tuple(stat_axes)
        self.cols = [*self.domain_axes, *self.stat_axes]
        self.n_domain_cols = len(self.domain_axes)
        domain_block = build_matrix(reports, self.domain_axes, domain_value)
        if scale is None:
            stat_block = build_matrix(reports, self.stat_axes, stat_value)
            self.raw = np.concatenate([domain_block, stat_block], axis=1)
            self.normalized = per_column_normalize(self.raw)
            self.scale_note = "color per column, raw value in cell"
        else:
            if scale.axes != self.stat_axes:
                raise ValueError(
                    f"scale fitted for {scale.axes}, chart wants {self.stat_axes}"
                )
            # domain shares are already comparable [0,1] rates — only the
            # stat block moves to the equal-weight percentile scale (d31)
            stat_block = scale.dataset_medians(self.rows)
            self.raw = np.concatenate([domain_block, stat_block], axis=1)
            self.normalized = np.concatenate([
                per_column_normalize(domain_block),
                scale.percentile_matrix(self.rows),
            ], axis=1)
            self.scale_note = (
                "domains min-max; stats equal-weight percentile, "
                "median in cell (d31)"
            )

    def render(self, ax: plt.Axes | None = None) -> Figure:
        height = max(2.2, 0.55 * len(self.rows) + 1.2)
        width = max(6.0, 0.55 * len(self.cols) + 2.0)
        fig, axis = make_axis(ax, size=(width, height))
        axis.imshow(self.normalized, aspect="auto", cmap="Blues", vmin=0, vmax=1)
        self._annotate(axis)
        self._configure_axes(axis)
        axis.set_title(f"Dataset fingerprints — {self.scale_note}")
        return fig

    def _annotate(self, axis: plt.Axes) -> None:
        rows, cols = self.raw.shape
        for i in range(rows):
            for j in range(cols):
                axis.text(
                    j, i, self._cell_text(i, j),
                    ha="center", va="center", fontsize=8,
                    color="white" if self._light_text(i, j) else "black",
                )

    def _cell_text(self, i: int, j: int) -> str:
        raw = self.raw[i, j]
        if j < self.n_domain_cols:
            return f"{raw:.2f}" if raw > 0 else "·"
        return f"{raw:.1f}" if raw >= 10 else f"{raw:.2f}"

    def _light_text(self, i: int, j: int) -> bool:
        return self.normalized[i, j] > 0.6

    def _configure_axes(self, axis: plt.Axes) -> None:
        axis.set_xticks(range(len(self.cols)))
        axis.set_xticklabels(self.cols, rotation=45, ha="right", fontsize=8)
        axis.set_yticks(range(len(self.rows)))
        axis.set_yticklabels(self.rows, fontsize=9)
        axis.axvline(x=self.n_domain_cols - 0.5, color="black", linewidth=1.2)
        n = self.n_domain_cols
        for x, label in (((n - 1) / 2, "domain query-share"),
                         (n + (len(self.cols) - n - 1) / 2, "stat means")):
            axis.text(x, -0.9, label, ha="center", va="bottom", fontsize=9)


def fingerprint_heatmap(
    reports: dict[str, CorpusReport],
    domain_axes: Sequence[str] = _DOMAIN_AXES,
    stat_axes: Sequence[str] = _STAT_AXES,
    ax: plt.Axes | None = None,
) -> Figure:
    """Functional shim — falls back to a blank chart on empty input so
    scripts don't need to guard the call."""
    if not reports:
        return blank(ax, "no profiles to compare")[0]
    return FingerprintHeatmap(reports, domain_axes, stat_axes).render(ax)
