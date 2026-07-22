"""SPEC d28 spider comparison — 2-4 datasets overlaid, fixed axis order."""

from __future__ import annotations

import math
from collections.abc import Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from query_taxonomy.reporting import CorpusReport

from ._base import DOMAIN_PALETTE, ROUTER_SIGNAL_AXES, blank, make_axis
from ._matrix import build_matrix, per_column_normalize, stat_value
from .fingerprint_spider_grid import FingerprintSpiderGrid


class FingerprintSpider:
    """Polar overlay. Shape reads as relative position on each axis;
    raw ranges surface on tick labels so magnitudes stay visible."""

    def __init__(
        self,
        reports: dict[str, CorpusReport],
        axes: Sequence[str] = ROUTER_SIGNAL_AXES,
    ) -> None:
        if not 2 <= len(reports) <= 4:
            raise ValueError(
                f"spider takes 2-4 datasets; got {len(reports)}"
            )
        self.reports = reports
        self.axes = tuple(axes)
        self.labels = list(reports)
        self.raw = build_matrix(reports, self.axes, stat_value)
        self.normalized = per_column_normalize(self.raw)
        n = len(self.axes)
        self.angles = [i * 2 * math.pi / n for i in range(n)] + [0.0]

    def render(self, ax: plt.Axes | None = None) -> Figure:
        fig, polar = make_axis(ax, size=(6.5, 6.5), projection="polar")
        self._draw_overlays(polar)
        self._configure_axes(polar)
        polar.set_title(
            "Dataset fingerprint — per-axis min-max across the compared set",
            y=1.10, fontsize=11,
        )
        polar.legend(
            loc="upper right", bbox_to_anchor=(1.35, 1.10),
            frameon=False, fontsize=9,
        )
        return fig

    def _draw_overlays(self, polar: plt.Axes) -> None:
        for i, label in enumerate(self.labels):
            values = list(self.normalized[i]) + [self.normalized[i][0]]
            color = DOMAIN_PALETTE[i % len(DOMAIN_PALETTE)]
            polar.plot(
                self.angles, values, color=color, linewidth=1.8, label=label
            )
            polar.fill(self.angles, values, color=color, alpha=0.15)

    def _configure_axes(self, polar: plt.Axes) -> None:
        polar.set_theta_offset(math.pi / 2)
        polar.set_theta_direction(-1)
        polar.set_xticks(self.angles[:-1])
        polar.set_xticklabels(
            [self._axis_label(name, self.raw[:, j])
             for j, name in enumerate(self.axes)],
            fontsize=9,
        )
        polar.tick_params(pad=12)
        polar.set_ylim(0, 1)
        polar.set_yticks([0.25, 0.5, 0.75, 1.0])
        polar.set_yticklabels(["", "", "", ""], fontsize=7)
        polar.grid(True, alpha=0.3)

    @staticmethod
    def _axis_label(name: str, column: np.ndarray) -> str:
        lo, hi = column.min(), column.max()
        fmt = "{:.1f}" if hi >= 10 else "{:.2f}"
        return f"{name}\n[{fmt.format(lo)}, {fmt.format(hi)}]"


def fingerprint_spider(
    reports: dict[str, CorpusReport],
    axes: Sequence[str] = ROUTER_SIGNAL_AXES,
    ax: plt.Axes | None = None,
) -> Figure:
    """Dispatches by set size: 2-4 → overlay (single chart, storytelling
    view); 5+ → small-multiples grid (one mini per dataset, shared
    per-axis normalization). Under 2 → blank."""
    if len(reports) < 2:
        return blank(ax, "spider needs ≥2 datasets to compare")[0]
    if len(reports) <= 4:
        return FingerprintSpider(reports, axes).render(ax)
    if ax is not None:
        raise ValueError("spider grid renders its own figure; drop `ax`")
    return FingerprintSpiderGrid(reports, axes).render()
