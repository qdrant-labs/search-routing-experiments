"""Shared canvas + palette for report_charts.

Colorblind-safe qualitative palette (Wong 2011). All chart modules import
figure setup and constants from here so the visual grammar stays uniform
across donut, share-bar, signals, heatmap, and spider views.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.figure import Figure

DOMAIN_PALETTE: tuple[str, ...] = (
    "#0072B2", "#D55E00", "#009E73", "#CC79A7",
    "#F0E442", "#56B4E9", "#E69F00", "#999999",
)
"""Cycled slice-by-slice for donut/bar; also used to color spider overlays."""

# Fixed axis order (SPEC d28 arch-validator constraint): reshuffling spokes
# changes the perceived shape, so the canonical set is defined once here.
ROUTER_SIGNAL_AXES: tuple[str, ...] = (
    "natural_language_share",
    "word_variation_share",
    "nesting_depth",
    "length_words",
)


def palette_for(n: int) -> list[str]:
    """Cyclic palette lookup for `n` slices."""
    return [DOMAIN_PALETTE[i % len(DOMAIN_PALETTE)] for i in range(n)]


def make_axis(
    ax: plt.Axes | None,
    size: tuple[float, float],
    *,
    projection: str | None = None,
) -> tuple[Figure, plt.Axes]:
    """Uniform axis handling: reuse a caller-supplied axis, else create a
    fresh figure at the requested size (with an optional projection for
    polar/spider charts)."""
    if ax is not None:
        return ax.figure, ax
    fig = plt.figure(figsize=size)
    if projection:
        return fig, fig.add_subplot(111, projection=projection)
    return fig, fig.add_subplot(111)


def blank(
    ax: plt.Axes | None, message: str
) -> tuple[Figure, plt.Axes]:
    """Empty-state placeholder — used when a chart's input can't produce
    a meaningful figure (no spans, too few datasets to compare)."""
    fig, axis = make_axis(ax, size=(4, 3))
    axis.text(0.5, 0.5, message, ha="center", va="center", fontsize=10)
    axis.set_axis_off()
    return fig, axis
