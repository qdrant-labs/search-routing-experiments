"""Target composition — one horizontal stacked bar, 50K.

Macro split 60/20/20 (span-evidence, statistical strata, dark forest);
inside span-evidence, the 80/20 sub-split (type-targeted vs
no-preference) reads as a solid + lighter-shade pair of the same color.
The dark-forest slice carries an inline marker at x = 50% of its width,
so the ≤50%-per-champion cap is geometrically visible rather than
buried in a caption."""

from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle

from ._base import DOMAIN_PALETTE, make_axis

TOTAL = 50_000
SPAN_EVIDENCE = 30_000
SPAN_TYPE_TARGETED = 24_000
SPAN_NO_PREFERENCE = 6_000
STAT_STRATA = 10_000
DARK_FOREST = 10_000
DARK_CHAMPION_CAP = 0.50
DARK_MIN_CHAMPIONS = 3

_SPAN_COLOR = DOMAIN_PALETTE[0]
_STAT_COLOR = DOMAIN_PALETTE[1]
_DARK_COLOR = DOMAIN_PALETTE[2]
_SUBSHADE_ALPHA = 0.55

_BAR_Y = 0.40
_BAR_HEIGHT = 0.55


def target_composition(ax: plt.Axes | None = None) -> Figure:
    """Render the d32 target as one stacked bar; returns the figure."""
    fig, axis = make_axis(ax, size=(11, 3))

    _segment(axis, 0, SPAN_TYPE_TARGETED, _SPAN_COLOR)
    _segment(
        axis, SPAN_TYPE_TARGETED, SPAN_NO_PREFERENCE,
        _SPAN_COLOR, alpha=_SUBSHADE_ALPHA,
    )
    _segment(axis, SPAN_EVIDENCE, STAT_STRATA, _STAT_COLOR)
    _segment(axis, SPAN_EVIDENCE + STAT_STRATA, DARK_FOREST, _DARK_COLOR)
    _draw_champion_cap(axis)

    _macro_label(axis, 0, SPAN_EVIDENCE, "SPAN-EVIDENCE", "30K · 60%")
    _macro_label(axis, SPAN_EVIDENCE, STAT_STRATA, "STAT STRATA", "10K · 20%")
    _macro_label(
        axis, SPAN_EVIDENCE + STAT_STRATA, DARK_FOREST,
        "DARK FOREST", "10K · 20%",
    )

    _sublabel(axis, 0, SPAN_TYPE_TARGETED,
              "type-targeted", "24K · 80% of slice")
    _sublabel(axis, SPAN_TYPE_TARGETED, SPAN_NO_PREFERENCE,
              "no-preference", "6K · 20%")
    _sublabel(axis, SPAN_EVIDENCE, STAT_STRATA,
              "zero-span rows", "scalar-signal bands")
    _sublabel(axis, SPAN_EVIDENCE + STAT_STRATA, DARK_FOREST,
              "feature-blind", f"≥{DARK_MIN_CHAMPIONS} champions")

    axis.set_xlim(0, TOTAL)
    axis.set_ylim(-0.30, 1.20)
    axis.set_xticks([])
    axis.set_yticks([])
    for spine in axis.spines.values():
        spine.set_visible(False)
    axis.set_title(
        "Target composition — SPEC d32 (50K queries)",
        fontsize=11, y=1.02,
    )
    fig.tight_layout()
    return fig


def _segment(
    axis: plt.Axes, x: float, w: float, color: str,
    *, alpha: float = 1.0,
) -> None:
    axis.add_patch(Rectangle(
        (x, _BAR_Y), w, _BAR_HEIGHT,
        facecolor=color, edgecolor="white", linewidth=1.5, alpha=alpha,
    ))


def _draw_champion_cap(axis: plt.Axes) -> None:
    """Dashed marker at the ≤50% single-champion line, with an inline
    label so the constraint reads without a legend."""
    dark_start = SPAN_EVIDENCE + STAT_STRATA
    cap_x = dark_start + DARK_FOREST * DARK_CHAMPION_CAP
    axis.plot(
        [cap_x, cap_x], [_BAR_Y, _BAR_Y + _BAR_HEIGHT],
        color="white", linewidth=1.5, linestyle=(0, (2, 2)),
    )
    axis.text(
        cap_x, _BAR_Y + _BAR_HEIGHT + 0.03,
        f"≤{int(DARK_CHAMPION_CAP * 100)}% cap",
        ha="center", va="bottom", fontsize=7, color="#444",
    )


def _macro_label(
    axis: plt.Axes, x: float, w: float, top: str, bottom: str,
) -> None:
    cx = x + w / 2
    axis.text(
        cx, _BAR_Y + _BAR_HEIGHT * 0.65, top,
        ha="center", va="center",
        fontsize=10, color="white", weight="bold",
    )
    axis.text(
        cx, _BAR_Y + _BAR_HEIGHT * 0.32, bottom,
        ha="center", va="center", fontsize=9, color="white",
    )


def _sublabel(
    axis: plt.Axes, x: float, w: float, top: str, bottom: str,
) -> None:
    cx = x + w / 2
    axis.text(
        cx, _BAR_Y - 0.04, top,
        ha="center", va="top", fontsize=8, color="#333",
    )
    axis.text(
        cx, _BAR_Y - 0.16, bottom,
        ha="center", va="top", fontsize=7, color="#666",
    )
