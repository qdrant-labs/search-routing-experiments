"""SPEC d27 two-ring donut with certified/-like honesty stripe."""

from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.patches import Wedge

from query_taxonomy.reporting import CorpusReport, DomainSlice
from query_taxonomy.taxonomy import FeatureGroup

from ._base import blank, make_axis, palette_for

_ASSUMED_ALPHA, _TINY_SLICE_THRESHOLD = 0.55, 0.02
_GROUP_TAG: dict[FeatureGroup, str] = {
    FeatureGroup.SENTENCE_MARKERS: "marker",
    FeatureGroup.LOGICAL_STRUCTURES: "logical",
}


def feature_donut(
    report: CorpusReport,
    title: str | None = None,
    ax: plt.Axes | None = None,
) -> Figure:
    """Inner ring sized by span mass over every span-emitting group
    (identifier domains with `general` exploded, plus marker/logical
    banks). Outer ring splits certified vs shape-guesses (d27)."""
    slices = report.feature_rollup()
    if not slices:
        fig, axis = blank(ax, "no spans in this corpus")
        if title:
            axis.set_title(title)
        return fig

    fig, axis = make_axis(ax, size=(7.5, 6))
    colors = palette_for(len(slices))
    values = [s.spans for s in slices]
    total = sum(values)
    on_chart = [
        s.label if v / total >= _TINY_SLICE_THRESHOLD else ""
        for s, v in zip(slices, values, strict=True)
    ]
    axis.pie(
        values, radius=1.0, colors=colors, labels=on_chart,
        labeldistance=1.08,
        wedgeprops={"width": 0.35, "edgecolor": "white"},
        startangle=90, counterclock=False,
        textprops={"fontsize": 9},
    )
    _draw_outer_ring(axis, slices, colors)
    axis.set(aspect="equal")
    axis.set_title(
        title or f"Query features ({total:,} spans, striped = shape-guesses)"
    )
    _draw_legend(axis, slices, colors, total)
    return fig


def _draw_outer_ring(
    axis: plt.Axes, slices: list[DomainSlice], colors: list[str]
) -> None:
    """Certified block solid, shape-guess block dimmed and hatched."""
    total = sum(s.spans for s in slices)
    theta = 90.0
    for piece, color in zip(slices, colors, strict=True):
        arc = 360.0 * piece.spans / total
        cert_arc = 360.0 * piece.certified / total if total else 0.0
        if cert_arc:
            axis.add_patch(Wedge(
                (0, 0), 1.0, theta - cert_arc, theta,
                width=0.18, facecolor=color, edgecolor="white",
            ))
        if arc - cert_arc:
            axis.add_patch(Wedge(
                (0, 0), 1.0, theta - arc, theta - cert_arc,
                width=0.18, facecolor=color,
                alpha=_ASSUMED_ALPHA, hatch="//", edgecolor="white",
            ))
        theta -= arc


def _draw_legend(
    axis: plt.Axes, slices: list[DomainSlice], colors: list[str], total: int
) -> None:
    handles, labels = [], []
    for piece, color in zip(slices, colors, strict=True):
        share = 100 * piece.spans / total if total else 0.0
        guess = 100 * piece.assumed / piece.spans if piece.spans else 0.0
        tag = _GROUP_TAG.get(piece.group)
        suffix = f", {guess:.0f}% guesses" if guess else ""
        handles.append(plt.Rectangle((0, 0), 1, 1, color=color))
        labels.append(
            f"{piece.label}{f' [{tag}]' if tag else ''} — "
            f"{piece.spans} ({share:.1f}%){suffix}"
        )
    axis.legend(
        handles, labels, loc="center left", bbox_to_anchor=(1.05, 0.5),
        frameon=False, fontsize=8,
    )
