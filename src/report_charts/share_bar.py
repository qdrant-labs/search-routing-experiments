"""Companion bar for the donut: per-domain query share (SPEC d27)."""

from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from query_taxonomy.reporting import CorpusReport
from query_taxonomy.taxonomy import Domain

from ._base import DOMAIN_PALETTE, blank, make_axis, palette_for


def domain_share_bar(
    report: CorpusReport,
    title: str | None = None,
    ax: plt.Axes | None = None,
) -> Figure:
    """Per-domain share of queries touching that domain. Aggregates
    `general` (unlike the donut) — one query may touch several domains,
    so this doesn't sum to 100%. Colors align to the donut so the two
    charts read as a pair."""
    shares = report.domain_query_share()
    fig, axis = make_axis(ax, size=(6, 3.5))
    if not shares:
        return blank(ax, "no identifier spans in this corpus")[0]

    donut_palette = _label_to_color(report)
    labels = list(shares.keys())
    values = [100 * v for v in shares.values()]
    colors = [donut_palette.get(label, DOMAIN_PALETTE[0]) for label in labels]
    axis.barh(labels, values, color=colors)
    axis.invert_yaxis()
    axis.set_xlabel("% of queries carrying at least one span")
    axis.set_title(title or "Domain query share")
    for i, v in enumerate(values):
        axis.text(v + 0.3, i, f"{v:.1f}%", va="center", fontsize=8)
    axis.margins(x=0.15)
    return fig


def _label_to_color(report: CorpusReport) -> dict[str, str]:
    """Reproduce the donut's label→color mapping. `general` inherits the
    color of its top exploded slice — a small honest inconsistency
    (share bar aggregates, donut explodes) made visually gentler by hue
    continuity."""
    slices = report.domain_rollup()
    palette = palette_for(len(slices))
    domain_labels = {d.value for d in Domain}
    mapping: dict[str, str] = {}
    for piece, color in zip(slices, palette, strict=True):
        mapping.setdefault(piece.label, color)
        if piece.label not in domain_labels:
            mapping.setdefault(Domain.GENERAL.value, color)
    return mapping
