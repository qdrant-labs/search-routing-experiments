"""Per-corpus horizontal bars over the four router signals."""

from __future__ import annotations

from collections.abc import Iterable

import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from query_taxonomy.reporting import CorpusReport

from ._base import DOMAIN_PALETTE, ROUTER_SIGNAL_AXES


def signal_bars(
    reports: dict[str, CorpusReport],
    signals: Iterable[str] = ROUTER_SIGNAL_AXES,
    ax: plt.Axes | None = None,
) -> Figure:
    """One panel per signal, horizontal bars across corpora. Y-axis
    carries dataset names in a shared order so rows align across the
    four panels; per-panel x-limits keep ratios (0-1) and counts (up to
    100s) from squashing each other. Raw values annotate every bar so
    outliers stay legible even when they squash the visual scale."""
    del ax  # signal_bars owns its own grid; can't share a single axis
    names = tuple(signals)
    labels = list(reports.keys())
    means = {
        name: [reports[label].stat_means().get(name, 0.0) for label in labels]
        for name in names
    }
    height = max(2.8, 0.28 * len(labels) + 1.2)
    fig, axes = plt.subplots(
        1, len(names), sharey=True,
        figsize=(3.5 * len(names), height),
    )
    if len(names) == 1:
        axes = [axes]
    for i, (axis, name) in enumerate(zip(axes, names, strict=True)):
        _draw_panel(axis, labels, means[name], name, i)
    fig.suptitle("Router signals — per-corpus means", y=1.01)
    fig.tight_layout()
    return fig


def _draw_panel(
    axis: plt.Axes,
    labels: list[str],
    values: list[float],
    name: str,
    color_index: int,
) -> None:
    color = DOMAIN_PALETTE[color_index % len(DOMAIN_PALETTE)]
    axis.barh(labels, values, color=color)
    axis.invert_yaxis()
    axis.set_title(name, fontsize=10)
    axis.tick_params(axis="y", labelsize=8)
    axis.tick_params(axis="x", labelsize=8)
    max_v = max(values) if values else 1.0
    pad = 0.02 * (max_v if max_v > 0 else 1.0)
    for j, v in enumerate(values):
        axis.text(v + pad, j, _fmt(v), va="center", fontsize=7)
    axis.margins(x=0.18)


def _fmt(v: float) -> str:
    return f"{v:.1f}" if v >= 10 else f"{v:.2f}"
