"""Per-corpus grouped bars over the four router signals."""

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
    """One panel per signal, grouped bars across corpora — the four
    router signals side by side, ready for eyeball comparison."""
    del ax  # signal_bars owns its own grid; can't share a single axis
    names = tuple(signals)
    labels = list(reports.keys())
    per_corpus = {label: reports[label].stat_means() for label in labels}
    means = {
        name: [per_corpus[label].get(name, 0.0) for label in labels]
        for name in names
    }
    fig, axes = plt.subplots(1, len(names), figsize=(3.0 * len(names), 3.2))
    if len(names) == 1:
        axes = [axes]
    for axis, name in zip(axes, names, strict=True):
        axis.bar(labels, means[name], color=DOMAIN_PALETTE[: len(labels)])
        axis.set_title(name, fontsize=10)
        axis.tick_params(axis="x", rotation=25, labelsize=8)
        axis.margins(y=0.15)
    fig.suptitle("Router signals — per-corpus means", y=1.02)
    fig.tight_layout()
    return fig
