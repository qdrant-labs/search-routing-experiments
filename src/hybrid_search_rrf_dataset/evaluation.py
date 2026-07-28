from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import pandas as pd
from pydantic import BaseModel

from hybrid_search_rrf_dataset.golden import FusionRow

if TYPE_CHECKING:
    from matplotlib.figure import Figure


class ComparisonSummary(BaseModel):
    """Head-to-head numbers for a candidate router vs the oracle reference.

    Metric-agnostic: `metric_name` carries whatever the rows were scored with
    and every regret number is in that metric's units. Compare against the
    constant-route baselines too, not just the oracle — a candidate can post
    respectable regret while still losing to always picking one route
    (SPEC d37h).
    """

    n_queries: int
    metric_name: str

    mean_metric_golden: float
    """average oracle metric — the attainable ceiling for this strategy"""

    mean_metric_candidate: float

    mean_regret: float
    """average metric lost vs the oracle. The killer metric: ~0 = candidate is
    nearly as good as knowing the per-query optimum."""

    median_regret: float
    """robust central tendency — unlike the mean, not dragged by a few
    catastrophic queries"""

    p90_regret: float
    """how bad the worst decile of queries gets"""

    oracle_hit_rate_pct: float
    """% of queries where the candidate reaches the golden metric (within
    epsilon). Outcome-space, so picking a *different* route that scores just as
    well counts as a hit — as it should."""

    route_agreement_pct: float
    """% of queries where the candidate picked the oracle's route. Weaker than
    `oracle_hit_rate_pct` as a quality signal (a tie broken the other way reads
    as disagreement) but it is the quantity the router is trained to predict."""


def _single_metric_name(rows: Iterable[FusionRow]) -> str:
    names = {r.metric_name for r in rows}
    if len(names) != 1:
        raise ValueError(
            f"Rows must share a single metric to be comparable, got {sorted(names)}."
        )
    return names.pop()


def _joined(
    golden: list[FusionRow], candidate: list[FusionRow]
) -> pd.DataFrame:
    keys = ["dataset_name", "query_id"]
    ref = pd.DataFrame([r.model_dump() for r in golden]).set_index(keys)
    cand = pd.DataFrame([r.model_dump() for r in candidate]).set_index(keys)
    return ref[["metric", "query", "strategy_name"]].join(
        cand[["metric", "strategy_name"]],
        lsuffix="_golden",
        rsuffix="_candidate",
        how="inner",
    )


def compare(
    golden: list[FusionRow],
    candidate: list[FusionRow],
    regret_epsilon: float = 1e-6,
) -> ComparisonSummary:
    """Compare a candidate builder's rows against the golden (optimal) rows.

    Regret = golden.metric - candidate.metric, in units of the rows' shared
    metric. Positive means the candidate left quality on the table; it cannot go
    negative, since the golden rows take the max over the same three routes the
    candidate chose from.
    """
    metric_name = _single_metric_name([*golden, *candidate])
    df = _joined(golden, candidate)
    if df.empty:
        raise ValueError("No overlapping (dataset_name, query_id) between golden and candidate.")

    regret = df["metric_golden"] - df["metric_candidate"]
    return ComparisonSummary(
        n_queries=len(df),
        metric_name=metric_name,
        mean_metric_golden=float(df["metric_golden"].mean()),
        mean_metric_candidate=float(df["metric_candidate"].mean()),
        mean_regret=float(regret.mean()),
        median_regret=float(regret.median()),
        p90_regret=float(regret.quantile(0.9)),
        oracle_hit_rate_pct=float((regret <= regret_epsilon).mean() * 100),
        route_agreement_pct=float(
            (df["strategy_name_golden"] == df["strategy_name_candidate"]).mean() * 100
        ),
    )


def plot_comparison(
    golden: list[FusionRow],
    candidate: list[FusionRow],
    candidate_label: str = "candidate",
) -> Figure:
    """Three panels: metric agreement, regret, and the route confusion matrix."""
    metric_name = _single_metric_name([*golden, *candidate])
    df = _joined(golden, candidate)
    if df.empty:
        raise ValueError("No overlapping (dataset_name, query_id) between golden and candidate.")

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

    ax = axes[0]
    ax.scatter(df["metric_golden"], df["metric_candidate"], alpha=0.6)
    lim = max(df["metric_golden"].max(), df["metric_candidate"].max()) * 1.05
    ax.plot([0, lim], [0, lim], "k--", alpha=0.4, label="y=x")
    ax.set_xlabel(f"oracle {metric_name} (best possible)")
    ax.set_ylabel(f"{candidate_label} {metric_name}")
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_title(f"{metric_name} agreement")
    ax.legend()

    ax = axes[1]
    regret = df["metric_golden"] - df["metric_candidate"]
    ax.hist(regret, bins=20, edgecolor="black")
    ax.axvline(0, color="k", linestyle="--", alpha=0.6)
    ax.set_xlabel(f"{metric_name} regret (oracle - {candidate_label})")
    ax.set_ylabel("queries")
    ax.set_title(f"{metric_name} regret (mean = {regret.mean():.3f})")

    ax = axes[2]
    routes = sorted(
        set(df["strategy_name_golden"]) | set(df["strategy_name_candidate"])
    )
    counts = pd.crosstab(
        df["strategy_name_golden"], df["strategy_name_candidate"]
    ).reindex(index=routes, columns=routes, fill_value=0)
    ax.imshow(counts.to_numpy(), cmap="Blues", aspect="auto")
    peak = counts.to_numpy().max()
    for i, row in enumerate(counts.to_numpy()):
        for j, n in enumerate(row):
            ax.text(
                j, i, str(n), ha="center", va="center",
                color="white" if n > peak * 0.5 else "black",
            )
    ax.set_xticks(range(len(routes)), routes, rotation=45, ha="right")
    ax.set_yticks(range(len(routes)), routes)
    ax.set_xlabel(f"{candidate_label} route")
    ax.set_ylabel("oracle route")
    agreement = (
        df["strategy_name_golden"] == df["strategy_name_candidate"]
    ).mean() * 100
    ax.set_title(f"Route choice (agreement = {agreement:.0f}%)")

    fig.tight_layout()
    return fig
