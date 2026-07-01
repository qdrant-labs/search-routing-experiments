from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import pandas as pd
from pydantic import BaseModel

from hybrid_search_rrf_dataset.golden import FusionRow

if TYPE_CHECKING:
    from matplotlib.figure import Figure


class ComparisonSummary(BaseModel):
    """Head-to-head numbers for a candidate fusion strategy vs a golden reference."""

    n_queries: int

    alpha_mae: float
    """mean absolute error between LLM alpha and golden alpha (lower is better)"""
    
    alpha_pearson: float
    """linear correlation (1.0 = perfect linear agreement, 0 = no relationship, negative = LLM predicts opposite direction)"""

    alpha_within_tolerance_pct: float
    """% of queries where LLM's alpha is within 0.1 of golden"""
    
    mean_metric_golden: float
    mean_metric_candidate: float

    mean_regret: float
    """ average NDCG lost by trusting LLM over golden. This is the killer metric. Close to 0 = LLM is nearly as good as knowing the optimum. Large = LLM is picking bad alphas."""
    max_regret: float


def _joined(
    golden: list[FusionRow], candidate: list[FusionRow]
) -> pd.DataFrame:
    ref = pd.DataFrame([r.model_dump() for r in golden]).set_index("query_id")
    cand = pd.DataFrame([r.model_dump() for r in candidate]).set_index("query_id")
    return ref[["alpha", "metric", "query"]].join(
        cand[["alpha", "metric"]], lsuffix="_golden", rsuffix="_candidate", how="inner"
    )


def compare(
    golden: list[FusionRow],
    candidate: list[FusionRow],
    alpha_tolerance: float = 0.1,
) -> ComparisonSummary:
    """Compare a candidate builder's rows against the golden (optimal) rows.

    Regret = golden.metric - candidate.metric. Positive means the candidate
    left NDCG on the table by picking a worse alpha.
    """
    df = _joined(golden, candidate)
    if df.empty:
        raise ValueError("No overlapping query_ids between golden and candidate.")

    alpha_delta = (df["alpha_candidate"] - df["alpha_golden"]).abs()
    regret = df["metric_golden"] - df["metric_candidate"]
    return ComparisonSummary(
        n_queries=len(df),
        alpha_mae=float(alpha_delta.mean()),
        alpha_pearson=float(df["alpha_golden"].corr(df["alpha_candidate"])),
        alpha_within_tolerance_pct=float(
            (alpha_delta <= alpha_tolerance).mean() * 100
        ),
        mean_metric_golden=float(df["metric_golden"].mean()),
        mean_metric_candidate=float(df["metric_candidate"].mean()),
        mean_regret=float(regret.mean()),
        max_regret=float(regret.max()),
    )


def plot_comparison(
    golden: list[FusionRow],
    candidate: list[FusionRow],
    candidate_label: str = "candidate",
) -> Figure:
    """Four-panel plot: alpha scatter, metric scatter, alpha delta, regret."""
    df = _joined(golden, candidate)
    if df.empty:
        raise ValueError("No overlapping query_ids between golden and candidate.")

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    ax = axes[0, 0]
    ax.scatter(df["alpha_golden"], df["alpha_candidate"], alpha=0.6)
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4, label="y=x")
    ax.set_xlabel("golden alpha (optimal)")
    ax.set_ylabel(f"{candidate_label} alpha")
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.set_title("Alpha agreement")
    ax.legend()

    ax = axes[0, 1]
    ax.scatter(df["metric_golden"], df["metric_candidate"], alpha=0.6)
    lim = max(df["metric_golden"].max(), df["metric_candidate"].max()) * 1.05
    ax.plot([0, lim], [0, lim], "k--", alpha=0.4, label="y=x")
    ax.set_xlabel("golden NDCG@10 (best possible)")
    ax.set_ylabel(f"{candidate_label} NDCG@10")
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_title("NDCG agreement")
    ax.legend()

    ax = axes[1, 0]
    delta = df["alpha_candidate"] - df["alpha_golden"]
    ax.hist(delta, bins=20, edgecolor="black")
    ax.axvline(0, color="k", linestyle="--", alpha=0.6)
    ax.set_xlabel(f"alpha delta ({candidate_label} - golden)")
    ax.set_ylabel("queries")
    ax.set_title(f"Alpha delta (MAE = {delta.abs().mean():.3f})")

    ax = axes[1, 1]
    regret = df["metric_golden"] - df["metric_candidate"]
    ax.hist(regret, bins=20, edgecolor="black")
    ax.axvline(0, color="k", linestyle="--", alpha=0.6)
    ax.set_xlabel(f"NDCG regret (golden - {candidate_label})")
    ax.set_ylabel("queries")
    ax.set_title(f"NDCG regret (mean = {regret.mean():.3f})")

    fig.tight_layout()
    return fig
