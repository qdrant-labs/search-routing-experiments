"""Numerical utilities shared by the fingerprint charts (heatmap, spider)."""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np

from query_taxonomy.reporting import CorpusReport


def build_matrix(
    reports: dict[str, CorpusReport],
    axes: Sequence[str],
    fetch: Callable[[CorpusReport, str], float],
) -> np.ndarray:
    """Rows = datasets in `reports` order, cols = `axes` in given order."""
    return np.array([
        [fetch(report, axis_name) for axis_name in axes]
        for report in reports.values()
    ], dtype=float)


def per_column_normalize(matrix: np.ndarray) -> np.ndarray:
    """Min-max per column. Flat columns (min==max) map to 0.5 so they
    render as mid-tone rather than confusingly saturated."""
    lo = matrix.min(axis=0)
    hi = matrix.max(axis=0)
    span = np.where(hi > lo, hi - lo, 1.0)
    normalized = (matrix - lo) / span
    normalized[:, hi == lo] = 0.5
    return normalized


def domain_value(report: CorpusReport, domain: str) -> float:
    return report.domain_query_share().get(domain, 0.0)


def stat_value(report: CorpusReport, stat: str) -> float:
    return report.stat_means().get(stat, 0.0)
