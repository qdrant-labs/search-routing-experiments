"""The route vocabulary: the three routes the router may emit, their relative
serving cost and the tie tolerance. Kept apart from the fusion engine so
naming a route costs nothing but the stdlib."""

from __future__ import annotations

from enum import StrEnum


class StrategyName(StrEnum):
    """The router's decision surface — exactly three routes.

    Weighted variants (weighted_rrf, weighted_dbsf) were removed 2026-07-28
    per SPEC d37: a continuous dense/sparse weight is not a label the router
    can emit, so tuning one served no downstream consumer.
    """

    DENSE_ONLY = "dense_only"
    PURE_RRF = "pure_rrf"
    SPARSE_ONLY = "sparse_only"


TIE_TOLERANCE = 1e-9
"""Two route scores within this are the same score (SPEC d41e: ties are
exact; near-ties stay `routes_differ` with a thin margin)."""

SERVING_COST: dict[StrategyName, int] = {
    StrategyName.SPARSE_ONLY: 0,
    StrategyName.DENSE_ONLY: 1,
    StrategyName.PURE_RRF: 2,
}
"""Relative query-time cost (SPEC d41c). pure_rrf above each component is
structural — it runs both plus fusion; sparse < dense (no query-side
transformer pass) is an assumption until the latency benchmark lands. A flip
re-derives the route column in seconds via `RouteLabels.rederive`."""
