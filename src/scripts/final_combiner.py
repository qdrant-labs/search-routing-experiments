"""The final combiner: per route, max(l1, l2) + alpha*llm_a + beta*r1, then argmax.
Plus a trust attribution (which source actually decided the label) and an alpha/beta
sweep — the goal is a decisive label on every row, read against how grounded it is."""
from __future__ import annotations

import numpy as np
import pandas as pd

ROUTES = ("dense_only", "sparse_only", "pure_rrf")
TOL = 1e-9

GEMINI_PRICE_PER_1M = 0.15  # openrouter google/gemini-embedding-001 (fetched 2026-08-31)
DENSE_QUERY_ROUTES = ("dense_only", "pure_rrf")  # sparse_only is local bm25 — no gemini call


def gemini_l2_cost(rows: list[dict], *, price_per_1m: float = GEMINI_PRICE_PER_1M) -> float:
    """The L2 query-embedding spend Qdrant's cloud_inference response never surfaces: each
    row embeds its query once per dense-bearing route through gemini-embedding-001. Doc-side
    indexing was the one-time bakeoff pass, amortized over the whole lane, not charged per
    label. Tokens estimated at 4 chars/token (English), so this is an estimate, not a bill."""
    chars = sum(len(row["query"]) for row in rows)
    tokens = chars / 4 * len(DENSE_QUERY_ROUTES)
    return tokens / 1e6 * price_per_1m


def _picked(scores: dict) -> str | None:
    """The route strictly above the next-best (None on a three-way tie or all-zero —
    position in ROUTES is not a decision)."""
    ordered = sorted(scores, key=scores.get, reverse=True)
    top = scores[ordered[0]]
    if top <= TOL or top - scores[ordered[1]] <= TOL:
        return None
    return ordered[0]


def combine(row: dict, alpha: float, beta: float) -> tuple[str | None, float, dict]:
    """`row` carries per-route dicts `l1`, `l2`, `r1`, `llm_a` (keys = ROUTES).
    Returns (winner_or_None, margin, combined_scores). Winner is None when no route
    is strictly ahead — a tie is not a label."""
    combined = {r: max(row["l1"][r], row["l2"][r]) + alpha * row["llm_a"][r] + beta * row["r1"][r]
                for r in ROUTES}
    ordered = sorted(combined.values(), reverse=True)
    return _picked(combined), ordered[0] - ordered[1], combined


def attribute(row: dict, alpha: float, beta: float, winner: str | None) -> str:
    """Trust tier = the minimal source set whose UNIQUE winner already matches the final one:
      - 'measurement' : max(l1, l2) alone picks the winner (verified, highest trust)
      - 'r1_assisted' : measurement + beta*r1 picks it (doc-checkable opinion)
      - 'llm_a_only'  : only the full combined pick agrees (pure opinion, lowest trust)
      - 'ungrounded'  : no source produces a unique winner (three-way tie or all-zero)
    Margin is reported separately (see `sweep`) as the quality gauge, not the tier."""
    if winner is None:
        return "ungrounded"
    meas = {r: max(row["l1"][r], row["l2"][r]) for r in ROUTES}
    if _picked(meas) == winner:
        return "measurement"
    mr1 = {r: meas[r] + beta * row["r1"][r] for r in ROUTES}
    if _picked(mr1) == winner:
        return "r1_assisted"
    return "llm_a_only"


def sweep(rows: list[dict], alphas, betas, *, class_margin: float = 0.1) -> pd.DataFrame:
    """Per (alpha, beta): decisive share (a real, grounded-or-not winner), grounded share
    (measurement + r1_assisted), the per-tier mix, and median margin. The 'knee' is the
    smallest (alpha, beta) reaching decisive == 1.0 while maximising grounded."""
    n = len(rows)
    out = []
    for a in alphas:
        for b in betas:
            tiers, margins = [], []
            for row in rows:
                w, m, _ = combine(row, a, b)
                tiers.append(attribute(row, a, b, w))
                margins.append(m)
            counts = {t: tiers.count(t) for t in ("measurement", "r1_assisted", "llm_a_only", "ungrounded")}
            out.append({
                "alpha": a, "beta": b,
                "decisive": (n - counts["ungrounded"]) / n,
                "grounded": (counts["measurement"] + counts["r1_assisted"]) / n,
                "measurement": counts["measurement"] / n,
                "r1_assisted": counts["r1_assisted"] / n,
                "llm_a_only": counts["llm_a_only"] / n,
                "certifiable": float(np.mean([m >= class_margin for m in margins])),
                "median_margin": float(np.median(margins)),
            })
    return pd.DataFrame(out)


def project_to_router_frame(rows: list[dict], *, alpha: float, beta: float) -> pd.DataFrame:
    """Project combiner outputs into the router's native score-triple schema so the frozen
    router (which derives winner/margin from `score_*` columns) consumes them as ordinary
    rows — no router change. Per-route combined scores are normalized to [0,1] by
    /(1+alpha+beta) so the router's [0,1]-calibrated decisive margin gates them consistently
    with OG rows. Carries `combiner_trust` for the include/exclude-opinion ablation and
    `bucket_origin` for provenance. A fused TIE (combine returns None with a nonzero top) is
    resolved to the `pure_rrf` hedge — the combiner's documented uncertainty route — so no
    decidable row is dropped; only a genuine all-zero row (no signal at all) is skipped, having
    nothing to label. TRAINING ONLY: the normalized scores are not real objective values, so
    these rows must never enter eval."""
    denom = 1.0 + alpha + beta
    out: list[dict] = []
    for r in rows:
        winner, _, combined = combine(r, alpha, beta)
        tie_hedged = False
        if winner is None:
            if max(combined.values()) <= TOL:
                continue  # all-zero: no signal fired, nothing to label
            winner, tie_hedged = "pure_rrf", True  # fused tie -> rrf hedge
        scores = {route: combined[route] / denom for route in ROUTES}
        if tie_hedged:
            # make the hedge the strict argmax so the router reads it as the label
            # (must beat the runner-up by > the routes_differ tolerance of 1e-9)
            scores[winner] = max(scores.values()) + 1e-6
        row: dict[str, object] = {
            "dataset": r["dataset"],
            "query_id": r["query_id"],
            "query": r["query"],
            "shape": "routes_differ",  # a strict winner by construction (tie -> rrf hedge)
            "bucket_origin": r.get("bucket"),
            "combiner_trust": "llm_a_only" if tie_hedged else attribute(r, alpha, beta, winner),
        }
        for route in ROUTES:
            row[f"score_{route}"] = scores[route]
        out.append(row)
    return pd.DataFrame(out)
