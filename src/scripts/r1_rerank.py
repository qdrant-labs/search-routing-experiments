"""R1 reranker calls + per-route rescoring, extracted from r1_reranker_pilot so the
final combiner round and the pilot share one implementation (not two copies)."""
from __future__ import annotations

import os
import time

import requests

from hybrid_search_rrf_dataset.objective import RouterObjective

RERANK_URL = "https://openrouter.ai/api/v1/rerank"
RERANK_MODEL = "voyageai/rerank-2.5-lite"
MAX_DOCUMENT_CHARS = 40_000  # provider rejects query/document pairs above ~10,240 tokens


def _key() -> str:
    key = os.environ.get("OPENROUTER_API_KEY") or os.environ.get("OPEN_ROUTER_API_KEY")
    if not key:
        raise RuntimeError("no OpenRouter key in env (OPENROUTER_API_KEY / OPEN_ROUTER_API_KEY)")
    return key


def rerank(query: str, documents: list[str], *, retries: int = 5,
           model: str = RERANK_MODEL, max_chars: int = MAX_DOCUMENT_CHARS) -> tuple[list[float], dict]:
    """Relevance scores aligned to `documents`' INPUT order (the endpoint returns them
    reordered by relevance with an `index` back-pointer; undone here). Burst-rate-limited
    tiers 429 under load with no `Retry-After` — exponential backoff. Returns (scores, usage).
    `max_chars` truncates each document before sending — the cost lever, since a reranker
    bills every query×document token pair (default is the provider's ~10,240-token ceiling)."""
    docs = [str(d)[:max_chars] for d in documents]
    key = _key()
    for attempt in range(retries):
        r = requests.post(
            RERANK_URL,
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
            json={"model": model, "query": query, "documents": docs, "top_n": len(docs)},
            timeout=30,
        )
        if r.status_code == 429 and attempt < retries - 1:
            time.sleep(3.0 * 2**attempt)
            continue
        r.raise_for_status()
        body = r.json()
        scores = [0.0] * len(docs)
        for item in body["results"]:
            scores[item["index"]] = item["relevance_score"]
        return scores, body.get("usage", {})
    raise RuntimeError("rerank: exhausted retries on 429")


def r1_route_scores(doc_scores: dict[str, float], judged: list[str],
                    rankings: dict[str, list[str]], min_relevance: int) -> dict[str, float]:
    """Per-route score under R1-augmented relevance, via the real objective (not a
    reimplementation). `doc_scores`: {doc_id: rerank_score} over the union; `judged`: known-
    relevant doc_ids; `rankings`: {route: [doc_id, ...]}. A union doc is R1-relevant if it
    beats the LOWEST judged doc's score (self-calibrating per-query floor — provisional, NOT
    A11 calibration; `min(judged)` is the fragile choice the plan flags). Returns
    {route: 0.7*HitRate@1 + 0.3*NDCG@10}."""
    judged_scores = [doc_scores[d] for d in judged if d in doc_scores]
    floor = min(judged_scores) if judged_scores else float("inf")
    r1_relevant = set(judged) | {d for d, s in doc_scores.items() if s > floor}
    gold = {d: 1 for d in r1_relevant}
    objective = RouterObjective(min_relevance=min_relevance)
    out: dict[str, float] = {}
    for route, ranked in rankings.items():
        ranking = {d: float(len(ranked) - i) for i, d in enumerate(ranked)}
        out[route], _ = objective.assess(ranking, gold)
    return out
