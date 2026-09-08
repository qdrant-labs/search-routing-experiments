"""LLM A (route-preference) calls, parsing, and route-vectorisation, extracted from
llm_a_pilot so the final combiner round and the pilot share one implementation."""
from __future__ import annotations

import os
import re

import requests

COMPLETIONS_URL = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_MODEL = "openai/gpt-5.6-luna"
ROUTE_MAP = {"dense": "dense_only", "sparse": "sparse_only", "hybrid": "pure_rrf"}

INSTRUCTION = (
    "Predict whether dense or sparse retrieval has the stronger retrieval advantage for this "
    "specific query against this specific document collection.\n\n"

    "The two strategies are:\n"
    "- dense: semantic embedding retrieval\n"
    "- sparse: BM25 lexical retrieval\n\n"

    "The known answer document is the GOLD top-1 target: it is the document that ideally should "
    "appear at rank 1 for this query. Your primary task is to predict whether dense or sparse "
    "retrieval has stronger evidence for placing this gold document at rank 1.\n\n"

    "You do NOT see retrieval results. Your evidence is limited to: "
    "(1) the query, "
    "(2) the gold top-1 answer document, and "
    "(3) supplied statistics about the target collection and its vocabulary.\n\n"

    "The gold document is already known to be the correct answer. Do not judge its relevance. "
    "Use it only to reason about how the answer is expressed relative to the query and collection: "
    "for example exact lexical overlap, paraphrase, rare or common terms, identifiers, terminology "
    "mismatch, semantic similarity, or collection-relative vocabulary characteristics.\n\n"

    "Focus on evidence that discriminates dense retrieval from sparse retrieval for this specific "
    "instance. Prioritize signals affecting whether the gold document reaches rank 1, rather than "
    "signals that merely suggest it could appear somewhere in the retrieved results.\n\n"

    "Do not rely on generic retrieval heuristics alone. Statements such as 'natural language favors "
    "dense', 'exact terms favor sparse', or 'semantic queries favor embeddings' are insufficient "
    "unless the supplied query, gold document, and collection statistics provide concrete evidence "
    "that the mechanism matters for this instance.\n\n"

    "ABSTAIN when the available evidence does not give a meaningful basis for preferring dense over "
    "sparse or sparse over dense. This includes weak, balanced, complementary, or conflicting "
    "signals, and cases where a preference would mostly reflect generic retrieval priors.\n\n"

    "Reply with EXACTLY either:\n"
    "ROUTE: dense\n"
    "WHY: <one short clause naming the discriminating evidence and retrieval mechanism>\n\n"
    "or:\n"
    "ROUTE: sparse\n"
    "WHY: <one short clause naming the discriminating evidence and retrieval mechanism>\n\n"
    "or:\n"
    "ABSTAIN: <one short reason>\n"
)


def _key() -> str:
    key = os.environ.get("OPENROUTER_API_KEY") or os.environ.get("OPEN_ROUTER_API_KEY")
    if not key:
        raise RuntimeError("no OpenRouter key in env (OPENROUTER_API_KEY / OPEN_ROUTER_API_KEY)")
    return key


def complete(system: str, user: str, *, model: str = DEFAULT_MODEL,
             max_tokens: int = 120) -> tuple[str, dict]:
    """One OpenRouter chat call. `reasoning: {effort: none}` so reasoning models don't spend
    the whole budget on hidden tokens (verified: else `content=None`, still billed). Returns
    (text, usage) — cost from OpenRouter's own `usage.cost`."""
    r = requests.post(
        COMPLETIONS_URL,
        headers={"Authorization": f"Bearer {_key()}", "Content-Type": "application/json"},
        json={"model": model, "max_tokens": max_tokens, "usage": {"include": True},
              "reasoning": {"effort": "none"},
              "messages": [{"role": "system", "content": system},
                           {"role": "user", "content": user}]},
        timeout=30,
    )
    r.raise_for_status()
    body = r.json()
    return (body["choices"][0]["message"]["content"] or "").strip(), body.get("usage", {})


def build_prompt(query: str, gold_text: str, stats_line: str) -> str:
    return (f"Query: {query}\n\nKnown answer document:\n{gold_text}\n\n"
            f"Target collection statistics for this query: {stats_line}")


def parse(raw: str) -> tuple[str | None, str | None, str | None]:
    """-> (route, why, abstain_reason). route is 'dense'|'sparse'|'hybrid' or None.
    A well-formed answer is either (route, why, None) or (None, None, abstain_reason)."""
    route = why = abstain = None
    for line in raw.splitlines():
        line = line.strip()
        if m := re.match(r"ROUTE:\s*(dense|sparse|hybrid)\b", line, re.IGNORECASE):
            route = m.group(1).lower()
        elif m := re.match(r"WHY:?\s*(.*)", line, re.IGNORECASE):
            why = m.group(1).strip() or None
        elif m := re.match(r"ABSTAIN:?\s*(.*)", line, re.IGNORECASE):
            abstain = m.group(1).strip() or "(no reason given)"
    if route is None and abstain is None:
        abstain = f"UNPARSEABLE: {raw!r}"
    return route, why, abstain


def route_vector(route: str | None) -> dict[str, float]:
    """Per-route preference over {dense_only, sparse_only, pure_rrf}: 1.0 on the picked
    route, 0.0 on the others. ABSTAIN / unparseable (route is None) -> all zeros, so an
    abstaining LLM A contributes nothing to the combiner."""
    vec = {"dense_only": 0.0, "sparse_only": 0.0, "pure_rrf": 0.0}
    rid = ROUTE_MAP.get(route) if route else None
    if rid is not None:
        vec[rid] = 1.0
    return vec
