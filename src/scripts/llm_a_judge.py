"""LLM-A escalation judge: two calls per row with dense/sparse order swap,
deterministic route policy (agree + min(edge) >= 2 -> direction, else rrf).
The judge predicts direction only; magnitude gating is decided here.
The judge is never told that edge triggers routing, that it runs twice, or
what RRF/hybrid means."""
from __future__ import annotations

import hashlib
import os
import re
from dataclasses import dataclass
from typing import Final, Literal

import requests

COMPLETIONS_URL: Final[str] = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_MODEL: Final[str] = "openai/gpt-5.6-luna"


ROLE_INSTRUCTION: Final[str] = (
    "A choice is required even when the evidence is weak. Your job is to identify the "
    "direction of the stronger evidence, not to decide whether the evidence is sufficient "
    "for deployment."
)

CORPUS_PROXY: Final[str] = (
    "You cannot see the competing documents in the collection. The collection statistics "
    "above stand in for them. Reason about whether the gold document's discriminating "
    "terms are rare in this specific collection, not merely about lexical overlap between "
    "the query and the gold document."
)


def system_prompt(first: str, second: str) -> str:
    """`first`/`second` control the presentation order of dense/sparse EVERYWHERE
    they appear in ordered mentions (task description, output options, mechanism
    field names). Two calls per row, with the roles swapped, is the sampling
    variation."""
    return (
        f"You compare {first} retrieval and {second} retrieval as candidates for "
        f"ranking a known-relevant gold document above competitors in a specific "
        f"document collection. "
        f"{ROLE_INSTRUCTION} "
        f"{CORPUS_PROXY}\n\n"
        f"Reply with EXACTLY these five lines, in this order:\n"
        f"{first.upper()}_MECHANISM: <one sentence: the specific mechanism by which "
        f"{first} retrieval would rank the gold document above competitors in this collection>\n"
        f"{second.upper()}_MECHANISM: <one sentence: same for {second} retrieval>\n"
        f"ORDER: {first} > {second} | {second} > {first}\n"
        f"EDGE: 1 | 2 | 3    (how clearly the evidence separates the two, "
        f"where 1 is barely and 3 is decisively)\n"
        f"WHY: <one sentence naming the discriminating mechanism that decided ORDER>"
    )


def user_prompt(query: str, gold_text: str, stats_block: str) -> str:
    return (
        f"Query: {query}\n\n"
        f"Gold document (known to answer the query):\n{gold_text}\n\n"
        f"Collection statistics (these describe the corpus the gold document must be "
        f"retrieved from):\n{stats_block}"
    )


@dataclass(frozen=True)
class JudgeResponse:
    order_winner: str        # "dense" | "sparse" | ""
    edge: int                # 1 | 2 | 3, or 0 if parse failed
    why: str
    dense_mechanism: str
    sparse_mechanism: str
    raw: str
    parse_error: str | None = None


_ORDER_RE = re.compile(r"ORDER:\s*(dense|sparse)\s*>\s*(dense|sparse)\b", re.IGNORECASE)
_EDGE_RE = re.compile(r"EDGE:\s*([123])\b")
_WHY_RE = re.compile(r"WHY:\s*(.+)", re.IGNORECASE)
_MECH_RE = re.compile(r"(DENSE|SPARSE)_MECHANISM:\s*(.+)", re.IGNORECASE)


def parse(raw: str) -> JudgeResponse:
    """Strict parse, tolerant of field reordering. `parse_error` is set on any
    missing/invalid field; caller decides retry vs fallback."""
    order_m = _ORDER_RE.search(raw)
    edge_m = _EDGE_RE.search(raw)
    why_m = _WHY_RE.search(raw)
    mechs: dict[str, str] = {}
    for m in _MECH_RE.finditer(raw):
        mechs[m.group(1).lower()] = m.group(2).strip()

    missing: list[str] = []
    if not order_m:
        missing.append("ORDER")
    if not edge_m:
        missing.append("EDGE")
    if not why_m:
        missing.append("WHY")
    if "dense" not in mechs:
        missing.append("DENSE_MECHANISM")
    if "sparse" not in mechs:
        missing.append("SPARSE_MECHANISM")

    order_winner = ""
    if order_m:
        first, second = order_m.group(1).lower(), order_m.group(2).lower()
        if first == second:
            missing.append("ORDER (same side twice)")
        else:
            order_winner = first

    return JudgeResponse(
        order_winner=order_winner,
        edge=int(edge_m.group(1)) if edge_m else 0,
        why=why_m.group(1).strip() if why_m else "",
        dense_mechanism=mechs.get("dense", ""),
        sparse_mechanism=mechs.get("sparse", ""),
        raw=raw,
        parse_error=("missing/invalid: " + ", ".join(missing)) if missing else None,
    )


def _openrouter_key() -> str:
    key = os.environ.get("OPENROUTER_API_KEY") or os.environ.get("OPEN_ROUTER_API_KEY")
    if not key:
        raise RuntimeError("no OpenRouter key in env (OPENROUTER_API_KEY / OPEN_ROUTER_API_KEY)")
    return key


def _call(system: str, user: str, model: str, temperature: float,
          max_tokens: int) -> tuple[str, dict]:
    r = requests.post(
        COMPLETIONS_URL,
        headers={"Authorization": f"Bearer {_openrouter_key()}", "Content-Type": "application/json"},
        json={
            "model": model,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "usage": {"include": True},
            "reasoning": {"effort": "none"},
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        },
        timeout=60,
    )
    r.raise_for_status()
    body = r.json()
    return (body["choices"][0]["message"]["content"] or "").strip(), body.get("usage", {})


FORMAT_REMINDER: Final[str] = (
    "Your previous response did not match the required format. Reply again with EXACTLY the "
    "five lines specified: DENSE_MECHANISM, SPARSE_MECHANISM, ORDER, EDGE, WHY."
)


def one_call(system: str, user: str, *, model: str = DEFAULT_MODEL,
             temperature: float = 0.0,
             max_tokens: int = 300) -> tuple[JudgeResponse, dict, int]:
    """Single call + one retry with format reminder on malformed.
    -> (parsed, usage_summed_across_attempts, retries_used)."""
    raw, usage = _call(system, user, model, temperature, max_tokens)
    parsed = parse(raw)
    if parsed.parse_error is None:
        return parsed, usage, 0
    raw2, usage2 = _call(system, f"{user}\n\n{FORMAT_REMINDER}", model, temperature, max_tokens)
    combined = _sum_usage(usage, usage2)
    return parse(raw2), combined, 1


def _sum_usage(a: dict, b: dict) -> dict:
    keys = set(a) | set(b)
    return {k: (a.get(k, 0) or 0) + (b.get(k, 0) or 0) for k in keys
            if isinstance(a.get(k, 0), (int, float)) and isinstance(b.get(k, 0), (int, float))}


Route = Literal["dense", "sparse", "rrf", "parse_failure"]


def decide_route(a: JudgeResponse, b: JudgeResponse) -> tuple[Route, int]:
    """agree(order) + min(edge) >= 2 -> direction. Else rrf.
    parse_failure on either call -> parse_failure (caller routes to rrf, tracked
    separately for the audit).
    Disagreement under order swap is a signal, not an error — don't rerun to converge."""
    if a.parse_error or b.parse_error:
        return "parse_failure", 0
    if a.order_winner != b.order_winner:
        return "rrf", 0
    edge = min(a.edge, b.edge)
    if edge >= 2:
        return a.order_winner, edge  # type: ignore[return-value]
    return "rrf", edge


def prompt_hash(system_a: str, user: str) -> str:
    """Stable id for the call-A prompt; call-B is derivable (only order differs)."""
    h = hashlib.sha256()
    h.update(system_a.encode())
    h.update(b"\n---\n")
    h.update(user.encode())
    return h.hexdigest()[:16]


@dataclass
class JudgeLog:
    call_a_raw: str
    call_a_order: tuple[str, str]        # (first, second) as rendered
    call_a_order_winner: str
    call_a_edge: int
    call_a_retries: int
    call_a_usage: dict
    call_b_raw: str
    call_b_order: tuple[str, str]
    call_b_order_winner: str
    call_b_edge: int
    call_b_retries: int
    call_b_usage: dict
    agreement: bool
    route: Route
    final_edge: int
    prompt_hash: str
    model: str


def judge_row(query: str, gold_text: str, stats_block: str, *,
              model: str = DEFAULT_MODEL,
              temperature: float = 0.0,
              max_tokens: int = 300) -> JudgeLog:
    """Full row-level judge: two calls with the dense/sparse order swapped,
    strict parse (with one retry each on malformed), deterministic policy.
    Neither call knows the other exists; the caller sees the final route."""
    user_msg = user_prompt(query, gold_text, stats_block)

    sys_a = system_prompt("dense", "sparse")
    a_parsed, a_usage, a_retries = one_call(sys_a, user_msg, model=model,
                                            temperature=temperature, max_tokens=max_tokens)

    sys_b = system_prompt("sparse", "dense")
    b_parsed, b_usage, b_retries = one_call(sys_b, user_msg, model=model,
                                            temperature=temperature, max_tokens=max_tokens)

    route, edge = decide_route(a_parsed, b_parsed)

    return JudgeLog(
        call_a_raw=a_parsed.raw,
        call_a_order=("dense", "sparse"),
        call_a_order_winner=a_parsed.order_winner,
        call_a_edge=a_parsed.edge,
        call_a_retries=a_retries,
        call_a_usage=a_usage,
        call_b_raw=b_parsed.raw,
        call_b_order=("sparse", "dense"),
        call_b_order_winner=b_parsed.order_winner,
        call_b_edge=b_parsed.edge,
        call_b_retries=b_retries,
        call_b_usage=b_usage,
        agreement=(not a_parsed.parse_error
                   and not b_parsed.parse_error
                   and a_parsed.order_winner == b_parsed.order_winner),
        route=route,
        final_edge=edge,
        prompt_hash=prompt_hash(sys_a, user_msg),
        model=model,
    )


def log_row(log: JudgeLog) -> dict:
    """Flatten a JudgeLog into a parquet-friendly row. Both raw responses + both
    parsed tuples are preserved so a disagreement bucket can be re-queried by a
    later adjudicator pass without re-labelling."""
    return {
        "route": log.route, "final_edge": log.final_edge, "agreement": log.agreement,
        "call_a_raw": log.call_a_raw, "call_a_first": log.call_a_order[0],
        "call_a_second": log.call_a_order[1],
        "call_a_order_winner": log.call_a_order_winner, "call_a_edge": log.call_a_edge,
        "call_a_retries": log.call_a_retries,
        "call_a_cost": float(log.call_a_usage.get("cost", 0) or 0),
        "call_a_prompt_tokens": log.call_a_usage.get("prompt_tokens"),
        "call_a_completion_tokens": log.call_a_usage.get("completion_tokens"),
        "call_b_raw": log.call_b_raw, "call_b_first": log.call_b_order[0],
        "call_b_second": log.call_b_order[1],
        "call_b_order_winner": log.call_b_order_winner, "call_b_edge": log.call_b_edge,
        "call_b_retries": log.call_b_retries,
        "call_b_cost": float(log.call_b_usage.get("cost", 0) or 0),
        "call_b_prompt_tokens": log.call_b_usage.get("prompt_tokens"),
        "call_b_completion_tokens": log.call_b_usage.get("completion_tokens"),
        "prompt_hash": log.prompt_hash, "model": log.model,
    }
