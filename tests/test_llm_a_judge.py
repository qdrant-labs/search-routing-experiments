"""LLM-A judge: parser tolerance, deterministic policy, and the acceptance
check that call A and call B differ only in dense/sparse token ordering."""
import re

from scripts.llm_a_judge import (
    JudgeResponse, decide_route, parse, system_prompt, user_prompt,
)


VALID_DENSE_WIN = """\
DENSE_MECHANISM: rare topical terms in the gold match the query semantically.
SPARSE_MECHANISM: high IDF terms would be exact-matched.
ORDER: dense > sparse
EDGE: 2
WHY: semantic match on paraphrased legal terminology drives it."""

VALID_SPARSE_WIN = """\
SPARSE_MECHANISM: exact-match on rare statutory identifier.
DENSE_MECHANISM: dense embeddings may dilute the identifier signal.
ORDER: sparse > dense
EDGE: 3
WHY: identifier match dominates."""


def test_parse_dense_first_valid():
    r = parse(VALID_DENSE_WIN)
    assert r.parse_error is None
    assert r.order_winner == "dense"
    assert r.edge == 2
    assert r.dense_mechanism.startswith("rare topical")
    assert r.sparse_mechanism.startswith("high IDF")


def test_parse_sparse_first_valid():
    r = parse(VALID_SPARSE_WIN)
    assert r.parse_error is None
    assert r.order_winner == "sparse"
    assert r.edge == 3


def test_parse_all_edge_values():
    for e in (1, 2, 3):
        raw = VALID_DENSE_WIN.replace("EDGE: 2", f"EDGE: {e}")
        assert parse(raw).edge == e


def test_parse_missing_order():
    raw = VALID_DENSE_WIN.replace("ORDER: dense > sparse\n", "")
    r = parse(raw)
    assert r.parse_error is not None and "ORDER" in r.parse_error


def test_parse_field_reordering_survives():
    # WHY moved before EDGE; parser is regex-based so field order is irrelevant
    raw = """\
DENSE_MECHANISM: a
SPARSE_MECHANISM: b
WHY: something
ORDER: sparse > dense
EDGE: 1"""
    r = parse(raw)
    assert r.parse_error is None
    assert r.order_winner == "sparse" and r.edge == 1 and r.why == "something"


def test_parse_same_side_twice_is_error():
    raw = VALID_DENSE_WIN.replace("ORDER: dense > sparse", "ORDER: dense > dense")
    r = parse(raw)
    assert r.parse_error is not None


def test_parse_totally_malformed():
    r = parse("I refuse to answer.")
    assert r.parse_error is not None
    assert r.order_winner == "" and r.edge == 0


def _rsp(winner: str, edge: int, err: str | None = None) -> JudgeResponse:
    return JudgeResponse(order_winner=winner, edge=edge, why="", dense_mechanism="x",
                         sparse_mechanism="y", raw="", parse_error=err)


def test_decide_route_agree_strong_direction():
    assert decide_route(_rsp("dense", 3), _rsp("dense", 2)) == ("dense", 2)
    assert decide_route(_rsp("sparse", 2), _rsp("sparse", 3)) == ("sparse", 2)


def test_decide_route_agree_edge_1_is_rrf():
    # min(1, 3) = 1 -> agreed but weak -> rrf
    assert decide_route(_rsp("dense", 3), _rsp("dense", 1)) == ("rrf", 1)


def test_decide_route_disagreement_is_rrf():
    assert decide_route(_rsp("dense", 3), _rsp("sparse", 3)) == ("rrf", 0)


def test_decide_route_parse_failure_propagates():
    assert decide_route(_rsp("dense", 2, err="x"), _rsp("dense", 2)) == ("parse_failure", 0)


def _swap_dense_sparse(text: str) -> str:
    """Mutual swap using placeholders so no double-substitution happens."""
    text = text.replace("dense", "\x00").replace("DENSE", "\x01")
    text = text.replace("sparse", "dense").replace("SPARSE", "DENSE")
    return text.replace("\x00", "sparse").replace("\x01", "SPARSE")


def test_call_a_and_call_b_prompts_differ_only_in_order():
    """The plan's acceptance check: rendered A and B are identical up to
    dense/sparse swap. The user prompt (query + gold + stats) is invariant."""
    sys_a = system_prompt("dense", "sparse")
    sys_b = system_prompt("sparse", "dense")
    assert _swap_dense_sparse(sys_a) == sys_b, "A and B differ in more than dense/sparse ordering"

    # user prompt has no dense/sparse tokens at all, so it MUST be identical
    u = user_prompt("q", "gold", "- stat: high")
    assert not re.search(r"\b(dense|sparse)\b", u, re.IGNORECASE), \
        "user prompt should not mention dense/sparse; the whole ordering lives in the system prompt"


def test_system_prompt_never_mentions_forbidden_concepts():
    """RRF, hybrid, fallback, thresholds, and 'confidence'/'abstain' are
    deliberately not in the judge's vocabulary — the deterministic layer above
    decides those, and the judge never learns downstream semantics."""
    for order in [("dense", "sparse"), ("sparse", "dense")]:
        sys = system_prompt(*order).lower()
        for banned in ("rrf", "hybrid", "fallback", "threshold", "confidence", "abstain"):
            assert banned not in sys, f"system prompt leaks '{banned}': the judge must not know"


def test_role_instruction_appears_verbatim():
    from scripts.llm_a_judge import ROLE_INSTRUCTION
    assert ROLE_INSTRUCTION in system_prompt("dense", "sparse")
    assert ROLE_INSTRUCTION in system_prompt("sparse", "dense")
