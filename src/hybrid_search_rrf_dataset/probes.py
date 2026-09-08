"""The seven archetype probes frozen from the auto-fusion disagreement run of
2026-08-10 — queries whose route follows from retrieval mechanics alone. A
coverage, predicate or extractor pass is checked against this fixed instrument
rather than against an aggregate that hides archetype-level failure."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import pandas as pd

from hybrid_search_rrf_dataset.fusion import StrategyName

if TYPE_CHECKING:
    from hybrid_search_rrf_dataset.router import AcceptabilityRouter, StrategyRouter


@dataclass(frozen=True)
class ArchetypeProbe:
    """One hand-ruled query and the route retrieval mechanics demand for it.
    `expected` is None where no rule decides and only the corpus can."""

    query: str
    expected: StrategyName | None
    why: str


ARCHETYPE_PROBES: tuple[ArchetypeProbe, ...] = (
    ArchetypeProbe(
        query="a3f5d8b9e12c4d56789abcdef0123456",
        expected=StrategyName.SPARSE_ONLY,
        why="hex digest with no semantic surface. Measured natural_language_"
        "share=1.0 — spaCy tags the unknown token AUX — and it is claimed by "
        "bare_concept_token, whose prior is dense_only.",
    ),
    ArchetypeProbe(
        query="/etc/nginx/nginx.conf",
        expected=StrategyName.SPARSE_ONLY,
        why="file path, reproduced verbatim wherever it is discussed. The one "
        "probe whose cells (web_locator_token, keyword_telegram_short) are "
        "both right.",
    ),
    ArchetypeProbe(
        query="ERR_CONNECTION_RESET",
        expected=StrategyName.SPARSE_ONLY,
        why="exact error string, claimed as error_code_like. No cell owns a "
        "bare code: status_code_idf_split needs 7-19 words, so this lands in "
        "bare_concept_token.",
    ),
    ArchetypeProbe(
        query="explain quicksort",
        expected=StrategyName.DENSE_ONLY,
        why="two common words naming one concept — the cleanest dense case, "
        "and correctly claimed by bare_concept_token. Failure here is the "
        "model alone, not the composition.",
    ),
    ArchetypeProbe(
        query="HTTP 502",
        expected=None,
        why="ubiquitous status code anchors nothing; only the corpus decides. "
        "Same missing-cell gap as ERR_CONNECTION_RESET.",
    ),
    ArchetypeProbe(
        query="comment volent les oiseaux",
        expected=StrategyName.DENSE_ONLY,
        why="French. Extractors are English-only, so the features are "
        "degenerate and the router degrades silently rather than erroring.",
    ),
    ArchetypeProbe(
        query="como aprender a programar en rust",
        expected=StrategyName.DENSE_ONLY,
        why="Spanish; same English-only boundary, and it reaches no cell at "
        "all.",
    ),
)


def probe(router: StrategyRouter | AcceptabilityRouter) -> pd.DataFrame:
    """Each probe's expected route beside what `router` actually serves."""
    rows = []
    for case in ARCHETYPE_PROBES:
        served = router.predict(case.query)
        rows.append(
            {
                "query": case.query,
                "expected": case.expected,
                "served": served,
                "agrees": pd.NA
                if case.expected is None
                else served == case.expected,
                "why": case.why,
            }
        )
    return pd.DataFrame(rows)


@dataclass(frozen=True)
class GoldenCase:
    """One auto-fusion golden query with its banded 0-9 identifier expectation."""

    query: str
    expected_hi: int
    expected_lo: int


# Frozen 2026-08-13 out of route_experiments.ipynb cell 1d9fc66a — the probe
# gate's second instrument (VERDICT.md thresholds); edit only with a dated reason.
GOLDEN_SET_AUTO_FUSION: tuple[GoldenCase, ...] = (
    GoldenCase("who founded apple?", 2, 0),
    GoldenCase("how does photosynthesis work in plants", 2, 0),
    GoldenCase("explain quicksort", 2, 0),
    GoldenCase("best laptop for college students 2024", 3, 0),
    GoldenCase("comment volent les oiseaux", 2, 0),
    GoldenCase("como aprender a programar en rust", 2, 0),
    GoldenCase("tell me about dogs", 2, 0),
    GoldenCase("hey can you help me out", 2, 0),
    GoldenCase("550e8400-e29b-41d4-a716-446655440000", 9, 8),
    GoldenCase("00000000-0000-0000-0000-000000000000", 9, 8),
    GoldenCase("ERR_CONNECTION_RESET", 9, 8),
    GoldenCase("ENOENT", 9, 7),
    GoldenCase("HTTP 502", 9, 6),
    GoldenCase("v1.2.3 changelog", 9, 6),
    GoldenCase("Python 3.11.4 release notes", 8, 5),
    GoldenCase("a3f5d8b9e12c4d56789abcdef0123456", 9, 8),
    GoldenCase("/etc/nginx/nginx.conf", 9, 7),
    GoldenCase("B07XJ8C8F5", 9, 7),
    GoldenCase("GPT-3", 8, 5),
    GoldenCase("BERT model paper", 7, 4),
    GoldenCase("ThinkPad X1 broken screen replacement", 6, 3),
    GoldenCase("iPhone 15 Pro Max battery life", 6, 3),
    GoldenCase("kubernetes pod CrashLoopBackOff", 8, 5),
    GoldenCase("why does my Java program throw NullPointerException at line 42", 6, 3),
    GoldenCase("C++ undefined reference to vtable", 8, 5),
    GoldenCase("how to fix broken screen on my Lenovo ThinkPad X1", 5, 2),
    GoldenCase("linux", 3, 1),
    GoldenCase("covid", 3, 1),
    GoldenCase("the", 2, 0),
    GoldenCase("hello", 2, 0),
)


def probe_ordering(router: StrategyRouter) -> pd.DataFrame:
    """Both probe sets against `router.explain()`: single-route expectations score
    by probability ordering (the §1 gate criterion); mixed bands, where no dense/sparse
    ordering is implied, fall back to served-route band membership."""
    from hybrid_search_rrf_dataset.router import _production_route

    def _ordering(e: dict, band: set[StrategyName]) -> object:
        if band == {StrategyName.SPARSE_ONLY}:
            return e["p_sparse"] > e["p_dense"]
        if band == {StrategyName.DENSE_ONLY}:
            return e["p_dense"] > e["p_sparse"]
        return e["route"] in band if band else pd.NA

    rows = []
    for case in ARCHETYPE_PROBES:
        e = router.explain(case.query)
        band = set() if case.expected is None else {case.expected}
        rows.append({
            "set": "archetype", "query": case.query,
            "expected": "corpus-decides" if case.expected is None else case.expected.value,
            "served": e["route"].value, "p_dense": e["p_dense"],
            "p_sparse": e["p_sparse"], "ordering_ok": _ordering(e, band),
        })
    for g in GOLDEN_SET_AUTO_FUSION:
        band = {_production_route(s) for s in range(g.expected_lo, g.expected_hi + 1)}
        e = router.explain(g.query)
        rows.append({
            "set": "golden", "query": g.query,
            "expected": "..".join(sorted(r.value for r in band)),
            "served": e["route"].value, "p_dense": e["p_dense"],
            "p_sparse": e["p_sparse"], "ordering_ok": _ordering(e, band),
        })
    return pd.DataFrame(rows)
