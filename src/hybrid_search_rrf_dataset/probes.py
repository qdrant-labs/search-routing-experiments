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
