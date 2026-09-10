"""Qrels expansion: judge (query, document) pairs the answer key never covered.

Wraps `relevance_judge.RelevanceJudge`. Batches are bounded and synchronous —
the judge caches on `(dataset, query_id, doc_id)`, so a caller looping over a
large work list re-pays nothing for pairs an earlier batch already banked.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from augmentation.engine import Budget, BudgetExceeded

if TYPE_CHECKING:
    from relevance_judge.judge import RelevanceJudge

MAX_PAIRS = 500
"""Per-request ceiling. Larger work lists loop; the cache makes that cheap."""

CHARS_PER_TOKEN = 3
"""Deliberately generous, matching `Budget.reserve`'s own estimate so a
dry-run figure is comparable to the ceiling that will enforce it."""

OPERATING_POINT = {
    "precision_relevant": 0.987,
    "recall_relevant": 0.308,
    "agreement_overall": 0.523,
    "measured_on_pairs": 3600,
    "measured": "2026-09-07",
}
"""The judge's MEASURED behaviour, travelling with every response. High
precision, low recall: `relevant=false` mostly means *not found*, not
*checked and rejected*. Never read a negative as evidence of irrelevance."""

PAIR_COLUMNS = ("dataset", "query_id", "doc_id", "query", "doc_text")


class JudgePair(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    dataset: str
    query_id: str
    doc_id: str
    query: str
    doc_text: str


class JudgePlan(BaseModel):
    """What a run would do and cost, without calling a model."""

    model_config = ConfigDict(frozen=True)

    submitted: int
    already_judged: int
    to_judge: int
    estimated_usd: float
    estimated_prompt_tokens: int
    within_ceiling: bool | None = Field(
        default=None,
        description="Whether the estimate fits max_spend_usd; null when the "
        "caller supplied no ceiling with the dry run.",
    )
    operating_point: dict = Field(default_factory=lambda: dict(OPERATING_POINT))


class JudgeResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    run_id: str
    submitted: int
    skipped_cached: int
    judged: int
    relevant: int
    unreadable: int
    spent_usd: float
    calls: int
    stopped_on_budget: bool = False
    operating_point: dict = Field(default_factory=lambda: dict(OPERATING_POINT))


def _frame(pairs: list[JudgePair]) -> pd.DataFrame:
    return pd.DataFrame(
        [p.model_dump() for p in pairs], columns=list(PAIR_COLUMNS)
    )


def _unjudged(frame: pd.DataFrame, judge: RelevanceJudge) -> pd.DataFrame:
    """Rows the judge has not already banked — the cache key is
    `(dataset, query_id, doc_id)`."""
    if frame.empty:
        return frame
    keys = pd.MultiIndex.from_frame(
        frame[["dataset", "query_id", "doc_id"]].astype(str)
    )
    return frame[~keys.isin(judge.judged_keys())]


def plan(
    pairs: list[JudgePair],
    judge: RelevanceJudge,
    *,
    max_spend_usd: float | None = None,
) -> JudgePlan:
    """Cost the work without spending. Prices unjudged pairs only."""
    _guard(pairs)
    frame = _frame(pairs)
    todo = _unjudged(frame, judge)
    engine = judge.config.engine

    prompt_tokens = int(
        sum(
            (len(row.query) + len(row.doc_text)) // CHARS_PER_TOKEN
            for row in todo.itertuples(index=False)
        )
    )
    answer_tokens = len(todo) * judge.config.max_answer_tokens
    usd = (
        prompt_tokens * engine.usd_per_mtok_in
        + answer_tokens * engine.usd_per_mtok_out
    ) / 1e6
    return JudgePlan(
        submitted=len(frame),
        already_judged=len(frame) - len(todo),
        to_judge=len(todo),
        estimated_usd=round(usd, 4),
        estimated_prompt_tokens=prompt_tokens,
        within_ceiling=None if max_spend_usd is None else usd <= max_spend_usd,
    )


def expand(
    pairs: list[JudgePair],
    judge: RelevanceJudge,
    *,
    run_id: str,
    max_spend_usd: float,
) -> JudgeResult:
    """Judge the unjudged pairs and bank the verdicts.

    A budget stop is a partial success, not a failure: `judge_pairs` banks the
    verdicts already paid for before it unwinds, so the result reports what
    landed rather than discarding it.
    """
    _guard(pairs)
    engine = judge.config.engine
    budget = Budget(
        max_spend_usd,
        usd_per_mtok_in=engine.usd_per_mtok_in,
        usd_per_mtok_out=engine.usd_per_mtok_out,
    )
    frame = _frame(pairs)
    stopped = False
    try:
        counts = judge.judge_pairs(frame, run_id=run_id, budget=budget)
    except BudgetExceeded:
        stopped = True
        counts = {"candidates": len(frame), "skipped": 0, "judged": 0,
                  "relevant": 0, "unreadable": 0}
    return JudgeResult(
        run_id=run_id,
        submitted=counts["candidates"],
        skipped_cached=counts["skipped"],
        judged=counts["judged"],
        relevant=counts["relevant"],
        unreadable=counts["unreadable"],
        spent_usd=round(budget.spent_usd, 4),
        calls=budget.calls,
        stopped_on_budget=stopped,
    )


def _guard(pairs: list[JudgePair]) -> None:
    if not pairs:
        raise ValueError("no pairs submitted")
    if len(pairs) > MAX_PAIRS:
        raise ValueError(
            f"{len(pairs)} pairs exceeds the {MAX_PAIRS} per-request cap; "
            "loop — cached pairs are not re-paid"
        )
