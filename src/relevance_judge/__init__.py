"""Relevance-atom judge (doc: relevance-judge-recovery). Per-(query,doc) binary
judgments that extend qrels; route scores recompute arithmetically at scoring
time. The judge supplies truth atoms, never route opinions."""

from relevance_judge.config import RelevanceJudgeConfig
from relevance_judge.judge import JudgeRunLog, RelevanceJudge
from relevance_judge.residual import (
    Arch5kDraw,
    JudgeQueue,
    V2Labels,
    above_gold,
    regime,
    tail_docs,
)
from relevance_judge.scoring import PilotScorer
from relevance_judge.sources import Sources
from relevance_judge.validation import Confusion, Gate, ValidationHarness

__all__ = [
    "RelevanceJudgeConfig",
    "RelevanceJudge",
    "JudgeRunLog",
    "JudgeQueue",
    "Arch5kDraw",
    "V2Labels",
    "above_gold",
    "tail_docs",
    "regime",
    "PilotScorer",
    "Sources",
    "ValidationHarness",
    "Confusion",
    "Gate",
]
