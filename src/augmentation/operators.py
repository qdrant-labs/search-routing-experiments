"""The operator families (d42d). Pass 1 ships Decorate — the gate-free
family; OperatorSyntaxRewrite, StatRewrite, Inject and Corrupt follow their
gates (d42h)."""

from __future__ import annotations

from random import Random
from typing import ClassVar

import pandas as pd

from augmentation.core import (
    AnswerKeyPath,
    CreditGate,
    Declaration,
    Grounding,
    Operator,
)
from taxonomy_generators.registry import generator_for
from taxonomy_generators.verify import SpanTarget, Targets

_DECORATIONS: dict[str, str] = {
    "greeting": "a natural greeting (e.g. 'hi there,', 'good morning,')",
    "interjection": "a natural interjection (e.g. 'hmm,', 'oh,', 'ugh,')",
    "politeness": "a politeness phrase (e.g. 'please', 'could you kindly')",
}
"""Register decorations only. Other marker banks (acronym, ...) are
detections, not weavable filler — Decorate's meaning-preserved claim holds
for exactly these."""


_FORMAL_FLOORS = ("logical:math_expression", "logical:code_fragment")
"""Parents carrying embedded formal content are not decorated — a greeting
on a geometry problem statement is register-incompatible (user-ruled
2026-07-29); the floors column already knows who they are."""


class DecorateOperator(Operator):
    """Weave a register marker into the query (d40d: politeness-class,
    meaning-preserving — parent qrels inherit, no grounding, no gate)."""

    declaration: ClassVar[Declaration] = Declaration(
        operator="decorate",
        floor_prefix="marker:",
        grounding=Grounding.NONE,
        answer_key=AnswerKeyPath.INHERIT,
        meaning_preserved=True,
        verifiable_by="target marker span present on local re-measure",
        credit_gate=CreditGate.NONE,
    )

    def __init__(self, seed: int = 0) -> None:
        self._rng = Random(seed)

    @staticmethod
    def marker(floor: str) -> str:
        return floor.removeprefix("marker:")

    def serves(self, floor: str) -> bool:
        return super().serves(floor) and self.marker(floor) in _DECORATIONS

    def eligible(self, selection: pd.DataFrame, floor: str) -> pd.DataFrame:
        """Checkable parents not already carrying the marker and free of
        embedded formal content — filters on the selection's own `floors`
        column, no bank run (d42e)."""
        lacks = ~selection["floors"].map(lambda floors: floor in floors)
        conversational = selection["floors"].map(
            lambda floors: not any(f in floors for f in _FORMAL_FLOORS)
        )
        return selection[lacks & conversational & selection["checkable"]]

    def instruction(self, floor: str) -> str:
        """Per-call seeded vocabulary examples — variety is supplied by the
        bank's own phrase list, never left to the LLM's favorite opener."""
        marker = self.marker(floor)
        sampled = generator_for(f"sentence_markers:{marker}").sample(self._rng, 4)
        examples = ", ".join(repr(s) for s in dict.fromkeys(sampled))
        return (
            "Rewrite the user's search query by weaving in "
            f"{_DECORATIONS[marker]}. Vocabulary inspirations (adapt freely): "
            f"{examples}. Pick a phrasing that fits the query's tone and "
            "world, and vary it — never default to one stock opener; it may "
            "sit at the start, middle, or end. Keep every content word and "
            "the meaning unchanged. Add no other information: no names, "
            "numbers, dates, or identifiers. Check your text with the "
            f"verify tool (span feature {marker!r}), then return the final "
            "query text."
        )

    def targets(self, floor: str) -> Targets:
        return Targets(spans=(SpanTarget(feature=self.marker(floor), min_count=1),))


OPERATORS: tuple[Operator, ...] = (DecorateOperator(),)


def operator_for(floor: str) -> Operator | None:
    """Dispatch by floor key (d42d) — `id:` / `marker:` / `logical:` /
    stat-axis prefixes, not the loop's grounding branches."""
    return next((op for op in OPERATORS if op.serves(floor)), None)
