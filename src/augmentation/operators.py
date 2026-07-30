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
"""Parents carrying embedded formal content get help-request framing in the
instruction instead of a bolted-on phrase (arch-validator 2026-07-30: the
observed absurdity was the weave, not the parent class — "could someone
help me with: <problem>" is attested register). Hard exclusion was
reverted; it returns only as a computed rule if the d34b audit measures a
high failure rate on these parents."""


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
        tool_loop=False,   # the model hits marker targets blind — d42n
    )

    def __init__(self, seed: int = 0) -> None:
        self._rng = Random(seed)

    @staticmethod
    def marker(floor: str) -> str:
        return floor.removeprefix("marker:")

    def serves(self, floor: str) -> bool:
        return super().serves(floor) and self.marker(floor) in _DECORATIONS

    def eligible(self, selection: pd.DataFrame, floor: str) -> pd.DataFrame:
        """Checkable parents not already carrying the marker — a filter on
        the selection's own `floors` column, no bank run (d42e)."""
        lacks = ~selection["floors"].map(lambda floors: floor in floors)
        return selection[lacks & selection["checkable"]]

    def instruction(self, floor: str, parent: pd.Series) -> str:
        """Per-call seeded vocabulary examples — variety is supplied by the
        bank's own phrase list, never left to the LLM's favorite opener.
        Parent-aware: formal-content parents get help-request framing."""
        marker = self.marker(floor)
        sampled = generator_for(f"sentence_markers:{marker}").sample(self._rng, 4)
        examples = ", ".join(repr(s) for s in dict.fromkeys(sampled))
        framing = (
            (
                "This query contains formal content (math or code): frame the "
                "decoration as a real person bringing the problem somewhere "
                "for help — like a forum post — never a phrase bolted onto a "
                "bare statement. "
            )
            if any(f in parent["floors"] for f in _FORMAL_FLOORS)
            else ""
        )
        return (
            "Rewrite the user's search query by weaving in "
            f"{_DECORATIONS[marker]}. Vocabulary inspirations (adapt freely): "
            f"{examples}. Pick a phrasing that fits the query's tone and "
            "world, and vary it — never default to one stock opener; it may "
            f"sit at the start, middle, or end. {framing}Keep every content "
            "word and the meaning unchanged. Add no other information: no "
            "names, numbers, dates, or identifiers."
        )

    def targets(self, floor: str) -> Targets:
        return Targets(spans=(SpanTarget(feature=self.marker(floor), min_count=1),))


OPERATORS: tuple[Operator, ...] = (DecorateOperator(),)


def operator_for(floor: str) -> Operator | None:
    """Dispatch by floor key (d42d) — `id:` / `marker:` / `logical:` /
    stat-axis prefixes, not the loop's grounding branches."""
    return next((op for op in OPERATORS if op.serves(floor)), None)
