"""The lane rung: doc-grounded queries minted INTO a lane whose measured
class yield the residual needs — the reverse of the cell rung's query-first
direction. Absent from `OPERATOR_FAMILIES`: `AugmentationLoop.synthesize_lane`
drives it against `lane_order.parquet`, never the floor dispatcher."""

from __future__ import annotations

from typing import ClassVar

import pandas as pd

from augmentation.config import AugmentationConfig
from augmentation.core import (
    AnswerKeyPath,
    CreditGate,
    Declaration,
    Operator,
    SurfaceOrigin,
)

LANE_OPERATOR = "lane_synthesize"


def normalized(text: str) -> str:
    """The dup-guard's equality key."""
    return " ".join(str(text).lower().split())


class LaneSyntheticOperator(Operator):
    """Mint ONE query a sampled real corpus document answers, in the lane's
    own register; the class outcome is earned later by retrieval labels."""

    declaration: ClassVar[Declaration] = Declaration(
        operator=LANE_OPERATOR,
        floors="lane_order.parquet quotas (the class residual, keyed by lane)",
        surface_origin=SurfaceOrigin.SYNTHETIC,
        answer_key=AnswerKeyPath.MINTED,
        meaning_preserved=False,
        verifiable_by=(
            "the grounding document is REAL, so answerability is judged by "
            "the coherence audit and the route label is earned by retrieval "
            "against the live lane collection — never asserted"
        ),
        credit_gate=CreditGate.COHERENCE_GATE,
        tool_loop=False,
    )

    def __init__(self, config: AugmentationConfig | None = None) -> None:
        super().__init__(config or AugmentationConfig())

    def serves(self, floor: str) -> bool:
        return False

    def eligible(self, selection: pd.DataFrame, floor: str) -> pd.DataFrame:
        return selection.iloc[0:0]

    def mints(self, band) -> bool:
        return True

    def targets(self, floor: str, parent: pd.Series, requirement: tuple = ()):
        from taxonomy_generators.verify import Targets

        return Targets()

    def structural(self, parent: pd.Series, text: str, targets) -> list[str]:
        """Declared: the guards live in `rejects` because they need the
        grounding document and the dup set, which no parent carries."""
        return []

    def instruction(
        self, lane: str, doc_text: str, exemplars: tuple[str, ...] = ()
    ) -> str:
        style = (
            "\nMatch the style of real queries from this collection:\n"
            + "\n".join(f"- { ' '.join(str(e).split()) }" for e in exemplars)
            if exemplars else ""
        )
        return (
            "Below is a document from a search collection. Write ONE search "
            "query a real user of that collection might actually type, which "
            "this document answers. Do not copy a sentence from the document; "
            "a query is what someone types BEFORE reading it."
            + style
            + f"\n\nDocument:\n{doc_text}\n\nReply with the query text only."
        )

    def rejects(self, query: str, doc_text: str, taken: set[str]) -> str | None:
        """The local guards, as a reason or None: a duplicate of an existing
        query is supply we already own, and a sentence lifted from the
        document is retrieval bait, not a query."""
        key = normalized(query)
        if not key:
            return "empty"
        if key in taken:
            return "duplicate of an existing query"
        if key in normalized(doc_text):
            return "verbatim from the grounding document"
        return None
