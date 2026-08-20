"""The synthetic rung: a query written from a cell's predicate alone, plus the
document that answers it.

Reached when no corpus supplies the surface, so there is no parent to rewrite
and nothing to copy — the one rung where a model is doing generation rather
than selection or placement. Deliberately absent from `OPERATOR_FAMILIES`: the
dispatcher would hand it real parents at QUERY_ONLY and every cell would route
here, which is the opposite of what the gate decides.
"""

from __future__ import annotations

from random import Random
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
from composition.cell_targets import generation_branches
from composition.cells import CELLS_BY_NAME, AxisBand
from taxonomy_generators.registry import generator_for
from taxonomy_generators.verify import Targets

_EXAMPLES = 3
"""Surfaces sampled per span band — the bank's own vocabulary, so the model
is shown what the detector accepts rather than guessing at it."""


def _phrase(band: AxisBand) -> str:
    """One band as an instruction line."""
    if band.is_span:
        if band.demands_presence:
            return f"contains at least {int(band.at_least)} {band.member}"
        return f"contains no {band.member}"
    low = "" if band.at_least is None else f"at least {band.at_least:g}"
    high = "" if band.below is None else f"under {band.below:g}"
    return f"{band.member} {' and '.join(p for p in (low, high) if p)}"


class SyntheticOperator(Operator):
    """Generate the query a cell describes, with no parent and no corpus."""

    declaration: ClassVar[Declaration] = Declaration(
        operator="synthesize",
        floors="any cell whose bands rung 1 cannot reach",
        surface_origin=SurfaceOrigin.SYNTHETIC,
        answer_key=AnswerKeyPath.MINTED,
        # nothing to preserve: there is no parent whose meaning could survive
        meaning_preserved=False,
        verifiable_by=(
            "every band of the chosen branch re-measured on the generated "
            "query. The document is written to answer that query and is "
            "judged by the coherence audit, never asserted"
        ),
        credit_gate=CreditGate.COHERENCE_GATE,
        tool_loop=False,
    )

    def __init__(self, config: AugmentationConfig | None = None) -> None:
        config = config or AugmentationConfig()
        super().__init__(config)
        self._seed = config.seed

    def branch(self, floor: str, index: int = 0):
        """The reading this row aims at — a cell's `any_of` cannot be
        generated as an OR, so each row commits to one alternative and the
        index spreads rows across them."""
        branches = generation_branches(CELLS_BY_NAME[floor])
        return branches[index % len(branches)]

    def mints(self, band: AxisBand) -> bool:
        """Everything: the query is written from scratch, so no band is out
        of reach on the mint side — whether the model HITS it is the local
        re-measure's verdict, not a declaration."""
        return True

    def serves(self, floor: str) -> bool:
        return floor in CELLS_BY_NAME

    def eligible(self, selection: pd.DataFrame, floor: str) -> pd.DataFrame:
        """None, by construction — this rung exists because no parent
        qualifies. `AugmentationLoop.synthesize` drives it directly rather
        than through the parent queue."""
        return selection.iloc[0:0]

    def instruction(
        self, floor: str, parent: pd.Series, requirement: tuple = ()
    ) -> str:
        cell = CELLS_BY_NAME[floor]
        index = int(parent.get("branch_index", 0))
        branch = self.branch(floor, index)
        bands = cell.predicate + (
            (branch.alternative,) if branch.alternative is not None else ()
        )
        rng = Random(f"{self._seed}:{floor}:{index}")
        wanted = [
            f"- {_phrase(band)}"
            + self._examples(band, rng)
            for band in bands
        ]
        shape = (
            f"\nA query of this kind looks like: {cell.looks_like}"
            if cell.looks_like
            else ""
        )
        return (
            "Write ONE realistic search query somebody would actually type. "
            "It must satisfy every one of these measured properties:\n"
            + "\n".join(wanted)
            + shape
            + "\nReply with the query text only."
        )

    @staticmethod
    def _examples(band: AxisBand, rng: Random) -> str:
        """The bank's own sampler, so an identifier band is shown real
        accepted shapes instead of the model's idea of one."""
        if not band.is_span or not band.demands_presence:
            return ""
        # a band names its column `group.member`; a generator names itself
        # `group:member` — the same identity, one character apart
        try:
            surfaces = generator_for(band.column.replace(".", ":", 1)).sample(
                rng, _EXAMPLES
            )
        except (KeyError, ValueError):
            return ""
        # reverse-regex sampling can emit newlines; they would break a
        # "reply with the query only" instruction into two apparent answers
        cleaned = dict.fromkeys(" ".join(str(s).split()) for s in surfaces)
        return f" (shapes that count: {', '.join(cleaned)})"

    def document(self, floor: str, query: str, ordinal: int = 1) -> str:
        """A passage that ANSWERS the generated query; docs past the first
        must answer in DIFFERENT wording, because a verbatim twin is a
        dup-cluster artifact, not qrels depth."""
        base = (
            "Write a short factual passage (3-6 sentences) that fully answers "
            "the user's search query. Write it as a document that would "
            "legitimately be retrieved for it — no preamble, no restating the "
            "query, no addressing the reader. Reply with the passage only."
        )
        if ordinal <= 1:
            return base
        return base + (
            " This is an INDEPENDENT second source: cover the same answer "
            "with different wording, sentence structure and vocabulary than "
            "an encyclopedia would — as if written by a different author."
        )

    def targets(
        self, floor: str, parent: pd.Series, requirement: tuple = ()
    ) -> Targets:
        return self.branch(floor, int(parent.get("branch_index", 0))).targets

    def structural(
        self, parent: pd.Series, text: str, targets: Targets
    ) -> list[str]:
        """Declared: no parent-relative check exists, because there is no
        parent. Everything checkable about this row is in `targets`, and its
        answerability is what the coherence gate audits."""
        return []

    def candidate(self, parent, floor, outcome):
        base = super().candidate(parent, floor, outcome)
        return base.model_copy(update={
            # the placeholder id IS the row's id — there is no parent to
            # derive one from, and `generated_from` stays empty rather than
            # naming a lineage that does not exist
            "query_id": str(parent["query_id"]),
            "generated_from": "",
        })
