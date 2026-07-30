"""Operator declarations and the pool-row contract (SPEC d42c/d).

Operators are grounding-aware — they read the order sheet, pick parents,
and decide the answer-key path — which is why they live here and not in
grounding-blind taxonomy_generators (d34a/d42b). Every operator carries a
frozen `Declaration`; anything undeclared is feature-stock by default.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from enum import StrEnum
from typing import ClassVar

import pandas as pd
from pydantic import BaseModel, ConfigDict

from taxonomy_generators.verify import Targets


class Grounding(StrEnum):
    NONE = "none"
    DOC_COPIED = "doc_copied"
    SYNTHETIC = "synthetic"


class AnswerKeyPath(StrEnum):
    INHERIT = "inherit"   # meaning preserved -> parent qrels stay valid
    MINTED = "minted"     # answer minted from the grounding doc


class CreditGate(StrEnum):
    """admission sequencing: what must pass before this operator's
    rows earn floor credit. Rows produced before the gate bank as
    feature-stock."""

    NONE = "none"
    DECLARATION_AUDIT = "declaration_audit"
    COHERENCE_GATE = "coherence_gate"


class Declaration(BaseModel):
    """The default-deny contract (d42c), auditable in code review."""

    model_config = ConfigDict(frozen=True)

    operator: str
    floor_prefix: str
    grounding: Grounding
    answer_key: AnswerKeyPath
    meaning_preserved: bool
    verifiable_by: str
    credit_gate: CreditGate


class AugmentedCandidate(BaseModel):
    """One accepted row awaiting the mini-fill — born with its d40(c)
    certificate. `generated_from` is the lineage edge (d42j); the
    parent-side `superseded` view is derived, never stored."""

    model_config = ConfigDict(frozen=True)

    query_id: str
    query: str
    floor: str
    operator: str
    provenance: str = "augmented"
    generated_from: str
    parent_dataset: str
    home_lane: str
    grounding_doc_id: str | None = None
    meaning_preserved: bool
    answer_key: AnswerKeyPath
    attempts: int


class Operator(ABC):
    """One augmentation operator: declaration + deterministic selection +
    the LLM's instruction + the verifiable postcondition (a `Targets`)."""

    declaration: ClassVar[Declaration]

    def serves(self, floor: str) -> bool:
        return floor.startswith(self.declaration.floor_prefix)

    @abstractmethod
    def eligible(self, selection: pd.DataFrame, floor: str) -> pd.DataFrame:
        """Parents that fit — a table filter (d42e), never an LLM."""

    @abstractmethod
    def instruction(self, floor: str) -> str:
        """System prompt: what to weave and what must not change."""

    @abstractmethod
    def targets(self, floor: str) -> Targets:
        """The postcondition local verify re-measures (d42g)."""

    def candidate(
        self, parent: pd.Series, floor: str, text: str, attempts: int
    ) -> AugmentedCandidate:
        """Assemble the pool row for inherit-path operators. Minted-path
        operators (Inject) override to attach the grounding doc."""
        slug = floor.replace(":", "-")
        return AugmentedCandidate(
            query_id=f"aug-{slug}-{parent['query_id']}",
            query=text,
            floor=floor,
            operator=self.declaration.operator,
            generated_from=str(parent["query_id"]),
            parent_dataset=str(parent["dataset"]),
            home_lane=str(parent["dataset"]),
            meaning_preserved=self.declaration.meaning_preserved,
            answer_key=self.declaration.answer_key,
            attempts=attempts,
        )
