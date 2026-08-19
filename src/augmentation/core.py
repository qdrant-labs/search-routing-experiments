"""Operator declarations and the pool-row contract (SPEC d42c/d).

Operators are surface_origin-aware — they read the order sheet, pick parents,
and decide the answer-key path — which is why they live here and not in
surface_origin-blind taxonomy_generators (d34a/d42b). Every operator carries a
frozen `Declaration`; anything undeclared is feature-stock by default.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from enum import StrEnum
from typing import TYPE_CHECKING, ClassVar

import pandas as pd
from pydantic import BaseModel, ConfigDict

from augmentation.config import AugmentationConfig
from taxonomy_generators.verify import Targets

if TYPE_CHECKING:  # a type only — keeps the base class off composition's graph
    from composition.cells import AxisBand
    from augmentation.engine import AugmentationOutcome


class SurfaceOrigin(StrEnum):
    NONE = "none"
    DOC_COPIED = "doc_copied"
    SYNTHETIC = "synthetic"

    @property
    def provenance(self) -> str:
        """The selection's provenance value for a row this origin produced
        (d51k) — no document consulted still means the query changed."""
        return {
            SurfaceOrigin.NONE: "augmented",
            SurfaceOrigin.DOC_COPIED: "doc_grounded",
            SurfaceOrigin.SYNTHETIC: "synthetic",
        }[self]


class AnswerKeyPath(StrEnum):
    INHERIT = "inherit"   # meaning preserved -> parent qrels stay valid
    MINTED = "minted"     # answer minted from the surface_origin doc


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
    floors: str
    """Human-readable statement of the floors served — the dispatch
    itself lives in `Operator.serves()` (band parsing, membership
    checks), never in string prefixes."""
    surface_origin: SurfaceOrigin
    answer_key: AnswerKeyPath
    meaning_preserved: bool
    verifiable_by: str
    credit_gate: CreditGate
    tool_loop: bool
    """Engine mode (d42n): False = single-shot (one completion, local
    accept, retry on feedback); True = agentic tool loop for operators
    that chase ranges and need in-loop measuring."""


class AugmentedCandidate(BaseModel):
    """One accepted row awaiting the mini-fill — born with its d40(c)
    certificate. `generated_from` is the lineage edge (d42j); the
    parent-side `superseded` view is derived, never stored."""

    model_config = ConfigDict(frozen=True)

    query_id: str
    query: str
    floor: str
    operator: str
    provenance: str
    """Set from the operator's `surface_origin` at birth — the selection reads
    it rather than re-deriving it from the operator name (d51k)."""
    generated_from: str
    parent_dataset: str
    home_lane: str
    grounding_doc_id: str | None = None
    meaning_preserved: bool
    answer_key: AnswerKeyPath
    attempts: int
    hops: int = 0
    """`completion()` round-trips spent on this row — see
    `AugmentationOutcome.hops`; a cell's own call sequence (additions, then
    the cut) can cost more than one hop per attempt."""
    tokens: int = 0
    elapsed_s: float = 0.0
    credit_gate: str = "none"
    """The operator's gate at creation (d42h): rows born behind an open
    gate ('none') are admissible; gated rows are feature-stock until
    their audit passes and flips them."""


NOTHING_ELSE = (
    "Add nothing beyond what is asked above: no other facts, names, numbers, "
    "dates, identifiers, greetings, or politeness phrases — anything extra "
    "changes the query's feature profile and fails the check."
)
"""The one exclusion clause, emitted once per request by
`AugmentationLoop.instruction_for` (d52d). Operators state the positive move
only: composed mints repeating their own exclusions told the model to insert a
version string AND to add no numbers."""


class Operator(ABC):
    """One augmentation operator: declaration + deterministic selection +
    the LLM's instruction + the verifiable postcondition (a `Targets`)."""

    declaration: ClassVar[Declaration]

    def __init__(self, config: AugmentationConfig) -> None:
        self.first_generation_only = config.first_generation_only
        """d43-review guard: augmented rows are never parents — no
        second-generation drift — unless deliberately switched off."""

    def parent_pool(self, selection: pd.DataFrame) -> pd.DataFrame:
        """The rows eligibility may draw from, honoring the
        first-generation guard. Every eligible() starts here."""
        if self.first_generation_only and "generated_from" in selection.columns:
            return selection[selection["generated_from"].isna()]
        return selection

    def unsatisfied(self, pool: pd.DataFrame, floor: str) -> pd.DataFrame:
        """Parents a FLOOR demand does not already cover, read off the
        selection's `floors` membership column. A cell pool carries no such
        column and needs no filter: `dispatch.predicate_minus_one` has already
        dropped every row that satisfies the cell."""
        if "floors" not in pool.columns:
            return pool
        return pool[~pool["floors"].map(lambda floors: floor in floors)]

    @abstractmethod
    def mints(self, band: AxisBand) -> bool:
        """Whether this operator can put this band's feature INTO a query —
        the declaration cell dispatch derives from (d51c). Abstract for the
        same reason `structural` is: a family that mints nothing says so."""

    @abstractmethod
    def serves(self, floor: str) -> bool:
        """Whether this operator serves the floor — every operator states
        its own dispatch (the declaration's `floors` field is the
        human-readable twin)."""

    @abstractmethod
    def eligible(self, selection: pd.DataFrame, floor: str) -> pd.DataFrame:
        """Parents that fit, in preference order — a table filter (d42e),
        never an LLM."""

    def apply(
        self,
        parent: pd.Series,
        floor: str,
        text: str,
        requirement: tuple[AxisBand, ...] = (),
    ) -> str | None:
        """The rewritten text when this operator needs no model, else None and
        the row goes to the LLM. Takes the working text rather than the
        parent's query because a cell's calls feed each other."""
        return None

    @abstractmethod
    def instruction(
        self,
        floor: str,
        parent: pd.Series,
        requirement: tuple[AxisBand, ...] = (),
    ) -> str:
        """System prompt: the positive move only — `NOTHING_ELSE` carries the
        exclusion for the whole request. Receives the parent row so operators
        can adapt framing to its profile, and the cell requirement this call
        serves so a band never has to be looked up in a global registry."""

    @abstractmethod
    def targets(
        self,
        floor: str,
        parent: pd.Series,
        requirement: tuple[AxisBand, ...] = (),
    ) -> Targets:
        """The postcondition local verify re-measures (d42g). Receives the
        parent because some postconditions are pair-specific (Inject's
        target bank is the chosen surface's bank)."""

    @abstractmethod
    def structural(
        self, parent: pd.Series, text: str, targets: Targets
    ) -> list[str]:
        """Parent-relative checks Targets cannot express (d43b):
        no-new-spans, content tokens unchanged, literal containment.
        `targets` is what the request AUTHORISED, so a check reads "new" as
        "nobody asked for it" and composed mints cannot veto each other
        (d52d). Returns failure reasons; empty = pass. Run by the loop after
        accept(); any failure drops the row. Abstract on purpose (d42c
        default-deny): an operator with no such check DECLARES that with
        an explicit `return []` and its reason — never inherits silence."""

    def candidate(
        self, parent: pd.Series, floor: str, outcome: "AugmentationOutcome"
    ) -> AugmentedCandidate:
        """Assemble the pool row for inherit-path operators. Minted-path
        operators (Inject) override to attach the surface_origin doc. Takes
        the whole outcome, not `text`/`attempts` loose — `hops`/`tokens`/
        `elapsed_s` live there too, and passing them as separate params would
        just be the same fact carried twice."""
        slug = floor.replace(":", "-")
        return AugmentedCandidate(
            provenance=self.declaration.surface_origin.provenance,
            query_id=f"aug-{slug}-{parent['query_id']}",
            query=outcome.text,
            floor=floor,
            operator=self.declaration.operator,
            generated_from=str(parent["query_id"]),
            parent_dataset=str(parent["dataset"]),
            home_lane=str(parent["dataset"]),
            meaning_preserved=self.declaration.meaning_preserved,
            answer_key=self.declaration.answer_key,
            attempts=outcome.attempts,
            hops=outcome.hops,
            tokens=outcome.tokens,
            elapsed_s=outcome.elapsed_s,
            credit_gate=str(self.declaration.credit_gate),
        )
