from abc import ABC, abstractmethod
from functools import lru_cache
from typing import TYPE_CHECKING

from query_taxonomy.core import (
    AmbiguityTier,
    Engine,
    FeatureGroup,
    FeatureSpan,
    GeneralBank,
)

if TYPE_CHECKING:
    from gliner2 import GLiNER2

MODEL_ID = "fastino/gliner2-base-v1"

# GLiNER2 outputs are schema-composition-dependent (SPEC decision 13):
# adding or removing a label shifts every other label's scores. This dict is
# therefore part of the determinism pin — every gliner2 bank reads from the
# SAME schema and the SAME forward pass, never a private one.
PINNED_SCHEMA: dict[str, str] = {
    "person": "name of a person or people",
    "location": (
        "name of a place such as a city, country, region, or landmark"
    ),
    "proper noun": (
        "a proper noun in any casing — the name of a specific person, "
        "place, organization, brand, title, or work"
    ),
    "temporal": (
        "a date, time, or temporal expression, absolute or relative, "
        "such as 1995, yesterday, q3 2024, the 90s, next week"
    ),
}

# extraction floor: per-bank thresholds filter upward from here
_FLOOR = 0.3


@lru_cache(maxsize=1)
def _engine() -> "GLiNER2":
    # lazy: torch lives in the optional `model` dependency group
    from gliner2 import GLiNER2

    return GLiNER2.from_pretrained(MODEL_ID)


@lru_cache(maxsize=4096)
def _extract(text: str) -> tuple[tuple[str, FeatureSpan, float], ...]:
    """One forward pass per text over the pinned schema, shared by all
    gliner2 banks. One text per call keeps batch composition fixed
    (near-threshold float determinism). Offsets are clamped to len(text)
    and mismatching spans dropped — the model hallucinates a trailing
    period after state-abbreviation-shaped tails (smoke eval)."""
    result = _engine().extract_entities(
        text,
        PINNED_SCHEMA,
        threshold=_FLOOR,
        include_spans=True,
        include_confidence=True,
    )
    outputs: list[tuple[str, FeatureSpan, float]] = []
    for label, spans in result.get("entities", {}).items():
        for span in spans:
            end = min(span["end"], len(text))
            surface = text[span["start"] : end]
            if not surface or not span["text"].startswith(surface):
                continue
            outputs.append(
                (label, FeatureSpan(surface, span["start"], end), span["confidence"])
            )
    return tuple(outputs)


class Gliner2Bank(GeneralBank[FeatureSpan, str], ABC):
    """
    Model-engine span bank over the shared pinned GLiNER2 schema. Always
    AMBIGUOUS: the model layer claims after every deterministic bank in its
    group (layered-banks doctrine, SPEC decision 16). Per-bank confidence
    thresholds come from the hand-audited smoke eval (SPEC decision 13).
    """

    engine = Engine.GLINER

    def __init__(self) -> None:
        super().__init__()
        self._description = self.define(PINNED_SCHEMA[self.label])

    @property
    @abstractmethod
    def label(self) -> str:
        """Key into PINNED_SCHEMA this bank reads from the shared pass."""

    @property
    @abstractmethod
    def threshold(self) -> float:
        """Audited per-label confidence cutoff."""

    @property
    def ambiguity(self) -> AmbiguityTier:
        return AmbiguityTier.AMBIGUOUS

    def define(self, builder: str) -> str:
        """The definition artifact is the pinned label description; banks
        must not alter it — the schema is composition-pinned."""
        return builder

    def compute(self, text: str) -> list[FeatureSpan]:
        return [
            span
            for label, span, confidence in _extract(text)
            if label == self.label and confidence >= self.threshold
        ]


class EntityBank(Gliner2Bank, ABC):
    """Named entities join the Structured Identifiers group (the CSV places
    the Named Entity feature there): regex identifier banks claim first, so
    CVE-2024-3094 is never re-tagged as a proper noun."""

    @property
    def group(self) -> FeatureGroup:
        return FeatureGroup.STRUCTURED_IDENTIFIERS
