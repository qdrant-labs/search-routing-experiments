from typing import override

from query_taxonomy.core import FeatureGroup
from query_taxonomy.entities.core import EntityBank, Gliner2Bank
from query_taxonomy.taxonomy import LogicalStructure, StructuralIdentifier


class PersonLikeBank(EntityBank):
    """Person-like spans — audited precision 0.87 at threshold 0.32.
    Assumptive (MODEL, SPEC d22): the model is a soft classifier — audit
    precision doesn't hold on all corpora (short medical titles, lowercased
    web queries)."""

    @property
    @override
    def name(self) -> StructuralIdentifier:
        return StructuralIdentifier.PERSON_LIKE

    @property
    @override
    def label(self) -> str:
        return "person"

    @property
    @override
    def threshold(self) -> float:
        return 0.32


class LocationLikeBank(EntityBank):
    """Location-like spans — audited precision 0.84 at threshold 0.34.
    Assumptive (MODEL, SPEC d22)."""

    @property
    @override
    def name(self) -> StructuralIdentifier:
        return StructuralIdentifier.LOCATION_LIKE

    @property
    @override
    def label(self) -> str:
        return "location"

    @property
    @override
    def threshold(self) -> float:
        return 0.34


class ProperNounLikeBank(EntityBank):
    """Coarse name-like spans — audited 0.84 (0.77 on all-lowercase text)
    at threshold 0.31. Covers dropped org/product mentions coarsely.
    Assumptive (MODEL, SPEC d22) — coarse net widener by design."""

    @property
    @override
    def name(self) -> StructuralIdentifier:
        return StructuralIdentifier.PROPER_NOUN_LIKE

    @property
    @override
    def label(self) -> str:
        return "proper noun"

    @property
    @override
    def threshold(self) -> float:
        return 0.31


class TemporalLikeBank(Gliner2Bank):
    """The model layer of the layered temporal feature (SPEC d22): backstops
    the RIGID regex `TemporalBank` at a DIFFERENT emit name — the regex bank
    claims first as `temporal`; anything left over that this model catches
    emits as `temporal_like` so the confidence gap is visible in the name.
    Audited 0.83 at 0.58."""

    @property
    @override
    def group(self) -> FeatureGroup:
        return FeatureGroup.LOGICAL_STRUCTURES

    @property
    @override
    def name(self) -> LogicalStructure:
        return LogicalStructure.TEMPORAL_LIKE

    @property
    @override
    def label(self) -> str:
        return "temporal"

    @property
    @override
    def threshold(self) -> float:
        return 0.58
