from typing import override

from query_taxonomy.core import FeatureGroup
from query_taxonomy.entities.core import EntityBank, Gliner2Bank
from query_taxonomy.taxonomy import LogicalStructure, StructuralIdentifier


class PersonBank(EntityBank):
    """Person names — audited precision 0.87 at threshold 0.32."""

    @property
    @override
    def name(self) -> StructuralIdentifier:
        return StructuralIdentifier.PERSON

    @property
    @override
    def label(self) -> str:
        return "person"

    @property
    @override
    def threshold(self) -> float:
        return 0.32


class LocationBank(EntityBank):
    """Place names — audited precision 0.84 at threshold 0.34."""

    @property
    @override
    def name(self) -> StructuralIdentifier:
        return StructuralIdentifier.LOCATION

    @property
    @override
    def label(self) -> str:
        return "location"

    @property
    @override
    def threshold(self) -> float:
        return 0.34


class ProperNounBank(EntityBank):
    """Coarse name-like spans — audited 0.84 (0.77 on all-lowercase text)
    at threshold 0.31. Covers dropped org/product mentions coarsely."""

    @property
    @override
    def name(self) -> StructuralIdentifier:
        return StructuralIdentifier.PROPER_NOUN

    @property
    @override
    def label(self) -> str:
        return "proper noun"

    @property
    @override
    def threshold(self) -> float:
        return 0.31


class Gliner2TemporalBank(Gliner2Bank):
    """The model layer of the layered TEMPORAL feature: backstops the
    relative-vocabulary regex bank at the same feature name — the claim
    registry gives the regex layer priority. Audited 0.83 at 0.58."""

    @property
    @override
    def group(self) -> FeatureGroup:
        return FeatureGroup.LOGICAL_STRUCTURES

    @property
    @override
    def name(self) -> LogicalStructure:
        return LogicalStructure.TEMPORAL

    @property
    @override
    def label(self) -> str:
        return "temporal"

    @property
    @override
    def threshold(self) -> float:
        return 0.58
