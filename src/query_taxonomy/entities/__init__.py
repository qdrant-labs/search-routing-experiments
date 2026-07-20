from query_taxonomy.entities.core import EntityBank, Gliner2Bank
from query_taxonomy.entities.general import (
    LocationLikeBank,
    PersonLikeBank,
    ProperNounLikeBank,
    TemporalLikeBank,
)

ENTITY_BANKS: tuple[type[EntityBank], ...] = (
    PersonLikeBank,
    LocationLikeBank,
    ProperNounLikeBank,
)

__all__ = [
    "ENTITY_BANKS",
    "EntityBank",
    "Gliner2Bank",
    "LocationLikeBank",
    "PersonLikeBank",
    "ProperNounLikeBank",
    "TemporalLikeBank",
]
