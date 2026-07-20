from query_taxonomy.entities.core import EntityBank, Gliner2Bank
from query_taxonomy.entities.general import (
    Gliner2TemporalBank,
    LocationBank,
    PersonBank,
    ProperNounBank,
)

ENTITY_BANKS: tuple[type[EntityBank], ...] = (
    PersonBank,
    LocationBank,
    ProperNounBank,
)

__all__ = [
    "ENTITY_BANKS",
    "EntityBank",
    "Gliner2Bank",
    "Gliner2TemporalBank",
    "LocationBank",
    "PersonBank",
    "ProperNounBank",
]
