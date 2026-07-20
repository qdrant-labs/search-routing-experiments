from typing import Any

from query_taxonomy.banks import BANKS
from query_taxonomy.core import Engine, GeneralBank
from query_taxonomy.corruption import CORRUPTION_BANKS
from query_taxonomy.entities import ENTITY_BANKS, Gliner2TemporalBank
from query_taxonomy.logical import LOGICAL_BANKS
from query_taxonomy.markers import MARKER_BANKS
from query_taxonomy.metrics import METRIC_BANKS
from query_taxonomy.metrics.pos import (
    MorphologyBank,
    PosProfileBank,
    SyntacticDepthBank,
)
from query_taxonomy.taxonomy import FeatureGroup


# GeneralBank is the common root: RegexBank, StatBank, and Gliner2Bank are
# siblings under it, so listing only two would miss the GLiNER family and
# the spaCy StatBank["Language"] specialization.
Bank = GeneralBank[Any, Any]
BankTypes = type[Bank]

# ALL banks of every engine live here — model-engine banks import their
# heavy dependencies lazily (at instantiation), so this registry stays
# import-light. Selection happens in FeatureExtractor via the `engines`
# filter, BEFORE instantiation.
#
# Must be defined before any `features` re-export: features.py imports this
# dict from the package, so it has to exist by the time that module loads.
FEATURE_BANKS = {
    FeatureGroup.STRUCTURED_IDENTIFIERS: BANKS + ENTITY_BANKS,
    FeatureGroup.SENTENCE_MARKERS: MARKER_BANKS,
    FeatureGroup.LOGICAL_STRUCTURES: LOGICAL_BANKS + (Gliner2TemporalBank,),
    FeatureGroup.CORRUPTION: CORRUPTION_BANKS,
    FeatureGroup.STATISTICAL_METRICS: METRIC_BANKS
    + (PosProfileBank, MorphologyBank, SyntacticDepthBank),
}

__all__ = ["FEATURE_BANKS", "Bank", "BankTypes", "Engine"]
