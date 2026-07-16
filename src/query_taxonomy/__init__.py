from query_taxonomy.banks import BANKS
from query_taxonomy.core import FeatureStat, RegexBank, StatBank
from query_taxonomy.markers import MARKER_BANKS
from query_taxonomy.taxonomy import FeatureGroup


Bank = RegexBank | StatBank[FeatureStat]
BankTypes = type[Bank]

# Must be defined before any `features` re-export: features.py imports this
# dict from the package, so it has to exist by the time that module loads.
FEATURE_BANKS: dict[FeatureGroup, tuple[BankTypes, ...]] = {
    FeatureGroup.STRUCTURED_IDENTIFIERS: BANKS,
    FeatureGroup.SENTENCE_MARKERS: MARKER_BANKS,
}

__all__ = ["FEATURE_BANKS", "Bank", "BankTypes"]
