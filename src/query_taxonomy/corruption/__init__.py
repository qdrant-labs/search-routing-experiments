from query_taxonomy.corruption.core import CorruptionBank
from query_taxonomy.corruption.general import EncodingArtifactBank

CORRUPTION_BANKS: tuple[type[CorruptionBank], ...] = (EncodingArtifactBank,)

__all__ = [
    "CORRUPTION_BANKS",
    "CorruptionBank",
    "EncodingArtifactBank",
]
