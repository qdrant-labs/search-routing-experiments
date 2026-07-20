from typing import override

from edify import RegexBuilder

from query_taxonomy.core import AmbiguityTier
from query_taxonomy.corruption.core import CorruptionBank
from query_taxonomy.taxonomy import CorruptionKind

# "â€" alone covers the whole curly-quote/dash mojibake family (â€™ â€œ â€“)
ARTIFACT_SEQUENCES: tuple[str, ...] = (
    "�",
    "â€",
    "â‚¬",
    "Ã©",
    "Ã¨",
    "Ã¼",
    "Ã¶",
    "Ã¤",
    "Ã±",
    "Ã¡",
    "Ã­",
    "Ã³",
    "Ãº",
    "ÃŸ",
)


class EncodingArtifactBank(CorruptionBank):
    """UTF-8-as-Latin-1 mojibake digraphs and the U+FFFD replacement char —
    tokens in neither query intent nor corpus vocabulary."""

    @property
    @override
    def name(self) -> CorruptionKind:
        return CorruptionKind.ENCODING_ARTIFACT

    @property
    @override
    def ambiguity(self) -> AmbiguityTier:
        return AmbiguityTier.RIGID

    @override
    def define(self, builder: RegexBuilder) -> RegexBuilder:
        chain = builder.any_of()
        for sequence in sorted(ARTIFACT_SEQUENCES, key=len, reverse=True):
            chain = chain.string(sequence)
        return chain.end()
