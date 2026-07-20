from typing import override

from edify import RegexBuilder

from query_taxonomy.core import AmbiguityTier
from query_taxonomy.markers.core import MarkerBank, phrase_alternation
from query_taxonomy.taxonomy import SentenceMarker

NEGATION_PHRASES: tuple[str, ...] = (
    "not",
    "no",
    "never",
    "without",
    "except",
    "excluding",
    "excluded",
    "neither",
    "nor",
    "cannot",
    "don't",
    "doesn't",
    "isn't",
    "won't",
    "can't",
    "versus",
    "vs",
)


class NegationBank(MarkerBank):
    """Negation / exclusion cues — the constraint sparse bag-of-words cannot
    represent. "no" is a deliberate high-recall inclusion despite "no. 5"
    style false positives."""

    @property
    @override
    def name(self) -> SentenceMarker:
        return SentenceMarker.NEGATION

    @property
    @override
    def ambiguity(self) -> AmbiguityTier:
        return AmbiguityTier.AMBIGUOUS

    @override
    def define(self, builder: RegexBuilder) -> RegexBuilder:
        return phrase_alternation(builder, NEGATION_PHRASES)


GREETING_PHRASES: tuple[str, ...] = (
    "hi",
    "hello",
    "hey",
    "howdy",
    "greetings",
    "yo",
    "good morning",
    "good afternoon",
    "good evening",
)


class GreetingBank(MarkerBank):
    """Conversational openers — retrieval noise that signals NL register."""

    @property
    @override
    def name(self) -> SentenceMarker:
        return SentenceMarker.GREETING

    @property
    @override
    def ambiguity(self) -> AmbiguityTier:
        return AmbiguityTier.MODERATE

    @override
    def define(self, builder: RegexBuilder) -> RegexBuilder:
        return phrase_alternation(builder, GREETING_PHRASES)


POLITENESS_PHRASES: tuple[str, ...] = (
    "please",
    "thanks",
    "thank you",
    "kindly",
    "could you",
    "would you",
    "can you",
    "may i",
    "excuse me",
)


class PolitenessBank(MarkerBank):
    """Politeness / request markers — distinguish user-written from
    programmatic queries."""

    @property
    @override
    def name(self) -> SentenceMarker:
        return SentenceMarker.POLITENESS

    @property
    @override
    def ambiguity(self) -> AmbiguityTier:
        return AmbiguityTier.MODERATE

    @override
    def define(self, builder: RegexBuilder) -> RegexBuilder:
        return phrase_alternation(builder, POLITENESS_PHRASES)


INTERJECTION_PHRASES: tuple[str, ...] = (
    "wow",
    "ugh",
    "oh",
    "ah",
    "hmm",
    "yay",
    "ouch",
    "omg",
    "lol",
    "huh",
    "uh",
    "um",
    "wtf",
    "oops",
    "yikes",
)


class InterjectionBank(MarkerBank):
    """Interjections — noise tokens that dilute sparse signal."""

    @property
    @override
    def name(self) -> SentenceMarker:
        return SentenceMarker.INTERJECTION

    @property
    @override
    def ambiguity(self) -> AmbiguityTier:
        return AmbiguityTier.AMBIGUOUS

    @override
    def define(self, builder: RegexBuilder) -> RegexBuilder:
        return phrase_alternation(builder, INTERJECTION_PHRASES)


COMPARATIVE_PHRASES: tuple[str, ...] = (
    "better",
    "best",
    "worse",
    "worst",
    "more",
    "most",
    "less",
    "least",
    "than",
)


class ComparativeBank(MarkerBank):
    """Comparative / superlative markers and irregulars. Suffix morphology
    (-er/-est) deliberately excluded: water/forest-class false positives."""

    @property
    @override
    def name(self) -> SentenceMarker:
        return SentenceMarker.COMPARATIVE

    @property
    @override
    def ambiguity(self) -> AmbiguityTier:
        return AmbiguityTier.AMBIGUOUS

    @override
    def define(self, builder: RegexBuilder) -> RegexBuilder:
        return phrase_alternation(builder, COMPARATIVE_PHRASES)


class AcronymBank(MarkerBank):
    """Cased acronym shapes: bare uppercase runs (NASA) and dotted forms
    (N.Y.). Case-sensitive by design — lowercase acronyms are shape-invisible
    and belong to the fine-tuned model path (SPEC decision 13)."""

    @property
    @override
    def name(self) -> SentenceMarker:
        return SentenceMarker.ACRONYM

    @property
    @override
    def ambiguity(self) -> AmbiguityTier:
        return AmbiguityTier.AMBIGUOUS

    @override
    def define(self, builder: RegexBuilder) -> RegexBuilder:
        return (
            builder
            .any_of()
                # dotted before bare: the consuming branch goes first
                .group()
                    .word_boundary()
                    .at_least(2).group().range("A", "Z").char(".").end()
                .end()
                .group()
                    .word_boundary()
                    .between(2, 6).range("A", "Z")
                    .word_boundary()
                .end()
            .end()
        )
