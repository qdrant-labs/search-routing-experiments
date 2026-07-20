from typing import override

from edify import RegexBuilder

from query_taxonomy.core import FeatureStat
from query_taxonomy.metrics.core import MetricBank
from query_taxonomy.taxonomy import StatisticalMetric


class LengthBank(MetricBank):
    """Query size: token and character counts."""

    @property
    @override
    def name(self) -> StatisticalMetric:
        return StatisticalMetric.LENGTH

    @override
    def define(self, builder: RegexBuilder) -> RegexBuilder:
        return builder.one_or_more().word()

    @override
    def compute(self, text: str) -> list[FeatureStat]:
        return [
            FeatureStat("length_words", float(len(self.tokens(text)))),
            FeatureStat("length_chars", float(len(text))),
        ]


STOPWORDS: frozenset[str] = frozenset(
    """a an the and or but if of to in on at for with by from as is are was
    were be been being am do does did have has had will would should could
    can may might must this that these those it its i you he she we they me
    him her us them my your his our their what which who whom whose when
    where why how not no nor so than too very just there about into over
    under between through during before after""".split()
)


class StopwordRatioBank(MetricBank):
    """Function-word share of the query — the REGEX fallback of the POS
    profile's closed_class_share (SPEC decision 14). High ratio = natural
    phrasing; near-zero = keyword telegram."""

    @property
    @override
    def name(self) -> StatisticalMetric:
        return StatisticalMetric.STOPWORD_RATIO

    @override
    def define(self, builder: RegexBuilder) -> RegexBuilder:
        return builder.one_or_more().word()

    @override
    def compute(self, text: str) -> list[FeatureStat]:
        tokens = self.tokens(text)
        stopwords = sum(token.lower() in STOPWORDS for token in tokens)
        ratio = stopwords / len(tokens) if tokens else 0.0
        return [
            FeatureStat("stopword_count", float(stopwords)),
            FeatureStat("stopword_ratio", ratio),
        ]
