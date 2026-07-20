import re
from abc import ABC, abstractmethod

from edify import RegexBuilder

from query_taxonomy.core import Engine, FeatureGroup, StatBank
from query_taxonomy.taxonomy import StatisticalMetric


class MetricBank(StatBank[RegexBuilder], ABC):
    """
    Token-regex stat bank: `define` builds the tokenizer pattern the stats
    are computed over. Inherits the RIGID ambiguity default — stats are
    solid numbers.
    """

    engine = Engine.REGEX

    def __init__(self) -> None:
        super().__init__()
        self._tokens: re.Pattern[str] = self.define(RegexBuilder()).to_regex()

    @property
    def group(self) -> FeatureGroup:
        return FeatureGroup.STATISTICAL_METRICS

    @property
    @abstractmethod
    def name(self) -> StatisticalMetric:
        """
        Name of the current bank
        """

    def tokens(self, text: str) -> list[str]:
        return self._tokens.findall(text)
