"""spaCy-engine stat banks: POS profile, morphology, syntactic depth.
All three share ONE cached pipeline (tagger + parser + lemmatizer, ner
disabled) — the pinned model is part of the determinism pin (SPEC d14/d17).
spaCy is a main dependency; the model needs a separate download
(`poetry run python -m spacy download en_core_web_sm`)."""

from abc import ABC
from functools import lru_cache
from typing import TYPE_CHECKING, override

from query_taxonomy.core import Engine, FeatureGroup, FeatureStat, StatBank
from query_taxonomy.taxonomy import StatisticalMetric

if TYPE_CHECKING:
    from spacy.language import Language
    from spacy.tokens import Doc, Token

SPACY_MODEL = "en_core_web_sm"

UD_TAGS = (
    "ADJ", "ADP", "ADV", "AUX", "CCONJ", "DET", "INTJ", "NOUN", "NUM",
    "PART", "PRON", "PROPN", "PUNCT", "SCONJ", "SYM", "VERB", "X",
)
OPEN_CLASS = frozenset({"ADJ", "ADV", "INTJ", "NOUN", "PROPN", "VERB"})
CLOSED_CLASS = frozenset(
    {"ADP", "AUX", "CCONJ", "DET", "NUM", "PART", "PRON", "SCONJ"}
)
CLAUSAL_DEPS = frozenset(
    {"ROOT", "ccomp", "xcomp", "advcl", "acl", "relcl", "csubj"}
)


@lru_cache(maxsize=1)
def _pipeline() -> "Language":
    # lazy: keeps regex-only extractor construction import-light
    import spacy

    return spacy.load(SPACY_MODEL, disable=["ner"])


@lru_cache(maxsize=4096)
def _doc(text: str) -> "Doc":
    """One parse per text, shared by every spaCy bank — mirrors the shared
    GLiNER2 forward pass."""
    return _pipeline()(text)


class SpacyBank(StatBank["Language"], ABC):
    """Stat bank over the shared pinned spaCy pipeline."""

    engine = Engine.SPACY

    def __init__(self) -> None:
        super().__init__()
        self._nlp = self.define(_pipeline())

    @property
    @override
    def group(self) -> FeatureGroup:
        return FeatureGroup.STATISTICAL_METRICS

    @override
    def define(self, builder: "Language") -> "Language":
        """The definition artifact is the shared, component-trimmed
        pipeline; banks must not mutate it — the model is pinned."""
        return builder


class PosProfileBank(SpacyBank):
    """UD-17 POS histogram + derived shares (SPEC decision 14).
    closed_class_share is the canonical function-word measure; the
    stopword-ratio bank is its dependency-free fallback."""

    @property
    @override
    def name(self) -> StatisticalMetric:
        return StatisticalMetric.POS_PROFILE

    @override
    def compute(self, text: str) -> list[FeatureStat]:
        tokens = [token for token in _doc(text) if not token.is_space]
        total = len(tokens)
        counts = {tag: 0 for tag in UD_TAGS}
        for token in tokens:
            if token.pos_ in counts:
                counts[token.pos_] += 1

        def share(count: int) -> float:
            return count / total if total else 0.0

        stats = [
            FeatureStat(f"pos_{tag.lower()}", float(count))
            for tag, count in counts.items()
        ]
        stats.extend(
            [
                FeatureStat(
                    "open_class_share",
                    share(sum(counts[tag] for tag in OPEN_CLASS)),
                ),
                FeatureStat(
                    "closed_class_share",
                    share(sum(counts[tag] for tag in CLOSED_CLASS)),
                ),
                FeatureStat("noun_share", share(counts["NOUN"])),
                FeatureStat("verb_presence", float(counts["VERB"] > 0)),
                FeatureStat("propn_share", share(counts["PROPN"])),
            ]
        )
        return stats


class MorphologyBank(SpacyBank):
    """Inflection profile: tokens whose lemma differs from their surface
    form — the grammar-caused half of vocabulary mismatch (CSV row 7)."""

    @property
    @override
    def name(self) -> StatisticalMetric:
        return StatisticalMetric.MORPHOLOGY

    @override
    def compute(self, text: str) -> list[FeatureStat]:
        words = [token for token in _doc(text) if token.is_alpha]
        inflected = sum(
            token.lemma_.lower() != token.text.lower() for token in words
        )
        share = inflected / len(words) if words else 0.0
        return [
            FeatureStat("inflected_count", float(inflected)),
            FeatureStat("inflected_share", share),
        ]


def _depth(token: "Token") -> int:
    depth = 0
    while token.head is not token:
        token = token.head
        depth += 1
    return depth


class SyntacticDepthBank(SpacyBank):
    """Compositional structure: max dependency-tree depth and clause count
    (CSV row 18). Deep structure = meaning a bag-of-words loses; flat
    structure = keyword telegram."""

    @property
    @override
    def name(self) -> StatisticalMetric:
        return StatisticalMetric.SYNTACTIC_DEPTH

    @override
    def compute(self, text: str) -> list[FeatureStat]:
        tokens = [token for token in _doc(text) if not token.is_space]
        depth = max((_depth(token) for token in tokens), default=0)
        clauses = sum(token.dep_ in CLAUSAL_DEPS for token in tokens)
        return [
            FeatureStat("parse_depth", float(depth)),
            FeatureStat("clause_count", float(clauses)),
        ]
