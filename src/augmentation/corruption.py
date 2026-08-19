"""Deterministic query-junking: the perturbations that PRODUCE corrupted
supply for the corruption detectors to find. A degree is a delta over the
parent's OWN span count, and the detector a degree spends its budget on is
whichever has the most headroom in the parent's lane — a lane where typo
already fires on 46% of queries learns nothing from one more typo.
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from enum import StrEnum
from functools import lru_cache
from pathlib import Path
from random import Random
from typing import TYPE_CHECKING, ClassVar

import pandas as pd

from augmentation.config import AugmentationConfig
from augmentation.core import (
    AnswerKeyPath,
    CreditGate,
    Declaration,
    Operator,
    SurfaceOrigin,
)
from query_taxonomy.core import Engine
from query_taxonomy.features import FeatureExtractor
from query_taxonomy.taxonomy import CorruptionKind, FeatureGroup
from taxonomy_generators.verify import SpanTarget, Targets

if TYPE_CHECKING:
    from collections.abc import Callable

    from composition.cells import AxisBand

_LANG = "en"
_WORD = re.compile(r"[^\W\d_]+")
_FREQUENT_ZIPF = 3.0
"""Mirrors the detector's own bar: only a word this common has a one-edit
neighbour the typo bank will recognise as a misspelling of something."""
_MIN_TYPO_LEN, _MAX_TYPO_LEN = 3, 25

_QWERTY: dict[str, str] = {
    "q": "wa", "w": "qes", "e": "wrd", "r": "etf", "t": "ryg", "y": "tuh",
    "u": "yij", "i": "uok", "o": "ipl", "p": "ol", "a": "qsz", "s": "awdx",
    "d": "sefc", "f": "drgv", "g": "fthb", "h": "gyjn", "j": "hukm",
    "k": "jil", "l": "kop", "z": "asx", "x": "zsdc", "c": "xdfv",
    "v": "cfgb", "b": "vghn", "n": "bhjm", "m": "njk",
}
_ACCENTS = {"a": "á", "e": "é", "i": "í", "o": "ó",
            "u": "ú", "c": "ç", "n": "ñ"}


@lru_cache(maxsize=1)
def _zipf() -> Callable[..., float]:
    from wordfreq import zipf_frequency

    return zipf_frequency


class CorruptionDegree(StrEnum):
    """How far past its own natural noise a row is pushed, counted in
    detector spans rather than a hand-set character rate."""

    CLEAN = "clean"
    LIGHT = "light"
    HEAVY = "heavy"

    @property
    def added_spans(self) -> int:
        return {"clean": 0, "light": 1, "heavy": 3}[self.value]


class LaneBaseline:
    """Natural per-lane detector rates, read off the corruption census.
    Missing census or unseen lane reads as zero — every detector has full
    headroom until measurement says otherwise."""

    def __init__(self, census_path: Path) -> None:
        self._rates = (
            pd.read_parquet(census_path)
            if census_path.exists()
            else pd.DataFrame()
        )

    def rate(self, lane: str, kind: CorruptionKind) -> float:
        if lane not in self._rates.index or kind.value not in self._rates.columns:
            return 0.0
        return float(self._rates.at[lane, kind.value])

    def by_headroom(self, lane: str) -> tuple[CorruptionKind, ...]:
        """Detectors rarest-natural first — the census's whole job here."""
        return tuple(sorted(PERTURBED_KINDS, key=lambda k: self.rate(lane, k)))


class Perturbation(ABC):
    """One deterministic damage move, aimed at exactly one detector."""

    kind: ClassVar[CorruptionKind]
    repeatable: ClassVar[bool] = False
    """Whether a second application still reads as damage — only typos do;
    re-encoding mojibake nests into unreadable garbage and a second cut past
    an end-anchored ellipsis adds nothing."""

    @abstractmethod
    def apply(self, text: str, rng: Random) -> str:
        """The damaged text, or `text` unchanged when nothing is eligible."""


class QwertyTypo(Perturbation):
    """Swap one letter for a QWERTY neighbour inside a frequent word, and
    verify the result left the frequency table — the exact shape TypoBank
    recognises (absent, yet one edit from something common)."""

    kind = CorruptionKind.TYPO
    repeatable = True

    def apply(self, text: str, rng: Random) -> str:
        zipf = _zipf()
        words = [
            m for m in _WORD.finditer(text)
            if _MIN_TYPO_LEN <= len(m.group()) <= _MAX_TYPO_LEN
            and zipf(m.group().lower(), _LANG) >= _FREQUENT_ZIPF
        ]
        for match in rng.sample(words, len(words)):
            word = match.group()
            for i in rng.sample(range(len(word)), len(word)):
                for neighbour in _QWERTY.get(word[i].lower(), ""):
                    candidate = word[:i] + neighbour + word[i + 1:]
                    if zipf(candidate.lower(), _LANG) == 0.0:
                        return text[:match.start()] + candidate + text[match.end():]
        return text


class Mojibake(Perturbation):
    """Accent one letter, then mis-decode the query UTF-8-as-Latin-1 — the
    café -> cafÃ© round trip, the only source of encoding artifacts the
    census found (0.000 natural on 41 of 42 lanes)."""

    kind = CorruptionKind.ENCODING_ARTIFACT

    def apply(self, text: str, rng: Random) -> str:
        sites = [i for i, char in enumerate(text) if char.lower() in _ACCENTS]
        if not sites:
            return text
        i = rng.choice(sites)
        accented = text[:i] + _ACCENTS[text[i].lower()] + text[i + 1:]
        return accented.encode("utf-8").decode("latin-1")


class Truncate(Perturbation):
    """Cut the final word mid-way and mark the cut, which is the only
    truncation the detector can see — an unmarked cut is invisible by
    design (SPEC C6)."""

    kind = CorruptionKind.TRUNCATION

    def apply(self, text: str, rng: Random) -> str:
        stripped = text.rstrip()
        matches = list(_WORD.finditer(stripped))
        if not matches or stripped.endswith("..."):
            return text
        last = matches[-1]
        if len(last.group()) < 2:
            return text
        return stripped[: last.start() + rng.randrange(1, len(last.group()))] + "..."


PERTURBATIONS: tuple[Perturbation, ...] = (QwertyTypo(), Mojibake(), Truncate())
"""Word-order noise is deliberately absent: it has no detector (SPEC C6), so
it could never earn a degree's span credit."""

PERTURBED_KINDS: tuple[CorruptionKind, ...] = tuple(p.kind for p in PERTURBATIONS)


class QueryCorruptor:
    """Damages a query until it carries the degree's added spans over its own
    parent count, spending the lane's most-headroom detector first."""

    def __init__(
        self,
        baseline: LaneBaseline,
        extractor: FeatureExtractor | None = None,
        seed: int = 0,
    ) -> None:
        self._baseline = baseline
        self._extractor = extractor or FeatureExtractor(
            engines=(Engine.REGEX, Engine.WORDFREQ)
        )
        self._seed = seed
        self._moves = {p.kind: p for p in PERTURBATIONS}

    def spans(self, text: str) -> int:
        found = self._extractor.resolve(text, groups=[FeatureGroup.CORRUPTION])
        return sum(
            len(spans)
            for spans in found.spans.get(FeatureGroup.CORRUPTION, {}).values()
        )

    def corrupt(
        self,
        text: str,
        lane: str,
        degree: CorruptionDegree,
        query_id: str = "",
    ) -> str:
        target = self.spans(text) + degree.added_spans
        rng = Random(f"{self._seed}:{query_id}:{degree.value}")
        order = self._baseline.by_headroom(lane)
        used: set[CorruptionKind] = set()
        exhausted: set[CorruptionKind] = set()
        while self.spans(text) < target:
            move = next(
                (
                    self._moves[kind]
                    for kind in order
                    if kind not in exhausted
                    and (kind not in used or self._moves[kind].repeatable)
                ),
                None,
            )
            if move is None:
                return text
            damaged = move.apply(text, rng)
            if damaged == text:
                exhausted.add(move.kind)
                continue
            text = damaged
            used.add(move.kind)
        return text


class CorruptOperator(Operator):
    """Deterministic junking, no model in the loop (stdlib + wordfreq): a
    typo does not move which document answers, so parent qrels inherit."""

    declaration: ClassVar[Declaration] = Declaration(
        operator="corrupt",
        floors="corruption:light | corruption:heavy",
        surface_origin=SurfaceOrigin.NONE,
        answer_key=AnswerKeyPath.INHERIT,
        meaning_preserved=True,
        verifiable_by=(
            "corruption spans on local re-measure exceed the parent's own "
            "count by the degree's delta, spent on the detector with the most "
            "headroom in the parent's lane (corruption census)"
        ),
        # heavy damage poisons the spaCy tagger, so a corrupted row can leave
        # the class it was generated for — the audit rules on that drift
        credit_gate=CreditGate.DECLARATION_AUDIT,
        tool_loop=False,
    )

    def __init__(self, config: AugmentationConfig | None = None) -> None:
        config = config or AugmentationConfig()
        super().__init__(config)
        self._baseline = LaneBaseline(config.paths.corruption_census)
        self._corruptor = QueryCorruptor(self._baseline, seed=config.seed)
        self._order_seed = config.seed

    @staticmethod
    def degree(floor: str) -> CorruptionDegree | None:
        value = floor.removeprefix("corruption:")
        return next((d for d in CorruptionDegree if d.value == value), None)

    def mints(self, band: AxisBand) -> bool:
        return band.demands_presence and band.member in set(PERTURBED_KINDS)

    def serves(self, floor: str) -> bool:
        return floor.startswith("corruption:") and self.degree(floor) is not None

    def eligible(self, selection: pd.DataFrame, floor: str) -> pd.DataFrame:
        """Checkable parents the demand does not already cover, in seeded
        order — no bank run, the damage decides its own eligibility."""
        pool = self.unsatisfied(self.parent_pool(selection), floor)
        return pool[pool["checkable"]].sample(frac=1.0, random_state=self._order_seed)

    def apply(
        self,
        parent: pd.Series,
        floor: str,
        text: str,
        requirement: tuple = (),
    ) -> str | None:
        degree = self.degree(floor)
        if degree is None:
            return None
        return self._corruptor.corrupt(
            text, str(parent["dataset"]), degree, str(parent["query_id"])
        )

    def instruction(
        self, floor: str, parent: pd.Series, requirement: tuple = ()
    ) -> str:
        raise NotImplementedError(
            "corrupt is deterministic (`apply`): a model asked for a "
            "QWERTY-adjacent typo normalises it away and reproduces nothing"
        )

    def targets(
        self, floor: str, parent: pd.Series, requirement: tuple = ()
    ) -> Targets:
        degree = self.degree(floor)
        headroom = self._baseline.by_headroom(str(parent["dataset"]))
        return Targets(
            spans=tuple(
                SpanTarget(feature=kind.value, min_count=1)
                for kind in headroom[: max(1, degree.added_spans)]
            )
        )

    def structural(
        self, parent: pd.Series, text: str, targets: Targets
    ) -> list[str]:
        """An unchanged query is a silent no-op the span targets cannot
        catch, because the parent's own spans already satisfy them."""
        if text == str(parent["query"]):
            return ["corruption was a no-op: text unchanged"]
        return []
