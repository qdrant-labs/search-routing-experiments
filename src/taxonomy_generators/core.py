"""Generation twins of the detection banks (SPEC d34).

Generators are deliberately dumb and grounding-blind: they emit surfaces —
text snippets exhibiting exactly one taxonomy feature — and never touch
surrounding text. The default engine samples each bank's own pattern, so
lock-step with detection is structural; the round-trip test enforces it.
"""

import inspect
import re
from abc import ABC, abstractmethod
from random import Random

from edify import RegexBuilder
from rstr import Xeger

from query_taxonomy.core import AmbiguityTier, RegexBank
from query_taxonomy.taxonomy import FeatureGroup

_ZERO_WIDTH_ESCAPES = frozenset("bBAZ")


def strip_guards(pattern: str) -> str:
    """Remove zero-width guards — ``\\b``/``\\B``/``\\A``/``\\Z`` escapes and
    unescaped ``^``/``$`` — outside character classes. Guards constrain
    context, not content: a surface sampled from the stripped pattern still
    matches the original guarded pattern when it stands alone.
    """
    out: list[str] = []
    in_class = False
    i = 0
    while i < len(pattern):
        char = pattern[i]
        if char == "\\" and i + 1 < len(pattern):
            escaped = pattern[i + 1]
            if not in_class and escaped in _ZERO_WIDTH_ESCAPES:
                i += 2
                continue
            out.append(char)
            out.append(escaped)
            i += 2
            continue
        if char == "[" and not in_class:
            in_class = True
        elif char == "]" and in_class:
            in_class = False
        elif char in "^$" and not in_class:
            i += 1
            continue
        out.append(char)
        i += 1
    return "".join(out)


class SurfaceGenerator(ABC):
    """One feature's surface producer — the generation twin of a bank.

    Mirrors GeneralBank's identity triple (feature, group, tier) so
    registries and tools can present generators exactly like banks; the
    single output verb is `sample`.
    """

    @property
    @abstractmethod
    def feature(self) -> str:
        """Emitted feature name — must equal the twin bank's `str(name)`."""

    @property
    @abstractmethod
    def group(self) -> FeatureGroup:
        """Parent group from query-taxonomy.csv."""

    @property
    @abstractmethod
    def tier(self) -> AmbiguityTier:
        """The twin bank's ambiguity tier — surfaces of a `-like` feature
        are shape guesses, and consumers filter on this exactly as they do
        for detected spans."""

    @property
    @abstractmethod
    def description(self) -> str:
        """One-line, LLM-facing summary of what a surface looks like."""

    @abstractmethod
    def sample(self, rng: Random, n: int = 1) -> list[str]:
        """`n` surfaces from the injected seeded RNG."""


class _BoundedXeger(Xeger):
    """Xeger that repeats `+`/`*` pieces at most 8 times instead of 100.
    At 100, `math_expression` sampled 7,000-char formulas that its own
    detector took minutes to scan — surfaces must stay query-sized."""

    _REPEAT_CAP = 8

    def _handle_repeat(
        self, start_range: int, end_range: int, value: str
    ) -> str:
        # never cap below the pattern's own minimum: {10,12} stays >= 10
        bounded = min(end_range, max(start_range, self._REPEAT_CAP))
        return super()._handle_repeat(start_range, bounded, value)


class PatternGenerator(SurfaceGenerator):
    """Default generator: reverse-regex sampling over the twin bank's own
    guard-stripped pattern (rstr Xeger, arch-validator d34). Zero per-bank
    authoring; a bank pattern change flows in automatically.
    """

    _MAX_ATTEMPTS_PER_SURFACE = 25

    def __init__(self, bank_cls: type[RegexBank]) -> None:
        self._bank = bank_cls()
        source = self._bank.define(RegexBuilder()).to_regex()
        self._pattern = re.compile(strip_guards(source.pattern), source.flags)

    @property
    def feature(self) -> str:
        return f'{self._bank.group}:{self._bank.name}'
    
    @property
    def name(self) -> str:
        return self._bank.name

    @property
    def group(self) -> FeatureGroup:
        return self._bank.group

    @property
    def tier(self) -> AmbiguityTier:
        return self._bank.ambiguity

    @property
    def description(self) -> str:
        doc = inspect.getdoc(type(self._bank)) or ""
        return doc.split("\n\n")[0].replace("\n", " ").strip()

    def sample(self, rng: Random, n: int = 1) -> list[str]:
        xeger = _BoundedXeger(rng)
        surfaces: list[str] = []
        attempts = 0
        while len(surfaces) < n:
            if attempts >= self._MAX_ATTEMPTS_PER_SURFACE * n:
                raise RuntimeError(
                    f"{self.feature}: no bank-claimable surface after"
                    f" {attempts} attempts — the pattern needs a"
                    " hand-written override"
                )
            attempts += 1
            raw = xeger.xeger(self._pattern)
            spans = self._bank.compute(raw)
            if not spans:
                continue
            trimmed = spans[0].text
            if any(
                span.text == trimmed for span in self._bank.compute(trimmed)
            ):
                surfaces.append(trimmed)
        return surfaces
