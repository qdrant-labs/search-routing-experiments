"""Version-neutral vocabulary for marginal pre-label diversity strata."""

from __future__ import annotations

CORPUS_AXES: tuple[str, ...] = ("corpus_idf", "corpus_oov", "corpus_pmi")
"""Corpus-relative diversity axes, each measured marginally."""

UNKNOWN = "unknown"
"""The pre-label measurement did not cover the row; this is not a band."""

STRATA: dict[str, tuple[str, ...]] = {
    "corruption_degree": ("clean", "light", "heavy"),
    "corpus_idf": ("low_idf", "mid_idf", "high_idf"),
    "corpus_oov": ("in_vocab", "has_oov"),
    "corpus_pmi": ("co_occurring", "never_co_occurs", "unmeasured"),
}
"""Every band emitted by the pre-label catalog, including empty bands."""
