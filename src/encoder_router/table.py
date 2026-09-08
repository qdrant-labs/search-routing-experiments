"""The route vocabulary and the serve-time input encodings: threshold serving
over a probability matrix, the char-ngram SVD basis, and query-local term
rarity. Saved arms pickle `NgramSvd` by module path, so it stays here rather
than moving to a module named for serving."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

from hybrid_search_rrf_dataset.routes import SERVING_COST, StrategyName

MODELS_DIR = Path(__file__).resolve().parents[2] / "models"
"""Saved arms, outside the data root so serving pulls one 33 MB DVC output,
not the 7.7 GB dataset."""

ROUTES: tuple[str, ...] = tuple(
    sorted((s.value for s in StrategyName), key=lambda r: r)
)
PRIORITY: tuple[str, ...] = tuple(
    s.value for s in sorted(StrategyName, key=SERVING_COST.__getitem__)
)

EMBEDDING_PREFIXES = {
    "BAAI/bge-small-en-v1.5": (
        "Represent this sentence for searching relevant passages: "
    ),
    "intfloat/multilingual-e5-small": "query: ",
}

KEY = ["dataset", "query_id"]


HEDGE = "pure_rrf"
HEAD_ROUTES: tuple[str, ...] = tuple(r for r in PRIORITY if r != HEDGE)
"""The trained heads. RRF is never predicted, only served as the hedge:
its acceptability is ~the union of its parents' (87% identical), and on
every labelled row where both parents failed, fusion was acceptable —
a rule, not a classifier."""


HEDGE_BAR = 0.0
"""Inert by default: with the tuner's cost-adjusted reward pricing the
hedge, no-head-fired falls to rrf unconditionally and the tuner decides how
often that happens. Raise the bar only as a manual override — hand serving
rules lost to the tuner three times (constant-sparse, -rrf, -dense)."""


def serve_indices(
    ordered: np.ndarray,
    thresholds: float | np.ndarray,
    hedge_bar: float = HEDGE_BAR,
) -> np.ndarray:
    """Serve picks over a (rows, HEAD_ROUTES-ordered) probability matrix:
    most probable head among those clearing their thresholds — cost never
    breaks a both-fired case; nothing firing falls to the better parent
    unless every head is under `hedge_bar`, which is the rrf hedge
    (index len(HEAD_ROUTES))."""
    bar = np.broadcast_to(
        np.asarray(thresholds, dtype=float), (ordered.shape[1],)
    )
    fires = ordered >= bar
    fired = np.where(fires, ordered, -np.inf)
    doubt = ordered.max(axis=1) < hedge_bar
    fallback = np.where(doubt, ordered.shape[1], ordered.argmax(axis=1))
    return np.where(fires.any(axis=1), fired.argmax(axis=1), fallback)


def serve_from_probabilities(
    probs: pd.DataFrame,
    thresholds: float | np.ndarray,
    hedge_bar: float = HEDGE_BAR,
) -> list[str]:
    choices = [*HEAD_ROUTES, HEDGE]
    ordered = probs[list(HEAD_ROUTES)].to_numpy()
    return [choices[i] for i in serve_indices(ordered, thresholds, hedge_bar)]


ZIPF_RARE = 3.0
"""wordfreq's scale is log10 occurrences per billion words; 3.0 = once per
million words — the conventional rare-word line."""

_WORD = re.compile(r"[a-z0-9]+")


class ZipfStats:
    """Query-local term-rarity scalars from wordfreq's background-frequency
    table — serve-safe by construction: bundled data, no corpus, no
    pipeline."""

    COLUMNS = (
        "zipf.min", "zipf.mean", "zipf.max",
        "zipf.rare_share", "zipf.oov_share",
    )

    def frame(self, texts: pd.Series) -> pd.DataFrame:
        from wordfreq import zipf_frequency

        rows = np.zeros((len(texts), len(self.COLUMNS)), dtype=np.float32)
        for i, text in enumerate(texts.astype(str)):
            tokens = _WORD.findall(text.lower())
            if not tokens:
                continue
            freqs = np.array([zipf_frequency(t, "en") for t in tokens])
            rows[i] = (
                freqs.min(), freqs.mean(), freqs.max(),
                (freqs < ZIPF_RARE).mean(), (freqs == 0.0).mean(),
            )
        return pd.DataFrame(rows, columns=list(self.COLUMNS))


class LexicalShape:
    """Serve-safe character-shape scalars that tell an OOV *symbol* (code,
    identifier, error code) from a rare *word* — the split `zipf.min` alone
    conflates. Char-only: no corpus, no model."""

    COLUMNS = (
        "shape.digit_share", "shape.symbol_share", "shape.upper_share",
        "shape.max_token_len", "shape.symbolic_token_share",
    )

    def frame(self, texts: pd.Series) -> pd.DataFrame:
        rows = np.zeros((len(texts), len(self.COLUMNS)), dtype=np.float32)
        for i, text in enumerate(texts.astype(str)):
            if not text:
                continue
            tokens = text.split()
            alpha = [c for c in text if c.isalpha()]
            symbolic = [
                t for t in tokens
                if any(c.isdigit() or (not c.isalnum() and not c.isspace())
                       for c in t)
            ]
            rows[i] = (
                sum(c.isdigit() for c in text) / len(text),
                sum(not c.isalnum() and not c.isspace() for c in text) / len(text),
                (sum(c.isupper() for c in alpha) / len(alpha)) if alpha else 0.0,
                max((len(t) for t in tokens), default=0),
                (len(symbolic) / len(tokens)) if tokens else 0.0,
            )
        return pd.DataFrame(rows, columns=list(self.COLUMNS))


class NgramSvd:
    """The lexical channel: hashed character 3-5-grams reduced by truncated
    SVD, fit on training queries only."""

    def __init__(
        self, components: int = 128, hash_dim: int = 2**15, seed: int = 0
    ) -> None:
        from sklearn.decomposition import TruncatedSVD
        from sklearn.feature_extraction.text import HashingVectorizer

        self._hasher = HashingVectorizer(
            analyzer="char_wb",
            ngram_range=(3, 5),
            n_features=hash_dim,
            alternate_sign=False,
        )
        self._svd = TruncatedSVD(n_components=components, random_state=seed)

    def fit(self, texts: pd.Series) -> "NgramSvd":
        self._svd.fit(self._hasher.transform(texts.astype(str)))
        return self

    def transform(self, texts: pd.Series) -> np.ndarray:
        reduced = self._svd.transform(self._hasher.transform(texts.astype(str)))
        return reduced.astype(np.float32)

    def save(self, path: str | Path) -> Path:
        """The fitted SVD basis is serving state — a router checkpoint is
        incomplete without the exact basis its inputs were built on."""
        import joblib

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        return path

    @staticmethod
    def load(path: str | Path) -> "NgramSvd":
        import joblib

        return joblib.load(path)
