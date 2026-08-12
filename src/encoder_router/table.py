"""The training table: acceptability labels joined with query text and cell,
plus the model's input encodings and supervision targets. Everything here is
query-side and offline; fold-local pieces (SVD fit, z-scoring, outcome rates)
expose fit/transform seams instead of baking a split in."""

from __future__ import annotations

import re
from functools import cached_property
from pathlib import Path

import numpy as np
import pandas as pd

from composition.cells import CELLS
from composition.compose import DEFAULT_CATALOG
from composition.floors import with_derived
from composition.mini_catalog import mini_catalog
from hybrid_search_rrf_dataset.fusion import SERVING_COST, StrategyName
from hybrid_search_rrf_dataset.labels import DEFAULT_OUT_DIR, AcceptabilityLabels

from encoder_router.targets import OUT_DIR

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


class TrainingTable:
    """Owner of the assembled frame and its aligned target matrices."""

    def __init__(
        self,
        *,
        tolerance: float | None = None,
        labels_path: Path = DEFAULT_OUT_DIR / "labels.parquet",
        catalog_path: Path = DEFAULT_CATALOG,
        out_dir: Path = OUT_DIR,
    ) -> None:
        self._tolerance = tolerance
        self._labels_path = labels_path
        self._catalog_path = catalog_path
        self._out_dir = out_dir

    @cached_property
    def frame(self) -> pd.DataFrame:
        """Labels carry query text and cell themselves — no selection join."""
        labels = pd.read_parquet(self._labels_path)
        merged = AcceptabilityLabels(labels, self._tolerance).frame()
        missing = merged["query"].isna() | (merged["query"].astype(str) == "")
        if missing.any():
            print(f"dropping {int(missing.sum())} labelled rows without text")
        return merged[~missing].reset_index(drop=True)

    @cached_property
    def catalog_rows(self) -> pd.DataFrame:
        """One catalog-shaped row per frame row: natural rows join the stored
        catalog, rows the extractor never saw are measured from text."""
        catalog = with_derived(pd.read_parquet(self._catalog_path))
        keyed = self.frame[KEY + ["query"]].merge(
            catalog.drop_duplicates(KEY), on=KEY, how="left"
        )
        numeric = [
            c for c in catalog.columns
            if c not in (*KEY, "checkable") and catalog[c].dtype != object
        ]
        missing = keyed[numeric].isna().all(axis=1)
        if missing.any():
            fresh = self._measure(keyed[missing])
            for column in numeric:
                if column in fresh.columns:
                    keyed.loc[missing, column] = fresh[column].to_numpy()
        return keyed[KEY + numeric].fillna(0.0)

    def _measure(self, rows: pd.DataFrame) -> pd.DataFrame:
        from query_taxonomy.features import FeatureExtractor

        print(f"extracting features for {len(rows)} unindexed queries")
        pool = rows[KEY + ["query"]].rename(columns={"dataset": "home_lane"})
        needed = tuple({band.column for cell in CELLS for band in cell.bands})
        return mini_catalog(pool, FeatureExtractor(engines=None), columns=needed)

    @cached_property
    def cell_targets(self) -> pd.DataFrame:
        """Multi-hot archetype membership: every cell's predicate evaluated
        on every row — the fill's single assignment is bookkeeping, not shape."""
        return pd.DataFrame(
            {cell.name: cell.select(self.catalog_rows) for cell in CELLS}
        ).astype(float)

    @cached_property
    def feature_matrix(self) -> pd.DataFrame:
        return self.catalog_rows.drop(columns=KEY)

    def route_targets(self) -> pd.DataFrame:
        """Head-route acceptability as float with NaN where unanswerable —
        the mask the route loss reads. RRF has no head (see HEAD_ROUTES)."""
        columns = {
            r: self.frame[f"ok_{r}"].astype("Float64") for r in HEAD_ROUTES
        }
        return pd.DataFrame(columns).astype(float)

    def corpus_targets(self) -> pd.DataFrame:
        """Lane block + gold-doc block, joined by key; outcome rates are
        fold-local — see `outcome_rates`."""
        lane = pd.read_parquet(self._out_dir / "corpus_profile.parquet")
        gold = pd.read_parquet(self._out_dir / "gold_doc_profile.parquet")
        out = self.frame[KEY].merge(lane, on="dataset", how="left")
        out = out.merge(gold, on=KEY, how="left")
        return out.drop(columns=KEY)

    def outcome_rates(self, train_mask: np.ndarray) -> pd.DataFrame:
        """Per-lane acceptability rates from TRAINING rows only, broadcast
        onto every row of the frame."""
        train = self.frame[train_mask]
        rates = train.groupby("dataset").agg(
            **{
                f"outcome.ok_{r}": (f"ok_{r}", lambda s: s.astype(float).mean())
                for r in ROUTES
            },
            **{
                "outcome.routes_differ": (
                    "shape", lambda s: float((s == "routes_differ").mean())
                )
            },
        )
        return (
            self.frame[["dataset"]]
            .merge(rates, on="dataset", how="left")
            .drop(columns="dataset")
        )


class QueryEmbeddings:
    """Frozen sentence embeddings, computed once per model and cached keyed
    on (dataset, query_id)."""

    def __init__(self, model: str, *, out_dir: Path = OUT_DIR) -> None:
        self.model = model
        self._prefix = EMBEDDING_PREFIXES.get(model, "")
        self._out_dir = out_dir

    @property
    def path(self) -> Path:
        slug = self.model.rsplit("/", 1)[-1].replace(".", "-")
        return self._out_dir / f"embeddings_{slug}.parquet"

    def matrix(self, frame: pd.DataFrame, batch_size: int = 256) -> np.ndarray:
        cache = self._filled(frame, batch_size).set_index(KEY)
        aligned = cache.loc[list(frame[KEY].itertuples(index=False))]
        return aligned.to_numpy(dtype=np.float32)

    def _filled(self, frame: pd.DataFrame, batch_size: int) -> pd.DataFrame:
        done = (
            pd.read_parquet(self.path)
            if self.path.exists()
            else pd.DataFrame(columns=KEY)
        )
        have = set(done[KEY].itertuples(index=False))
        todo = frame[
            ~frame[KEY].apply(tuple, axis=1).isin(have)
        ].drop_duplicates(KEY)
        if todo.empty:
            return done
        print(f"embedding {len(todo)} queries with {self.model}")
        vectors = self._encode(todo["query"].astype(str).tolist(), batch_size)
        fresh = pd.DataFrame(
            vectors, columns=[f"e{i}" for i in range(vectors.shape[1])]
        )
        fresh.insert(0, "query_id", todo["query_id"].to_numpy())
        fresh.insert(0, "dataset", todo["dataset"].to_numpy())
        merged = pd.concat([done, fresh], ignore_index=True)
        self._out_dir.mkdir(parents=True, exist_ok=True)
        merged.to_parquet(self.path, index=False)
        return merged

    def _encode(self, texts: list[str], batch_size: int) -> np.ndarray:
        from sentence_transformers import SentenceTransformer

        encoder = SentenceTransformer(self.model)
        return encoder.encode(
            [self._prefix + text for text in texts],
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=True,
        )


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
