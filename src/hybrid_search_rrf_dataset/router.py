"""Strategy Router baseline — predict dense/sparse/rrf per query (SPEC d45/d46).

Track A of the router-baseline plan. Two one-vs-rest logistic binaries are
trained on the *decisive* rows of the golden set (where one route clearly won);
a query routes to `pure_rrf` when neither binary fires — rrf as the hedge, never
a trained class (145 decisive rows, a rank-fusion artifact). The module runs as
a three-representation ablation — engineered taxonomy features, a frozen
multilingual-e5-small embedding, or both — to measure which representation
carries routing signal.

Query-only by construction: `predict` takes a `collection_stats` argument that
v1 ignores, so the d44b corpus features graft in without an API break. The
encoder is deliberately NOT bge-small-en (the retriever that produced the
labels) — reusing it would let the model learn a shortcut through bge's own
quirks (SPEC d46c).
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from enum import StrEnum
from query_taxonomy.features import FeatureExtractor as _FE
from pathlib import Path
from sentence_transformers import SentenceTransformer
from typing import TYPE_CHECKING, ClassVar

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from tqdm.auto import tqdm
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from hybrid_search_rrf_dataset.fusion import StrategyName
from hybrid_search_rrf_dataset.golden import LLMScoreClient
from hybrid_search_rrf_dataset.objective import RouterObjective

if TYPE_CHECKING:
    from query_taxonomy.features import FeatureExtractor

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
LABELS_PATH = DATA_DIR / "route_labels" / "labels.parquet"
CATALOG_PATH = DATA_DIR / "feature_table" / "catalog.parquet"
EMBED_CACHE_PATH = DATA_DIR / "route_labels" / "e5_embeddings.parquet"

IDENTITY_COLS = ("dataset", "query_id", "checkable")
SCORE_COLS = {
    StrategyName.DENSE_ONLY: "score_dense_only",
    StrategyName.PURE_RRF: "score_pure_rrf",
    StrategyName.SPARSE_ONLY: "score_sparse_only",
}

# F2 derived columns (SPEC d47a): computed in FeatureSpace from the base
# catalog columns so the LR sees identifier density and word-length shape
# as direct features instead of learning them through the length pair.
_STRUCTURED_ID_PREFIX = "structured_identifiers."
_LENGTH_WORDS = "length.length_words"
_LENGTH_CHARS = "length.length_chars"
_DROPPED_ENGINEERED = frozenset({_LENGTH_WORDS})
_DERIVED_ENGINEERED = (
    "derived.identifier_density",
    "derived.avg_word_length",
    "derived.short_id_query",
)
_ROUTE_ORDER = list(StrategyName)  # column/index order for the score matrix
DENSE_IDX = _ROUTE_ORDER.index(StrategyName.DENSE_ONLY)
RRF_IDX = _ROUTE_ORDER.index(StrategyName.PURE_RRF)
SPARSE_IDX = _ROUTE_ORDER.index(StrategyName.SPARSE_ONLY)

ENCODER_MODEL = "intfloat/multilingual-e5-small"

# Abstention thresholds (SPEC d46e): a route probability below its tuned cutoff
# does not fire, and when neither fires the query routes to rrf. Tuning grid-
# searches cutoffs over [MIN, MAX]; STEPS=17 gives ~0.05 resolution.
DEFAULT_THRESHOLD = 0.5
THRESHOLD_GRID_MIN = 0.1
THRESHOLD_GRID_MAX = 0.9
THRESHOLD_GRID_STEPS = 17


class Representation(StrEnum):
    """Which feature block(s) feed the model — the ablation axis (SPEC d46b)."""

    ENGINEERED = "engineered"
    EMBEDDING = "embedding"
    BOTH = "both"


class QueryEncoder:
    """Frozen multilingual-e5-small query embeddings, disk-cached by query text.

    e5 expects the ``query:`` prefix on search queries; omitting it measurably
    degrades the representation. The model (~470MB) downloads on first use — a
    user-initiated step, never triggered at import.
    """

    def __init__(
        self, model_name: str = ENCODER_MODEL, cache_path: Path = EMBED_CACHE_PATH
    ) -> None:
        self.model_name = model_name
        self._cache_path = cache_path
        self._model = None
        self._cache: dict[str, np.ndarray] = {}
        if cache_path.exists():
            frame = pd.read_parquet(cache_path)
            dims = [c for c in frame.columns if c != "query"]
            self._cache = {
                row["query"]: row[dims].to_numpy(dtype=np.float32)
                for _, row in frame.iterrows()
            }

    def _load(self):
        if self._model is None:
            logger.info("loading encoder %s", self.model_name)
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def embed(self, queries: Sequence[str]) -> np.ndarray:
        missing = [q for q in dict.fromkeys(queries) if q not in self._cache]
        if missing:
            model = self._load()
            vecs = model.encode(
                [f"query: {q}" for q in missing],
                normalize_embeddings=True,
                show_progress_bar=len(missing) > 512,
            )
            for q, v in zip(missing, vecs, strict=True):
                self._cache[q] = np.asarray(v, dtype=np.float32)
            self._flush()
        return np.vstack([self._cache[q] for q in queries])

    def _flush(self) -> None:
        dim = len(next(iter(self._cache.values())))
        rows = (
            {"query": q, **{f"e{i}": v[i] for i in range(dim)}}
            for q, v in self._cache.items()
        )
        self._cache_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_parquet(self._cache_path, index=False)


class FeatureSpace:
    """Turns a labelled frame into a model matrix for one `Representation`.

    Stateful so PCA and the engineered-column set are fit on train only and
    reused on validation/test. The embedding block is PCA-reduced so
    it does not swamp the 57 engineered features on ~2.5K rows.
    """

    def __init__(
        self,
        representation: Representation,
        encoder: QueryEncoder | None = None,
        pca_dims: int = 50,
    ) -> None:
        self.representation = representation
        self._encoder = encoder
        self._pca_dims = pca_dims
        self._engineered_cols: list[str] = []
        self._pca: PCA | None = None

    @property
    def needs_engineered(self) -> bool:
        return self.representation != Representation.EMBEDDING

    @property
    def needs_embedding(self) -> bool:
        return self.representation != Representation.ENGINEERED

    def fit(self, train: pd.DataFrame) -> FeatureSpace:
        if self.needs_engineered:
            base = [
                c
                for c in train.columns
                if _is_engineered(c) and c not in _DROPPED_ENGINEERED
            ]
            self._engineered_cols = base + list(_DERIVED_ENGINEERED)
        if self.needs_embedding:
            if self._encoder is None:
                raise ValueError(
                    f"{self.representation} needs an encoder; none was passed."
                )
            raw = self._encoder.embed(train["query"].tolist())
            self._pca = PCA(n_components=min(self._pca_dims, raw.shape[1]))
            self._pca.fit(raw)
        return self

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        blocks: list[np.ndarray] = []
        if self.needs_engineered:
            # reindex (not frame[cols]) so the serving path — a one-row frame
            # carrying only the features that fired — fills absent columns with
            # 0 rather than raising KeyError. Derived columns (SPEC d47a) are
            # computed BEFORE reindex so they land in the output block.
            enriched = _derive_engineered(frame)
            engineered = enriched.reindex(
                columns=self._engineered_cols, fill_value=0.0
            )
            blocks.append(engineered.to_numpy(dtype=np.float64))
        if self.needs_embedding:
            assert self._pca is not None
            embedded = self._encoder.embed(frame["query"].tolist())
            blocks.append(self._pca.transform(embedded))
        return np.hstack(blocks)

    @property
    def feature_names(self) -> list[str]:
        """Column labels aligned with `transform`'s output — engineered feature
        names, then `pca_<i>` for the embedding block."""
        names = list(self._engineered_cols) if self.needs_engineered else []
        if self.needs_embedding:
            assert self._pca is not None
            names += [f"pca_{i}" for i in range(self._pca.n_components_)]
        return names


class StrategyRouter:
    """Two logistic binaries + the hedge rule (SPEC d46d).

    `predict` is the serving surface: it extracts the taxonomy features inline,
    so callers pass a raw query string. `collection_stats` is accepted and
    ignored — reserved for the d44b corpus features.
    """

    def __init__(
        self,
        representation: Representation = Representation.BOTH,
        encoder: QueryEncoder | None = None,
        pca_dims: int = 50,
        C: float = 1.0,
        extractor: FeatureExtractor | None = None,
    ) -> None:
        self.representation = representation
        self._space = FeatureSpace(representation, encoder, pca_dims)
        self._dense = _binary_pipeline(C)
        self._sparse = _binary_pipeline(C)
        self._t_dense = DEFAULT_THRESHOLD
        self._t_sparse = DEFAULT_THRESHOLD
        self._extractor = extractor
        self._decisive_margin = RouterObjective().decisive_margin

    @property
    def decisive_margin(self) -> float:
        return self._decisive_margin

    @property
    def thresholds(self) -> tuple[float, float]:
        return self._t_dense, self._t_sparse

    def fit(
        self,
        train: pd.DataFrame,
        *,
        all_rows: bool = False,
        max_class_share: float | None = None,
    ) -> StrategyRouter:
        """Fit both binaries. Default (SPEC d45a) uses decisive rows only —
        clean labels the LR can trust. `all_rows=True` expands to every
        routes_differ row (thin-margin included; still a real winner);
        `max_class_share` seeded-downsamples majority classes so no class
        exceeds that share of the training set. Both are SPEC d47a F3
        experimental flags — defaults preserve d45a."""
        if all_rows:
            rows = train[_routes_differ(train)]
        else:
            rows = train[_margin(train) >= self._decisive_margin]
        winner = _winner(rows)
        if max_class_share is not None:
            rows, winner = _cap_class_share(rows, winner, max_class_share)
        self._space.fit(train)
        x = self._space.transform(rows)
        self._dense.fit(x, (winner == StrategyName.DENSE_ONLY).to_numpy())
        self._sparse.fit(x, (winner == StrategyName.SPARSE_ONLY).to_numpy())
        return self

    def tune_thresholds(self, tune: pd.DataFrame) -> StrategyRouter:
        """Pick (t_dense, t_sparse) that maximise the router's mean objective
        on the routes_differ rows of an all-shapes frame — where the threshold
        actually changes the outcome. Amends SPEC d45(b)/d46(e) via d47(a)
        F1': measured on 2026-08-03, tuning on the full frame diluted the
        objective (67% of tune was all_tied/all_zero, threshold-invariant),
        landing the argmax at `t_sparse=0.9` (sparse never fires) while the
        empirical optimum on routes_differ was `t_sparse=0.6` (+0.05 on the
        eval-decisive mean). All_tied and all_zero rows have no route
        distinction to make, so excluding them aligns the tuner objective
        with the eval slice."""
        tune = tune[_routes_differ(tune)]
        grid = np.linspace(
            THRESHOLD_GRID_MIN, THRESHOLD_GRID_MAX, THRESHOLD_GRID_STEPS
        )
        p_dense, p_sparse = self._probabilities(tune)
        scores = _score_matrix(tune)
        rows = np.arange(len(tune))
        best, best_score = (DEFAULT_THRESHOLD, DEFAULT_THRESHOLD), -np.inf
        for t_d in grid:
            for t_s in grid:
                idx = _route_indices(p_dense, p_sparse, t_d, t_s)
                mean = float(scores[rows, idx].mean())
                if mean > best_score:
                    best, best_score = (t_d, t_s), mean
        self._t_dense, self._t_sparse = float(best[0]), float(best[1])
        return self

    def predict_routes(self, frame: pd.DataFrame) -> list[StrategyName]:
        p_dense, p_sparse = self._probabilities(frame)
        return _route_from_probs(p_dense, p_sparse, self._t_dense, self._t_sparse)

    def coefficients(self) -> pd.DataFrame:
        """Per-feature logistic weights for each binary — the instrument readout
        (SPEC d46c): which features push a query toward dense vs sparse. Weights
        are on standardised features, so magnitudes are comparable."""
        names = self._space.feature_names
        return pd.DataFrame(
            {
                "feature": names,
                "dense_weight": self._dense.named_steps["lr"].coef_[0],
                "sparse_weight": self._sparse.named_steps["lr"].coef_[0],
            }
        )

    def predict(
        self, query: str, *, collection_stats: dict[str, float] | None = None
    ) -> StrategyName:
        """Serving path: extract features from the raw query and route it.

        `collection_stats` is reserved for the d44b corpus features and ignored
        in v1 (the model is query-only).
        """
        del collection_stats
        return self.predict_routes(self._query_frame(query))[0]

    def explain(
        self, query: str, *, collection_stats: dict[str, float] | None = None
    ) -> dict[str, object]:
        """`predict` plus the numbers behind it: the two win-probabilities, the
        thresholds each must clear to fire, and the resulting route. For
        inspecting why a query routed the way it did."""
        del collection_stats
        p_dense, p_sparse = self._probabilities(self._query_frame(query))
        p_dense, p_sparse = float(p_dense[0]), float(p_sparse[0])
        route = _route_from_probs(
            np.array([p_dense]), np.array([p_sparse]), self._t_dense, self._t_sparse
        )[0]
        return {
            "query": query,
            "route": route,
            "p_dense": p_dense,
            "p_sparse": p_sparse,
            "t_dense": self._t_dense,
            "t_sparse": self._t_sparse,
            "dense_fires": p_dense >= self._t_dense,
            "sparse_fires": p_sparse >= self._t_sparse,
        }

    def _query_frame(self, query: str) -> pd.DataFrame:
        """A one-row frame for the serving path: the query text plus its
        inline-extracted engineered features when the representation needs them."""
        row: dict[str, object] = {"query": query}
        if self._space.needs_engineered:
            row.update(_extract_features(self._extractor, query))
        return pd.DataFrame([row])

    def _probabilities(self, frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        x = self._space.transform(frame)
        return self._dense.predict_proba(x)[:, 1], self._sparse.predict_proba(x)[:, 1]


def _binary_pipeline(C: float) -> Pipeline:
    return Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "lr",
                LogisticRegression(
                    class_weight="balanced", C=C, max_iter=1000
                ),
            ),
        ]
    )


def _score_matrix(frame: pd.DataFrame) -> np.ndarray:
    """The three per-route objective scores as an (n, 3) array, columns ordered
    by `_ROUTE_ORDER`."""
    return frame[[SCORE_COLS[r] for r in _ROUTE_ORDER]].to_numpy(dtype=np.float64)


def _route_indices(
    p_dense: np.ndarray, p_sparse: np.ndarray, t_dense: float, t_sparse: float
) -> np.ndarray:
    """The hedge rule (SPEC d46d), vectorised to indices into `_ROUTE_ORDER`:
    both below threshold ⇒ rrf, both fire ⇒ higher probability, else the route
    that fired."""
    fires_dense = p_dense >= t_dense
    fires_sparse = p_sparse >= t_sparse
    idx = np.full(p_dense.shape, RRF_IDX, dtype=int)
    idx[fires_dense & ~fires_sparse] = DENSE_IDX
    idx[fires_sparse & ~fires_dense] = SPARSE_IDX
    both = fires_dense & fires_sparse
    idx[both] = np.where(p_dense[both] >= p_sparse[both], DENSE_IDX, SPARSE_IDX)
    return idx


def _route_from_probs(
    p_dense: np.ndarray, p_sparse: np.ndarray, t_dense: float, t_sparse: float
) -> list[StrategyName]:
    idx = _route_indices(p_dense, p_sparse, t_dense, t_sparse)
    return [_ROUTE_ORDER[i] for i in idx]


def _is_engineered(col: str) -> bool:
    return col not in IDENTITY_COLS and "." in col


def _derive_engineered(frame: pd.DataFrame) -> pd.DataFrame:
    """Append the F2 derived columns (SPEC d47a): identifier_density,
    avg_word_length, short_id_query. Missing base columns default to 0 so
    the serving path (a one-row frame carrying only fired features) works
    without KeyError — same tolerance FeatureSpace.transform already uses
    for the catalog block."""
    words = frame.get(_LENGTH_WORDS, pd.Series(0.0, index=frame.index))
    chars = frame.get(_LENGTH_CHARS, pd.Series(0.0, index=frame.index))
    id_cols = [c for c in frame.columns if c.startswith(_STRUCTURED_ID_PREFIX)]
    id_sum = (
        frame[id_cols].sum(axis=1)
        if id_cols
        else pd.Series(0.0, index=frame.index)
    )
    safe_words = words.clip(lower=1)
    identifier_density = id_sum / safe_words
    avg_word_length = chars / safe_words
    short_id_query = ((identifier_density > 0) & (words <= 5)).astype(float)
    return frame.assign(
        **{
            "derived.identifier_density": identifier_density,
            "derived.avg_word_length": avg_word_length,
            "derived.short_id_query": short_id_query,
        }
    )


def _margin(frame: pd.DataFrame) -> pd.Series:
    ordered = np.sort(_score_matrix(frame), axis=1)
    return pd.Series(ordered[:, -1] - ordered[:, -2], index=frame.index)


def _routes_differ(frame: pd.DataFrame) -> pd.Series:
    """Mask for rows where the top route strictly beats the runner-up
    (margin > 1e-9 tolerance per SPEC d37i / d41e). Excludes all_tied and
    all_zero — the outcome-shapes without a clear winner."""
    return _margin(frame) > 1e-9


def _winner(frame: pd.DataFrame) -> pd.Series:
    idx = _score_matrix(frame).argmax(axis=1)
    return pd.Series([_ROUTE_ORDER[i] for i in idx], index=frame.index)


def _cap_class_share(
    rows: pd.DataFrame,
    winners: pd.Series,
    max_share: float,
    seed: int = 0,
) -> tuple[pd.DataFrame, pd.Series]:
    """Downsample majority classes so no class exceeds `max_share` of the
    training set (SPEC d47a F3 experimental knob). Seeded random selection;
    minority classes are always preserved. Iterates because capping one
    class shifts the others' shares."""
    if not 0 < max_share < 1:
        raise ValueError(f"max_class_share must be in (0, 1), got {max_share}")
    rng = np.random.default_rng(seed)
    keep = pd.Series(True, index=rows.index)
    while True:
        active_winners = winners[keep]
        counts = active_winners.value_counts()
        total = int(keep.sum())
        over_by = {c: n - max_share * total for c, n in counts.items() if n / total > max_share}
        if not over_by:
            break
        worst_class = max(over_by, key=over_by.get)
        class_idx = active_winners.index[active_winners == worst_class].to_numpy()
        others = total - counts[worst_class]
        target = int(np.floor(max_share * others / (1 - max_share)))
        drop_n = int(counts[worst_class]) - target
        drop = rng.choice(class_idx, size=drop_n, replace=False)
        keep.loc[drop] = False
    return rows[keep], winners[keep]


def decisive_rows(data: pd.DataFrame) -> pd.DataFrame:
    """Decisive rows (SPEC d41d) with a `winner` column — the router's training
    and evaluation substrate. Decisive = the top route's score beats the
    runner-up by at least the objective's decisive margin."""
    keep = _margin(data) >= RouterObjective().decisive_margin
    decisive = data[keep].copy()
    decisive["winner"] = _winner(decisive)
    return decisive


def _mean_objective(frame: pd.DataFrame, routes: Sequence[StrategyName]) -> float:
    scores = _score_matrix(frame)
    idx = np.fromiter(
        (_ROUTE_ORDER.index(r) for r in routes), dtype=int, count=len(routes)
    )
    return float(scores[np.arange(len(idx)), idx].mean())


def _extract_features(
    extractor: FeatureExtractor | None, query: str
) -> dict[str, float]:
    """Reproduce the catalog columns for one query (the convention in
    scripts/feature_table.py: span counts `<group>.<type>`, stats
    `<bank>.<stat>`). Builds the extractor lazily — it pulls in spaCy."""
    if extractor is None:
        extractor = _FE(engines=None)
    features = extractor.resolve(query)
    row: dict[str, float] = {}
    for group, counts in features.tfs.items():
        for type_, count in counts.items():
            row[f"{group.value}.{type_}"] = count
    for banks in features.stats.values():
        for bank_name, stats in banks.items():
            for stat in stats:
                row[f"{bank_name}.{stat.name}"] = stat.value
    return row


# Production classifier route bands (SPEC production_router memory / golden.py
# HybridRoutingBuilder): 0-2 dense, 3-6 rrf, 7-9 sparse.
def _production_route(score: int) -> StrategyName:
    if score <= 2:
        return StrategyName.DENSE_ONLY
    if score <= 6:
        return StrategyName.PURE_RRF
    return StrategyName.SPARSE_ONLY


class AutoFusionRouter:
    """The auto-fusion HTTP classifier as a router: `LLMScoreClient.score`
    per query, mapped to `StrategyName` via the production hard bands
    (`_production_route`). Skips retrieval entirely — evaluation uses the
    per-route scores already stored on `labels.parquet`. Deterministic per
    query text, so `(dataset, query_id) -> score` is cached to disk and
    re-runs are free."""

    CACHE_PATH: ClassVar[Path] = (
        DATA_DIR / "route_labels" / "autofusion_cache.parquet"
    )

    def __init__(
        self,
        client: LLMScoreClient | None = None,
        cache_path: Path | None = None,
    ) -> None:
        self._client = client or LLMScoreClient()
        self._cache_path = cache_path or self.CACHE_PATH
        self._cache: dict[tuple[str, str], int] = self._load_cache()

    def _load_cache(self) -> dict[tuple[str, str], int]:
        if not self._cache_path.exists():
            return {}
        df = pd.read_parquet(self._cache_path)
        return {
            (str(r.dataset), str(r.query_id)): int(r.score)
            for r in df.itertuples(index=False)
        }

    def _save_cache(self) -> None:
        if not self._cache:
            return
        df = pd.DataFrame(
            [
                {"dataset": ds, "query_id": qid, "score": score}
                for (ds, qid), score in sorted(self._cache.items())
            ]
        )
        self._cache_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(self._cache_path, index=False)

    def route_batch(
        self, frame: pd.DataFrame, desc: str = "auto-fusion classify"
    ) -> list[StrategyName]:
        """Route every row of `frame` (needs `dataset`, `query_id`, `query`).
        Cache hits skip the network; only misses trigger a POST. Saves cache
        once at end iff any miss fired."""
        routes: list[StrategyName] = []
        misses = 0
        for _, row in tqdm(
            frame.iterrows(), total=len(frame), desc=desc, leave=False
        ):
            key = (str(row["dataset"]), str(row["query_id"]))
            if key not in self._cache:
                self._cache[key] = self._client.score(str(row["query"]))
                misses += 1
            routes.append(_production_route(self._cache[key]))
        if misses:
            self._save_cache()
        return routes


def _phase(bar: tqdm | None, label: str) -> None:
    """Show the current sub-step of a round on the progress bar's postfix."""
    if bar is not None:
        bar.set_postfix_str(label)


class RouterExperiment:
    """Runs the ablation: three representations × two validation protocols, each
    reporting the six-column headroom table over held-out decisive rows.

    Both protocols answer different deployment questions (SPEC d46f): the random
    within-lane split is "new queries on a known collection", the held-out lane
    is "a collection never seen". The gap between them is corpus-dependence.
    """

    HOLDOUT_LANE = "rarb-math"

    def __init__(
        self,
        labels_path: Path = LABELS_PATH,
        catalog_path: Path = CATALOG_PATH,
        encoder: QueryEncoder | None = None,
        seed: int = 0,
    ) -> None:
        self._labels_path = labels_path
        self._catalog_path = catalog_path
        self._encoder = encoder
        self._seed = seed
        self._data: pd.DataFrame | None = None
        self._prod_cache: dict[str, StrategyName] = {}

    def load(self) -> pd.DataFrame:
        if self._data is None:
            labels = pd.read_parquet(self._labels_path)
            names = pq.read_schema(self._catalog_path).names
            feat_cols = [c for c in names if _is_engineered(c)]
            catalog = pd.read_parquet(
                self._catalog_path, columns=["dataset", "query_id", *feat_cols]
            )
            merged = labels.merge(catalog, on=["dataset", "query_id"], how="inner")
            self._data = merged.reset_index(drop=True)
        return self._data

    def run(
        self,
        representations: Sequence[Representation] | None = None,
        production_client: object | None = None,
        *,
        all_rows: bool = False,
        max_class_share: float | None = None,
        autofusion: bool | AutoFusionRouter = False,
        autofusion_sample: int | float | None = None,
    ) -> pd.DataFrame:
        representations = list(representations or Representation)
        if self._encoder is None:
            for rep in representations:
                if rep is not Representation.ENGINEERED:
                    logger.warning("skipping %s: no encoder passed", rep)
            representations = [
                r for r in representations if r is Representation.ENGINEERED
            ]
        data = self.load()
        splits = {
            "random_within_lane": self._split_random(data),
            "holdout_lane": self._split_holdout(data),
        }
        rounds = [(protocol, rep) for protocol in splits for rep in representations]
        rows = []
        bar = tqdm(rounds, desc="router ablation")
        for protocol, rep in bar:
            bar.set_description(f"{protocol}·{rep}")
            train, test = splits[protocol]
            result = self._run_one(
                train, test, rep, production_client, bar,
                all_rows=all_rows, max_class_share=max_class_share,
            )
            rows.append(
                {
                    "protocol": protocol,
                    "representation": str(rep),
                    "all_rows": all_rows,
                    "max_class_share": max_class_share,
                }
                | result
            )
            bar.set_postfix(
                headroom=f"{result['headroom_captured']:.3f}",
                n=result["n_test_decisive"],
            )
        if autofusion:
            af_router = (
                autofusion if isinstance(autofusion, AutoFusionRouter)
                else AutoFusionRouter()
            )
            for protocol, (_, test) in splits.items():
                rows.append(
                    {
                        "protocol": protocol,
                        "representation": "auto_fusion",
                        "all_rows": None,
                        "max_class_share": None,
                    }
                    | self._run_autofusion(
                        af_router, test, protocol, sample=autofusion_sample
                    )
                )
        return pd.DataFrame(rows)

    def _run_autofusion(
        self,
        af_router: AutoFusionRouter,
        test: pd.DataFrame,
        protocol: str,
        *,
        sample: int | float | None = None,
    ) -> dict[str, object]:
        """One auto-fusion row per protocol: no fit, no tune — classify each
        held-out decisive query, look up the chosen route's stored score, and
        report the same six-column table (SPEC d47 auto-fusion baseline).
        `sample` truncates the decisive frame for cheap smoke runs: int =
        absolute count, float in (0, 1] = fraction of the decisive slice."""
        decisive_margin = RouterObjective().decisive_margin
        decisive = test[_margin(test) >= decisive_margin]
        if sample is not None and len(decisive) > 0:
            if isinstance(sample, float) and 0 < sample < 1:
                n = max(1, int(len(decisive) * sample))
            else:
                n = min(int(sample), len(decisive))
            decisive = decisive.sample(n=n, random_state=self._seed)
        routes = af_router.route_batch(
            decisive, desc=f"{protocol}·auto_fusion classify"
        )
        return _six_column(decisive, routes) | {
            "n_test_decisive": len(decisive),
            "t_dense": None,
            "t_sparse": None,
        }

    def split(self, protocol: str) -> tuple[pd.DataFrame, pd.DataFrame]:
        """The (train, test) frames for one protocol — exposed so a notebook can
        inspect the training and held-out stages directly."""
        data = self.load()
        if protocol == "random_within_lane":
            return self._split_random(data)
        if protocol == "holdout_lane":
            return self._split_holdout(data)
        raise ValueError(
            f"unknown protocol {protocol!r}; use 'random_within_lane' or "
            "'holdout_lane'."
        )

    def _run_one(
        self,
        train: pd.DataFrame,
        test: pd.DataFrame,
        representation: Representation,
        production_client: object | None,
        bar: tqdm | None = None,
        *,
        all_rows: bool = False,
        max_class_share: float | None = None,
    ) -> dict[str, object]:
        _phase(bar, "fitting")
        train_model, train_tune = self._prep_train(train)
        router = StrategyRouter(representation, self._encoder).fit(
            train_model, all_rows=all_rows, max_class_share=max_class_share,
        )
        _phase(bar, "tuning thresholds")
        router.tune_thresholds(train_tune)
        _phase(bar, "evaluating")
        decisive = test[_margin(test) >= router.decisive_margin]
        router_routes = router.predict_routes(decisive)
        prod_routes = self._production_routes(production_client, decisive)
        t_dense, t_sparse = router.thresholds
        return _six_column(decisive, router_routes, prod_routes) | {
            "n_test_decisive": len(decisive),
            "t_dense": t_dense,
            "t_sparse": t_sparse,
        }

    def _prep_train(
        self, train: pd.DataFrame, tune_frac: float = 0.25
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Carve an all-shapes slice of train for threshold tuning; the rest
        feeds the model (which filters to decisive rows itself)."""
        tune = train.groupby("shape", group_keys=False).sample(
            frac=tune_frac, random_state=self._seed
        )
        model = train.drop(index=tune.index)
        return model, tune

    def _split_random(
        self, data: pd.DataFrame, test_frac: float = 0.2
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        # NOTE: near-duplicate-aware splitting (SPEC d46f) is deferred — the
        # ~5.5% cos>0.95 pairs are not yet an artifact on disk, so a duplicate
        # can straddle train/test here. Flagged for feature-review.
        test = data.groupby("dataset", group_keys=False).sample(
            frac=test_frac, random_state=self._seed
        )
        return data.drop(index=test.index), test

    def _split_holdout(
        self, data: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        test = data[data["dataset"] == self.HOLDOUT_LANE]
        return data[data["dataset"] != self.HOLDOUT_LANE], test

    def _production_routes(
        self, client: object | None, decisive: pd.DataFrame
    ) -> list[StrategyName] | None:
        if client is None:
            return None
        routes = []
        for query in tqdm(decisive["query"], desc="production classify", leave=False):
            if query not in self._prod_cache:
                self._prod_cache[query] = _production_route(client.score(query))
            routes.append(self._prod_cache[query])
        return routes


def _six_column(
    decisive: pd.DataFrame,
    router_routes: Sequence[StrategyName],
    production_routes: Sequence[StrategyName] | None = None,
) -> dict[str, float]:
    """Mean objective per method over held-out decisive rows, plus the headline
    share of headroom captured (SPEC d46g). Constants and oracle come from the
    stored per-route scores; production is appended only when supplied."""
    constants = {
        f"const_{s.value}": float(decisive[SCORE_COLS[s]].mean())
        for s in StrategyName
    }
    oracle = float(decisive[[SCORE_COLS[s] for s in StrategyName]].max(axis=1).mean())
    router = _mean_objective(decisive, router_routes)
    best_const = max(constants.values())
    denom = oracle - best_const
    table: dict[str, float] = {
        **constants,
        "oracle": oracle,
        "router": router,
        "headroom_captured": (router - best_const) / denom if denom > 0 else float("nan"),
    }
    if production_routes is not None:
        table["production"] = _mean_objective(decisive, production_routes)
    return table
