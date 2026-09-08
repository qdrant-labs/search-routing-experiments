"""Retrieval execution for run-scoped Rung A label partitions.

The orchestration contract lives in :mod:`rungs.label_run`; this module owns the
stateful Qdrant/indexing boundary and is imported only for actual execution.
"""

from __future__ import annotations

import math
import re
from pathlib import Path

import pandas as pd
from qdrant_client import QdrantClient
from qdrant_client.models import Distance
from tqdm.auto import tqdm

from augmentation.constructed import ConstructedDocs
from hybrid_search_rrf_dataset.fusion import (
    DenseOnlyStrategy,
    PureRRFStrategy,
    SparseOnlyStrategy,
)
from hybrid_search_rrf_dataset.indexer import (
    CorpusDocument,
    CorpusIndexer,
    EmbeddingCache,
    EmbeddingConfig,
)
from hybrid_search_rrf_dataset.labels import RouteLabels
from hybrid_search_rrf_dataset.lanes import LANES
from hybrid_search_rrf_dataset.objective import RouterObjective
from hybrid_search_rrf_dataset.retrieval import (
    RetrievalDataset,
    SnapshotDataset,
)
from scripts.label_routes import (
    DATA_DIR,
    DENSE_MODEL,
    DENSE_SIZE,
    SPARSE_MODEL,
    _corpus_rows,
    _source_name,
)

_COLLECTION_SAFE = re.compile(r"[^A-Za-z0-9_-]+")


class LaneLabelSweep:
    """Label natural or supplemented queries against frozen lane corpora."""

    def __init__(
        self,
        client: QdrantClient,
        selection: pd.DataFrame,
        *,
        out_dir: Path,
        supplemented: bool,
        run_fp: str | None = None,
    ) -> None:
        self._client = client
        self._selection = selection
        self._out_dir = Path(out_dir)
        self._supplemented = supplemented
        self._run_fp = run_fp or self._out_dir.parent.name
        self._dense, self._sparse = _embedding_configs()

    def run(self, *, only: tuple[str, ...] | None = None) -> list[str]:
        failed: list[str] = []
        for lane in _ordered_lanes(self._selection, only):
            source = _plan_source(lane, self._selection)
            collection = self._index(lane, source.corpus())
            min_relevance = LANES[lane].min_relevance if lane in LANES else 1
            labels = RouteLabels(
                self._selection,
                out_dir=self._out_dir,
                objective=RouterObjective(min_relevance=min_relevance),
                scored_against="supplemented" if self._supplemented else "natural",
            )
            args = (self._client, collection, self._dense, self._sparse)
            try:
                out = labels.label(
                    source,
                    DenseOnlyStrategy(*args),
                    PureRRFStrategy(*args),
                    SparseOnlyStrategy(*args),
                    dataset=lane,
                    include_gated=self._supplemented,
                )
            except ValueError as error:
                failed.append(lane)
                tqdm.write(f"[{lane}] SKIPPED: {error}")
                continue
            _report(lane, out, "supplemented" if self._supplemented else "natural")
        return failed

    def _index(self, lane: str, corpus: pd.DataFrame) -> str:
        collection = _collection_name(self._run_fp, lane, constructed=False)
        _ensure_index(self._client, collection, lane, corpus, self._dense, self._sparse)
        return collection


class ConstructedLabelSweep:
    """Label constructed-answer queries against isolated per-run collections."""

    def __init__(
        self,
        client: QdrantClient,
        selection: pd.DataFrame,
        *,
        out_dir: Path,
        run_fp: str | None = None,
    ) -> None:
        self._client = client
        self._selection = selection
        self._out_dir = Path(out_dir)
        self._run_fp = run_fp or self._out_dir.parent.name
        self._docs = ConstructedDocs()
        self._dense, self._sparse = _embedding_configs()

    def run(self, *, only: tuple[str, ...] | None = None) -> list[str]:
        failed: list[str] = []
        for lane in _ordered_lanes(self._selection, only):
            source = _plan_source(lane, self._selection)
            corpus = _constructed_corpus(lane, source, self._docs)
            collection = _collection_name(self._run_fp, lane, constructed=True)
            _ensure_index(
                self._client, collection, lane, corpus, self._dense, self._sparse
            )
            min_relevance = LANES[lane].min_relevance if lane in LANES else 1
            labels = RouteLabels(
                self._selection,
                out_dir=self._out_dir,
                objective=RouterObjective(min_relevance=min_relevance),
                scored_against="supplemented",
            )
            args = (self._client, collection, self._dense, self._sparse)
            try:
                out = labels.label(
                    source,
                    DenseOnlyStrategy(*args),
                    PureRRFStrategy(*args),
                    SparseOnlyStrategy(*args),
                    dataset=lane,
                    include_gated=True,
                )
            except ValueError as error:
                failed.append(lane)
                tqdm.write(f"[{lane}] SKIPPED: {error}")
                continue
            _report(lane, out, "constructed")
        return failed


class _FrozenPlanSource(RetrievalDataset):
    """Exact planned queries over an immutable lane corpus and judgments."""

    def __init__(self, source: RetrievalDataset, rows: pd.DataFrame) -> None:
        self.name = source.name
        self._source = source
        self._ids = set(rows["query_id"].astype(str))
        self._queries = pd.DataFrame({
            "query_id": rows["query_id"].astype(str),
            "text": rows["query"].astype(str),
        }).drop_duplicates("query_id", keep="first")
        self._provenance = pd.DataFrame({
            "query_id": rows["query_id"].astype(str),
            "provenance": rows["provenance"].fillna("natural").astype(str),
        }).drop_duplicates("query_id", keep="first")

    def corpus(self) -> pd.DataFrame:
        return self._source.corpus()

    def queries(self) -> pd.DataFrame:
        return self._queries.copy()

    def qrels(self) -> pd.DataFrame:
        qrels = self._source.qrels()
        return qrels[qrels["query_id"].astype(str).isin(self._ids)].reset_index(
            drop=True
        )

    def excluded(self) -> pd.DataFrame:
        excluded = self._source.excluded()
        return excluded[
            excluded["query_id"].astype(str).isin(self._ids)
        ].reset_index(drop=True)

    def provenance(self) -> pd.DataFrame:
        return self._provenance.copy()


def _plan_source(lane: str, selection: pd.DataFrame) -> _FrozenPlanSource:
    """Bind exact frozen-plan text to a lane snapshot without mutating it."""
    source = SnapshotDataset(_source_name(lane), path=str(DATA_DIR))
    rows = selection[selection["dataset"].astype(str) == lane]
    return _FrozenPlanSource(source, rows)


def _constructed_corpus(
    lane: str, source: RetrievalDataset, docs: ConstructedDocs
) -> pd.DataFrame:
    base = source.corpus()
    owned = docs.load()
    owned = owned[owned["source_dataset"].astype(str) == lane]
    constructed = pd.DataFrame({
        "doc_id": owned["doc_id"].astype(str),
        "title": "",
        "text": owned["text"].astype(str),
    })
    merged = pd.concat(
        [base, constructed.reindex(columns=base.columns, fill_value="")],
        ignore_index=True,
    )
    return merged.drop_duplicates("doc_id", keep="first")


def _embedding_configs() -> tuple[EmbeddingConfig, EmbeddingConfig]:
    return (
        EmbeddingConfig(
            name="dense_base",
            model_id=DENSE_MODEL,
            kind="dense",
            size=DENSE_SIZE,
            distance=Distance.COSINE,
            parallel=4,
        ),
        EmbeddingConfig(name="sparse_base", model_id=SPARSE_MODEL, kind="sparse"),
    )


def _ordered_lanes(selection: pd.DataFrame, only: tuple[str, ...] | None) -> list[str]:
    available = set(selection["dataset"].astype(str))
    wanted = available if only is None else available & set(only)
    sizes = {lane: _corpus_rows(lane) for lane in wanted}
    return sorted(
        wanted,
        key=lambda lane: sizes[lane] if sizes[lane] is not None else math.inf,
    )


def _collection_name(run_fp: str, lane: str, *, constructed: bool) -> str:
    suffix = "constructed_routes" if constructed else "routes"
    material = f"rung_{run_fp[:12]}_{_source_name(lane)}_{suffix}"
    return _COLLECTION_SAFE.sub("_", material)


def _ensure_index(
    client: QdrantClient,
    collection: str,
    lane: str,
    corpus: pd.DataFrame,
    dense: EmbeddingConfig,
    sparse: EmbeddingConfig,
) -> None:
    indexer = CorpusIndexer(
        client,
        collection,
        embeddings=[dense, sparse],
        cache=EmbeddingCache(namespace=_source_name(lane)),
    )
    indexer.ensure_collection()
    if client.count(collection, exact=True).count < len(corpus):
        docs = [CorpusDocument(**row) for row in corpus.to_dict("records")]
        fresh = indexer.missing(docs)
        tqdm.write(f"[{lane}] indexing {len(fresh):,}/{len(docs):,} -> {collection}")
        indexer.upload(fresh, batch_size=64)


def _report(lane: str, labelled: pd.DataFrame, regime: str) -> None:
    shape = labelled["shape"].value_counts() if not labelled.empty else {}
    tqdm.write(
        f"[{lane}] +{len(labelled):,} {regime} labels  "
        f"differ {shape.get('routes_differ', 0):,} | "
        f"tied {shape.get('all_tied', 0):,} | zero {shape.get('all_zero', 0):,}"
    )
