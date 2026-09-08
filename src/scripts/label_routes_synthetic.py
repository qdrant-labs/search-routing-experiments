"""Label the synthetic rung's rows against an ISOLATED collection.

A constructed document must never enter a lane's live `*_routes` collection —
one added doc shifts collection-level IDF and can steal rank-1 from labels the
lane already paid for. So each lane with synthetic rows gets its own
`<lane>_synthetic_routes` collection: the source corpus as distractors plus that
lane's constructed docs, exactly the design `mint_constructed` describes. Labels
land in data/v3/synthetic/, never in a paid artifact; the rows stay
feature-stock until their coherence audit passes (d42h) — a label is a
measurement, credit is what the gate holds back.

    poetry run python src/scripts/label_routes_synthetic.py --plan
    poetry run python src/scripts/label_routes_synthetic.py
"""

from __future__ import annotations

import argparse
import os

import pandas as pd
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import Distance
from tqdm.auto import tqdm

from augmentation.constructed import ConstructedDocs
from augmentation.pool import GeneratedPool
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
from hybrid_search_rrf_dataset.retrieval import SnapshotDataset
from scripts.label_routes import (
    DATA_DIR,
    DENSE_MODEL,
    DENSE_SIZE,
    SPARSE_MODEL,
    _collection,
    _source_name,
)

OUT_DIR = DATA_DIR / "v3" / "synthetic"
SYNTHETIC_OPERATOR = "synthesize"


def synthetic_collection(lane: str) -> str:
    """The isolated collection — asserted distinct from the paid one, because
    equality is exactly the contamination this script exists to prevent."""
    name = f"{_source_name(lane)}_synthetic_routes"
    assert name != _collection(lane), f"{lane}: would reuse the paid collection"
    return name


def synthetic_rows(pool: pd.DataFrame) -> pd.DataFrame:
    if pool.empty:
        return pool
    return pool[pool["operator"] == SYNTHETIC_OPERATOR]


def eval_corpus(lane: str, docs: ConstructedDocs) -> pd.DataFrame:
    """Source corpus as distractors plus the lane's constructed answer docs."""
    source = SnapshotDataset(_source_name(lane), path=str(DATA_DIR)).corpus()
    mine = docs.load()
    mine = mine[mine["source_dataset"] == lane]
    constructed = pd.DataFrame({
        "doc_id": mine["doc_id"].astype(str),
        "title": "",
        "text": mine["text"].astype(str),
    })
    merged = pd.concat(
        [source, constructed.reindex(columns=source.columns, fill_value="")],
        ignore_index=True,
    )
    return merged.drop_duplicates("doc_id", keep="first")


class SyntheticLabelSweep:
    """Index the isolated per-lane collections and label the synthetic rows."""

    def __init__(self, client: QdrantClient, selection: pd.DataFrame) -> None:
        self._client = client
        self._selection = selection
        self._docs = ConstructedDocs()
        self._dense = EmbeddingConfig(
            name="dense_base", model_id=DENSE_MODEL, kind="dense",
            size=DENSE_SIZE, distance=Distance.COSINE, parallel=4,
        )
        self._sparse = EmbeddingConfig(
            name="sparse_base", model_id=SPARSE_MODEL, kind="sparse",
        )

    def plan(self) -> pd.DataFrame:
        live = {c.name for c in self._client.get_collections().collections}
        rows = []
        for lane, group in self._selection.groupby("dataset"):
            corpus = eval_corpus(str(lane), self._docs)
            rows.append({
                "dataset": lane, "to_label": len(group),
                "constructed_docs": int(
                    corpus["doc_id"].str.startswith("constructed-").sum()
                ),
                "corpus_rows": len(corpus),
                "collection": synthetic_collection(str(lane)),
                "indexed": synthetic_collection(str(lane)) in live,
            })
        return pd.DataFrame(rows)

    def _index(self, lane: str, corpus: pd.DataFrame) -> str:
        collection = synthetic_collection(lane)
        # the lane, not the collection: this corpus is the paid lane's plus its
        # constructed docs, so sharing the lane's namespace reuses those vectors
        # while the `constructed-` ids stay unambiguous
        indexer = CorpusIndexer(
            self._client, collection,
            embeddings=[self._dense, self._sparse],
            cache=EmbeddingCache(namespace=_source_name(lane)),
        )
        indexer.ensure_collection()
        if self._client.count(collection, exact=True).count < len(corpus):
            docs = [CorpusDocument(**r) for r in corpus.to_dict("records")]
            fresh = indexer.missing(docs)
            tqdm.write(f"[{lane}] indexing {len(fresh):,}/{len(docs):,} -> {collection}")
            indexer.upload(fresh, batch_size=64)
        return collection

    def run(self, *, force: bool = False) -> list[str]:
        failed: list[str] = []
        for lane, group in self._selection.groupby("dataset"):
            lane = str(lane)
            collection = self._index(lane, eval_corpus(lane, self._docs))
            min_rel = LANES[lane].min_relevance if lane in LANES else 1
            labels = RouteLabels(
                group, out_dir=OUT_DIR,
                objective=RouterObjective(min_relevance=min_rel),
                scored_against="supplemented",
            )
            args = (self._client, collection, self._dense, self._sparse)
            source = SnapshotDataset(_source_name(lane), path=str(DATA_DIR))
            try:
                out = labels.label(
                    source,
                    DenseOnlyStrategy(*args), PureRRFStrategy(*args),
                    SparseOnlyStrategy(*args),
                    dataset=lane, force=force, include_gated=True,
                )
            except ValueError as error:
                failed.append(lane)
                tqdm.write(f"[{lane}] SKIPPED: {error}")
                continue
            shape = out["shape"].value_counts() if not out.empty else {}
            tqdm.write(
                f"[{lane}] +{len(out):,} synthetic labels  "
                f"differ {shape.get('routes_differ', 0):,} | "
                f"tied {shape.get('all_tied', 0):,} | zero {shape.get('all_zero', 0):,}"
            )
        return failed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", action="store_true", help="workload only, no retrieval")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    rows = synthetic_rows(GeneratedPool().load())
    if rows.empty:
        print("no synthetic rows in the pool — run the campaign's synthesize rung first")
        return
    selection = pd.DataFrame({
        "dataset": rows["home_lane"].astype(str),
        "query_id": rows["query_id"].astype(str),
        "query": rows["query"].astype(str),
    })
    print(f"synthetic rows to label: {len(selection):,} across "
          f"{selection['dataset'].nunique()} lane(s)")

    load_dotenv()
    client = QdrantClient(
        url=os.getenv("QDRANT_URL", "http://localhost:6333"),
        api_key=os.getenv("QDRANT_API_KEY"), timeout=60,
    )
    sweep = SyntheticLabelSweep(client, selection)
    if args.plan:
        print(sweep.plan().to_string(index=False))
        return
    failed = sweep.run(force=args.force)
    if failed:
        print(f"skipped: {failed}")
    print(f"synthetic labels -> {OUT_DIR / 'labels.parquet'}")


if __name__ == "__main__":
    main()
