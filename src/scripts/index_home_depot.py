"""Index the Home Depot corpus into Qdrant Cloud for the out-of-lane Top-1 bakeoff.

Dense all-MiniLM-L6-v2 embeds server-side via Qdrant Cloud Inference (no local
model load, no 124K local vectors); sparse bm25/IDF stays local — tokenization
is cheap and unpaid. Full catalog, idempotent via the indexer's `missing` skip.
"""

from __future__ import annotations

import argparse
import os
from collections.abc import Iterator

from qdrant_client import QdrantClient
from tqdm.auto import tqdm

from hybrid_search_rrf_dataset.indexer import (
    CorpusDocument,
    CorpusIndexer,
    EmbeddingConfig,
)
from hybrid_search_rrf_dataset.retrieval.wave3 import HOME_DEPOT_DIR, HomeDepotLane

# Vector-slot names the bakeoff's fusion strategies will query under. Dense embeds
# server-side (cloud=True); bm25 is a local, unpaid tokenization pass. all-MiniLM-L6-v2
# (384d, symmetric): bge-small has no endpoint on the cloud cluster.
DENSE = EmbeddingConfig(
    name="dense_base",
    model_id="sentence-transformers/all-MiniLM-L6-v2",
    kind="dense",
    size=384,
    cloud=True,
)
SPARSE = EmbeddingConfig(name="sparse_base", model_id="Qdrant/bm25", kind="sparse")


def _batches(items: Iterator[CorpusDocument], size: int) -> Iterator[list[CorpusDocument]]:
    batch: list[CorpusDocument] = []
    for item in items:
        batch.append(item)
        if len(batch) >= size:
            yield batch
            batch = []
    if batch:
        yield batch


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collection", default="home-depot")
    parser.add_argument(
        "--qdrant-url",
        default=os.environ.get("QDRANT_URL") or os.environ.get("QDRANT_CLOUD_URL"),
        help="Qdrant Cloud cluster URL (default: $QDRANT_URL or $QDRANT_CLOUD_URL)",
    )
    parser.add_argument("--data-dir", default=None, help="dir holding the Kaggle CSVs")
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument(
        "--parallel", type=int, default=8,
        help="concurrent cloud upserts — the embed runs server-side, so this scales throughput",
    )
    parser.add_argument("--recreate", action="store_true")
    parser.add_argument("--include-attributes", action="store_true")
    args = parser.parse_args()

    api_key = os.environ.get("QDRANT_API_KEY")
    if not args.qdrant_url:
        parser.error("set --qdrant-url or $QDRANT_URL/$QDRANT_CLOUD_URL (Qdrant Cloud)")
    if not api_key:
        parser.error("set $QDRANT_API_KEY — cloud inference needs an authenticated cluster")

    lane = HomeDepotLane(
        args.data_dir or HOME_DEPOT_DIR, include_attributes=args.include_attributes
    )
    client = QdrantClient(
        url=args.qdrant_url, api_key=api_key, cloud_inference=True, timeout=120
    )
    indexer = CorpusIndexer(
        client=client, collection_name=args.collection, embeddings=[DENSE, SPARSE]
    )
    indexer.ensure_collection(recreate=args.recreate)

    docs = (
        CorpusDocument(doc_id=row["doc_id"], title=row["title"], text=row["text"])
        for row in lane._iter_corpus()
    )
    uploaded = 0
    for batch in tqdm(_batches(docs, args.batch_size), desc="indexing", unit="batch"):
        fresh = indexer.missing(batch)
        if fresh:
            indexer.upload(fresh, batch_size=64, parallel=args.parallel)
            uploaded += len(fresh)
    print(
        f"indexed home-depot: {uploaded} new docs into '{args.collection}' "
        f"at {args.qdrant_url}"
    )


if __name__ == "__main__":
    main()
