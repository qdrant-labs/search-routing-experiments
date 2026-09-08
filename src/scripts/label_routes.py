"""Generate the router's training targets: index every lane's corpus into
Qdrant, then label its selection rows into `data/route_labels/labels.parquet`
(SPEC d38-d44, d61). Ports `notebooks/route_labels.ipynb`'s §16-17 sweep into
a script that survives a closed laptop lid — the dense pass on one lane alone
runs the better part of an hour, which a notebook cell does not survive a
kernel restart.

Cheapest corpus first, so a long dense-embedding pass on one big lane never
blocks a dozen quick wins. Every step is idempotent — an already-indexed
collection skips the upload, an already-labelled lane skips retrieval
entirely — so a rerun after a Ctrl-C or a dead laptop costs nothing twice.

Prerequisite: `docker compose up -d` (local Qdrant on :6333).

    poetry run python src/scripts/label_routes.py --plan
    poetry run python src/scripts/label_routes.py 2>&1 | tee label_routes.log
    poetry run python src/scripts/label_routes.py --only quest gooaq
    poetry run python src/scripts/label_routes.py --generation floor_based --force --only quest
"""

import argparse
import os
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from pyarrow.parquet import ParquetFile
from qdrant_client import QdrantClient
from qdrant_client.models import Distance
from tqdm.auto import tqdm

from hybrid_search_rrf_dataset.paths import LanePaths
from composition import CellFill
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

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DENSE_MODEL, DENSE_SIZE = "BAAI/bge-small-en-v1.5", 384
SPARSE_MODEL = "Qdrant/bm25"

COLLECTION_OVERRIDE = {
    "beir-nfcorpus": "nfcorpus_routes",
    "msmarco-passage-dev": "msmarco_routes",
}
"""The two anchor lanes predate the `{source.name}_routes` convention —
override so a run reuses their existing index instead of re-embedding under
a fresh name."""


def _source_name(key: str) -> str:
    return LANES[key].source.name if key in LANES else key



def _paths() -> LanePaths:
    """Resolved per call, so redirecting this module's DATA_DIR redirects the
    lane reads with it."""
    return LanePaths(data_dir=DATA_DIR)

def _collection(key: str) -> str:
    return COLLECTION_OVERRIDE.get(key, f"{_source_name(key)}_routes")


def _corpus_rows(key: str) -> int | None:
    path = _paths().lane_corpus(_source_name(key))
    return ParquetFile(path).metadata.num_rows if path.exists() else None


class RouteLabelSweep:
    """One pass over the composition's lanes: index into Qdrant, then label —
    cheapest corpus first, skipping whatever is already done."""

    def __init__(self, client: QdrantClient) -> None:
        self._client = client
        self._dense_cfg = EmbeddingConfig(
            name="dense_base", model_id=DENSE_MODEL, kind="dense",
            size=DENSE_SIZE, distance=Distance.COSINE, parallel=4,
        )
        self._sparse_cfg = EmbeddingConfig(
            name="sparse_base", model_id=SPARSE_MODEL, kind="sparse"
        )
        self.selection = CellFill().build()
        self.labels = RouteLabels(self.selection)

    def _order(self, keys: tuple[str, ...] | None) -> list[str]:
        wanted = (
            list(dict.fromkeys(keys)) if keys
            else sorted(self.selection["dataset"].astype(str).unique())
        )
        sizes = {key: _corpus_rows(key) for key in wanted}
        return sorted(wanted, key=lambda k: sizes[k] if sizes[k] is not None else float("inf"))

    def plan(self, keys: tuple[str, ...] | None = None) -> pd.DataFrame:
        """Readiness per lane without indexing or labeling anything."""
        live = {c.name for c in self._client.get_collections().collections}
        done = self._labelled()
        return pd.DataFrame([
            {
                "dataset": key,
                "corpus_rows": _corpus_rows(key),
                "indexed": _collection(key) in live,
                "labelled": key in done,
            }
            for key in self._order(keys)
        ])

    def _labelled(self) -> set[str]:
        if not self.labels.labels_path.exists():
            return set()
        return set(self.labels.load()["dataset"].unique())

    def _index(self, key: str, corpus: pd.DataFrame) -> str:
        collection = _collection(key)
        # per lane, not per sweep: `item_id` hashes a bare doc_id and lanes
        # share doc_ids, so one cache across lanes serves the wrong vectors
        indexer = CorpusIndexer(
            self._client, collection,
            embeddings=[self._dense_cfg, self._sparse_cfg],
            cache=EmbeddingCache(namespace=_source_name(key)),
        )
        indexer.ensure_collection()
        # count is the cheap "did this corpus grow?" trigger; the id diff is
        # what makes the upload incremental once it has.
        if self._client.count(collection, exact=True).count < len(corpus):
            docs = [CorpusDocument(**r) for r in corpus.to_dict("records")]
            fresh = indexer.missing(docs)
            tqdm.write(
                f"[{key}] indexing {len(fresh):,} of {len(docs):,} docs -> {collection}"
            )
            indexer.upload(fresh, batch_size=64)
        return collection

    def run(
        self,
        keys: tuple[str, ...] | None = None,
        *,
        force: bool = False,
        generation: str = "cell_based",
    ) -> list[str]:
        """Index and label every lane in cheapest-first order, returning the
        keys that failed — a lane with no judged queries must not kill the
        batch. `label()` itself is incremental (only new query_ids get
        scored), so every lane runs every time — cheap when there's nothing
        new, not a reason to pre-skip here."""
        failed: list[str] = []
        for key in self._order(keys):
            source = SnapshotDataset(_source_name(key), path=str(DATA_DIR))
            corpus = source.corpus()
            collection = self._index(key, corpus)

            min_rel = LANES[key].min_relevance if key in LANES else 1
            lane_labels = RouteLabels(
                self.selection, objective=RouterObjective(min_relevance=min_rel)
            )
            args = (self._client, collection, self._dense_cfg, self._sparse_cfg)
            tqdm.write(f"[{key}] labeling against {collection}")
            try:
                out = lane_labels.label(
                    source,
                    DenseOnlyStrategy(*args), PureRRFStrategy(*args), SparseOnlyStrategy(*args),
                    dataset=key, force=force, generation=generation,
                )
            except ValueError as error:
                failed.append(key)
                tqdm.write(f"[{key}] SKIPPED: {error}")
                continue

            if out.empty:
                tqdm.write(f"[{key}] nothing new — already labelled")
                continue
            shape = out["shape"].value_counts()
            tqdm.write(
                f"[{key}] corpus {len(corpus):>7,}  min_rel {min_rel}  "
                f"+{len(out):,} new  differ {shape.get('routes_differ', 0):,} "
                f"| tied {shape.get('all_tied', 0):,} | zero {shape.get('all_zero', 0):,}"
            )
        return failed


def _print_coverage(labels: RouteLabels) -> None:
    coverage = labels.coverage()
    print(coverage.to_string(index=False))
    print(
        f"selected {coverage.selected.sum():,} | labelled {coverage.labelled.sum():,} "
        f"| unlabelled {coverage.unlabelled.sum():,} "
        f"({coverage.unlabelled.sum() / coverage.selected.sum() * 100:.1f}%)"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Index and label every lane's selection rows into labels.parquet."
    )
    parser.add_argument(
        "--only", nargs="*", metavar="LANE",
        help="act on these lane keys only (default: every selected lane)",
    )
    parser.add_argument(
        "--plan", action="store_true",
        help="print per-lane readiness and overall coverage, then stop",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="relabel lanes that already have rows in labels.parquet",
    )
    parser.add_argument(
        "--generation", choices=["floor_based", "cell_based"], default="cell_based",
        help=(
            "which augmentation generation's rows to layer in via "
            "QuerySupplement (SPEC d61, default: cell_based)"
        ),
    )
    args = parser.parse_args()

    only = tuple(args.only) if args.only else None
    unknown = sorted(set(only or ()) - set(LANES))
    if unknown:
        parser.error(f"unknown lanes: {', '.join(unknown)}")

    load_dotenv(".env") or load_dotenv("../.env")
    client = QdrantClient(
        url=os.getenv("QDRANT_URL", "http://localhost:6333"),
        api_key=os.getenv("QDRANT_API_KEY"),
    )

    sweep = RouteLabelSweep(client)
    print(sweep.plan(only).to_string(index=False))
    _print_coverage(sweep.labels)
    if args.plan:
        return

    failed = sweep.run(only, force=args.force, generation=args.generation)
    print()
    _print_coverage(sweep.labels)
    if failed:
        print(f"failed ({len(failed)}): {', '.join(failed)}")


if __name__ == "__main__":
    main()
