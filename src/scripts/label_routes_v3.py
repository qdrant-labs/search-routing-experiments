"""v3 additive labelling — index + label MORE queries into
`data/v3/labels.parquet`, NEVER touching v2's `labels.parquet`. Driven by the
"label more" lanes in the v3 feasibility report (under-labelled lanes with good
decisive yield). Reuses the shared Qdrant collections + indexer + fusion
strategies; only the output path (data/v3) and the selection differ.

This is the in-loop labelling step: V3Composition.build -> label_routes_v3
-> build again (the new v3 labels feed the next selection). The
actual index+label pass embeds corpora and needs `docker compose up -d`;
`--plan` reports the workload with no retrieval.

    poetry run python src/scripts/label_routes_v3.py --plan
    poetry run python src/scripts/label_routes_v3.py --only bright-leetcode
    poetry run python src/scripts/label_routes_v3.py
"""

from __future__ import annotations

import argparse
import math
import os

import pandas as pd
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, Modifier
from tqdm.auto import tqdm

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
    _corpus_rows,
    _source_name,
)

V3_DIR = DATA_DIR / "v3"
PER_DATASET = DATA_DIR / "v3" / "per_dataset.parquet"
V2_LABELS = DATA_DIR / "route_labels" / "labels.parquet"
DEFAULT_YIELD_FLOOR = 0.05  # keep the "need" size finite when yield is tiny


def label_more_selection(oversample: float = 1.0) -> pd.DataFrame:
    """Additional (dataset, query_id, query) rows to label for the label-more
    lanes: source queries NOT already labelled in v2 OR a previous v3 campaign,
    sized to net each lane's floor gap given its decisive yield. The selection
    RouteLabels reads."""
    per_ds = pd.read_parquet(PER_DATASET)
    label_more = per_ds[per_ds["floor_action"] == "label more"]
    done = [pd.read_parquet(V2_LABELS, columns=["dataset", "query_id"])]
    v3_labels = V3_DIR / "labels.parquet"
    if v3_labels.exists():
        done.append(pd.read_parquet(v3_labels, columns=["dataset", "query_id"]))
    labelled = pd.concat(done, ignore_index=True).astype({"query_id": str})
    seen = labelled.groupby("dataset")["query_id"].agg(set).to_dict()

    frames: list[pd.DataFrame] = []
    for dataset, row in label_more.iterrows():
        qpath = DATA_DIR / _source_name(str(dataset)) / "queries.parquet"
        if not qpath.exists():
            continue
        queries = pd.read_parquet(qpath).astype({"query_id": str})
        if "query" not in queries.columns:
            queries = queries.rename(columns={"text": "query"})
        fresh = queries[
            ~queries["query_id"].isin(seen.get(str(dataset), set()))
        ]
        yield_rate = max(float(row["yield_rate"]), DEFAULT_YIELD_FLOOR)
        need = math.ceil(row["floor_gap"] / yield_rate * oversample)
        take = fresh.head(need).assign(dataset=str(dataset), home_lane=str(dataset))
        frames.append(take[["dataset", "query_id", "query", "home_lane"]])
    if not frames:
        return pd.DataFrame(columns=["dataset", "query_id", "query", "home_lane"])
    return pd.concat(frames, ignore_index=True)


def class_supply_selection(spec: dict[str, int | None]) -> pd.DataFrame:
    """Fresh queries from named lanes for CLASS supply (the sparse ceiling),
    not floor gaps: lane -> how many to take, None = every fresh query.
    Ordered by the useful-yield priority, capped where the lane cap makes
    further labelling wasted spend (clerc past ~16K)."""
    done = [pd.read_parquet(V2_LABELS, columns=["dataset", "query_id"])]
    v3_labels = V3_DIR / "labels.parquet"
    if v3_labels.exists():
        done.append(pd.read_parquet(v3_labels, columns=["dataset", "query_id"]))
    labelled = pd.concat(done, ignore_index=True).astype({"query_id": str})
    seen = labelled.groupby("dataset")["query_id"].agg(set).to_dict()

    frames: list[pd.DataFrame] = []
    for dataset, take_n in spec.items():
        qpath = DATA_DIR / _source_name(dataset) / "queries.parquet"
        if not qpath.exists():
            print(f"[{dataset}] no queries.parquet — skipped")
            continue
        queries = pd.read_parquet(qpath).astype({"query_id": str})
        if "query" not in queries.columns:
            queries = queries.rename(columns={"text": "query"})
        fresh = queries[~queries["query_id"].isin(seen.get(dataset, set()))]
        take = fresh if take_n is None else fresh.head(take_n)
        frames.append(
            take.assign(dataset=dataset, home_lane=dataset)[
                ["dataset", "query_id", "query", "home_lane"]
            ]
        )
    if not frames:
        return pd.DataFrame(columns=["dataset", "query_id", "query", "home_lane"])
    return pd.concat(frames, ignore_index=True)


class V3LabelSweep:
    """Index + label the v3 selection into data/v3, reusing shared collections."""

    def __init__(self, client: QdrantClient, selection: pd.DataFrame) -> None:
        self._client = client
        self._selection = selection
        self._dense = EmbeddingConfig(
            name="dense_base", model_id=DENSE_MODEL, kind="dense",
            size=DENSE_SIZE, distance=Distance.COSINE, parallel=4,
        )
        self._sparse = EmbeddingConfig(
            # IDF is collection schema; a fresh/recreated lane must match the
            # live *_routes collections, which already carry it. Omitting it
            # silently rebuilds TF-only and corrupts the sparse signal.
            name="sparse_base", model_id=SPARSE_MODEL, kind="sparse",
            modifier=Modifier.IDF,
        )
        self._cache = EmbeddingCache("./.embedding_cache")

    def _keys(self, keys: tuple[str, ...] | None) -> list[str]:
        wanted = list(keys) if keys else sorted(self._selection["dataset"].unique())
        sizes = {k: _corpus_rows(k) for k in wanted}
        return sorted(wanted, key=lambda k: sizes[k] if sizes[k] is not None else math.inf)

    def plan(self, keys: tuple[str, ...] | None = None) -> pd.DataFrame:
        """Workload per lane — no retrieval, no Qdrant writes."""
        live = {c.name for c in self._client.get_collections().collections}
        rows = []
        for key in self._keys(keys):
            n = int((self._selection["dataset"] == key).sum())
            rows.append({
                "dataset": key, "to_label": n,
                "corpus_rows": _corpus_rows(key),
                "indexed": _collection(key) in live,
            })
        return pd.DataFrame(rows)

    def _index(self, key: str, corpus: pd.DataFrame) -> str:
        collection = _collection(key)
        indexer = CorpusIndexer(
            self._client, collection,
            embeddings=[self._dense, self._sparse], cache=self._cache,
        )
        indexer.ensure_collection()
        if self._client.count(collection, exact=True).count < len(corpus):
            docs = [CorpusDocument(**r) for r in corpus.to_dict("records")]
            fresh = indexer.missing(docs)
            tqdm.write(f"[{key}] indexing {len(fresh):,}/{len(docs):,} -> {collection}")
            indexer.upload(fresh, batch_size=64)
        return collection

    def run(self, keys: tuple[str, ...] | None = None, *, force: bool = False) -> list[str]:
        failed: list[str] = []
        for key in self._keys(keys):
            source = SnapshotDataset(_source_name(key), path=str(DATA_DIR))
            collection = self._index(key, source.corpus())
            min_rel = LANES[key].min_relevance if key in LANES else 1
            labels = RouteLabels(
                self._selection, out_dir=V3_DIR,
                objective=RouterObjective(min_relevance=min_rel),
            )
            args = (self._client, collection, self._dense, self._sparse)
            try:
                out = labels.label(
                    source,
                    DenseOnlyStrategy(*args), PureRRFStrategy(*args), SparseOnlyStrategy(*args),
                    dataset=key, force=force,
                )
            except ValueError as error:
                failed.append(key)
                tqdm.write(f"[{key}] SKIPPED: {error}")
                continue
            shape = out["shape"].value_counts() if not out.empty else {}
            tqdm.write(
                f"[{key}] +{len(out):,} new v3 labels  "
                f"differ {shape.get('routes_differ', 0):,} | "
                f"tied {shape.get('all_tied', 0):,} | zero {shape.get('all_zero', 0):,}"
            )
        return failed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", action="store_true", help="workload only, no retrieval")
    parser.add_argument("--only", nargs="*", default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--oversample", type=float, default=1.0)
    parser.add_argument(
        "--supply", nargs="*", default=None, metavar="LANE[=N]",
        help="class-supply mode: label N fresh queries (default: all) from "
        "each named lane, instead of the floor-gap selection",
    )
    args = parser.parse_args()

    if args.supply:
        spec = {
            (part.split("=", 1)[0]): (
                int(part.split("=", 1)[1]) if "=" in part else None
            )
            for part in args.supply
        }
        selection = class_supply_selection(spec)
    else:
        selection = label_more_selection(oversample=args.oversample)
    print(f"v3 label-more selection: {len(selection):,} queries across "
          f"{selection['dataset'].nunique()} lanes")
    if selection.empty:
        return

    load_dotenv()
    client = QdrantClient(
        url=os.getenv("QDRANT_URL", "http://localhost:6333"),
        api_key=os.getenv("QDRANT_API_KEY"), timeout=60,
    )
    sweep = V3LabelSweep(client, selection)
    keys = tuple(args.only) if args.only else None
    if args.plan:
        print(sweep.plan(keys).to_string(index=False))
        return
    failed = sweep.run(keys, force=args.force)
    if failed:
        print(f"skipped (no judged queries): {failed}")
    print(f"v3 labels -> {V3_DIR / 'labels.parquet'}")


if __name__ == "__main__":
    main()
