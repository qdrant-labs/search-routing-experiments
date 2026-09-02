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
    poetry run python src/scripts/label_routes_v3.py --admitted
    poetry run python src/scripts/label_routes_v3.py --v4
    poetry run python src/scripts/label_routes_v3.py
"""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import Distance
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
from scripts.label_routes_synthetic import SYNTHETIC_OPERATOR

V3_DIR = DATA_DIR / "v3"
AUGMENTED_DIR = V3_DIR / "augmented"
V4_DIR = DATA_DIR / "v4"
V4_PICKS = V4_DIR / "rung_a_picks.parquet"
V4_TARGETS = V4_DIR / "composition_targets_v4.parquet"
V4_AUGMENTED_DIR = V4_DIR / "augmented"
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


def order_selection(cap: int | None = None) -> pd.DataFrame:
    """The selection the SelectionOrder rung wrote (data/v3/
    selection_order.parquet): query text joined from the source parquets,
    already-labelled rows dropped, `cap` distributed across lanes
    proportionally so a bounded batch keeps the order's lane mix."""
    order_path = V3_DIR / "selection_order.parquet"
    order = pd.read_parquet(order_path).astype({"query_id": str})
    done = [pd.read_parquet(V2_LABELS, columns=["dataset", "query_id"])]
    v3_labels = V3_DIR / "labels.parquet"
    if v3_labels.exists():
        done.append(pd.read_parquet(v3_labels, columns=["dataset", "query_id"]))
    labelled = pd.concat(done, ignore_index=True).astype({"query_id": str})
    seen = labelled.groupby("dataset")["query_id"].agg(set).to_dict()

    frames: list[pd.DataFrame] = []
    for dataset, lines in order.groupby("dataset"):
        qpath = DATA_DIR / _source_name(str(dataset)) / "queries.parquet"
        if not qpath.exists():
            continue
        queries = pd.read_parquet(qpath).astype({"query_id": str})
        if "query" not in queries.columns:
            queries = queries.rename(columns={"text": "query"})
        fresh = lines[~lines["query_id"].isin(seen.get(str(dataset), set()))]
        if cap is not None:
            fresh = fresh.head(max(1, math.ceil(cap * len(lines) / len(order))))
        take = fresh.merge(queries[["query_id", "query"]], on="query_id")
        frames.append(
            take.assign(dataset=str(dataset), home_lane=str(dataset))[
                ["dataset", "query_id", "query", "home_lane"]
            ]
        )
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


def picks_selection(path: Path) -> pd.DataFrame:
    """Queries a pick file names, minus everything a label file already covers;
    the synthesize rung is excluded because its answer docs live only in the
    isolated collection label_routes_synthetic indexes."""
    columns = ["dataset", "query_id", "query", "home_lane"]
    if not path.exists():
        print(f"no {path} — build the picks first")
        return pd.DataFrame(columns=columns)
    admitted = pd.read_parquet(path)
    if admitted.empty:
        return pd.DataFrame(columns=columns)
    return _fresh_picks(admitted)


def _fresh_picks(admitted: pd.DataFrame) -> pd.DataFrame:
    """Pick rows minus everything a label file already covers, keyed on
    (dataset, query_id); the synthesize rung is excluded (its answer docs live
    only in the isolated synthetic collection). Existing labels save the work,
    they never pick the rows — the caller already chose these."""
    columns = ["dataset", "query_id", "query", "home_lane"]
    if admitted.empty:
        return pd.DataFrame(columns=columns)
    admitted = admitted.astype({"query_id": str})
    done = [pd.read_parquet(V2_LABELS, columns=["dataset", "query_id"])]
    for labels in (
        V3_DIR / "labels.parquet",
        V3_DIR / "synthetic" / "labels.parquet",
        AUGMENTED_DIR / "labels.parquet",
        V4_DIR / "labels.parquet",
        V4_AUGMENTED_DIR / "labels.parquet",
    ):
        if labels.exists():
            done.append(pd.read_parquet(labels, columns=["dataset", "query_id"]))
    # Key on (dataset, query_id): natural query_ids are per-dataset locals,
    # only augmented UUIDs are globally unique.
    done_frame = pd.concat(done, ignore_index=True).astype(
        {"dataset": str, "query_id": str}
    )
    seen = set(zip(done_frame["dataset"], done_frame["query_id"]))
    admitted_lane = admitted["home_lane"].astype(str)
    fresh = admitted.assign(dataset=admitted_lane, home_lane=admitted_lane)
    keys = list(zip(fresh["dataset"].astype(str), fresh["query_id"].astype(str)))
    is_fresh = pd.Series([key not in seen for key in keys], index=fresh.index)
    fresh = fresh[
        is_fresh & (fresh["operator"] != SYNTHETIC_OPERATOR)
    ].drop_duplicates(["dataset", "query_id"])
    return fresh[columns]


def answer_covered(selection: pd.DataFrame) -> pd.DataFrame:
    """The rows a lane's on-disk qrels answer at its own min_relevance. Runs
    AFTER materialize_rung_a has landed tonight's qrels: retrieval on a query
    with no judged doc is paid for and then dropped by `label`."""
    kept = []
    for lane, group in selection.groupby("dataset"):
        spec = LANES.get(str(lane))
        source = spec.source.name if spec is not None else str(lane)
        min_rel = spec.min_relevance if spec is not None else 1
        path = DATA_DIR / source / "qrels.parquet"
        if not path.exists():
            continue
        qrels = pd.read_parquet(path, columns=["query_id", "relevance"])
        judged = set(
            qrels.loc[qrels["relevance"] >= min_rel, "query_id"].astype(str)
        )
        kept.append(group[group["query_id"].astype(str).isin(judged)])
    return (
        pd.concat(kept, ignore_index=True) if kept else selection.iloc[:0]
    )


def lane_minted_selection(audit_path: Path | None = None) -> pd.DataFrame:
    """Lane-rung rows cleared for labelling, straight from the generated
    pool: a row's own passed verdict always clears it, a failed verdict
    always blocks it, and an UNJUDGED row flows once the operator's judged
    sample passes at the coherence bar (decided 2026-08-21: doc-grounded
    mints measured 98.4% pass, so per-row judging is a sample audit now) —
    minus the free refusal-text guard."""
    from augmentation.config import AugmentationConfig
    from augmentation.lane_synthetic import LANE_OPERATOR, is_refusal
    from augmentation.pool import GeneratedPool

    columns = ["dataset", "query_id", "query", "home_lane"]
    pool = GeneratedPool().load()
    if pool.empty:
        return pd.DataFrame(columns=columns)
    pool = pool.astype({"query_id": str})
    lane_rows = pool[pool["operator"] == LANE_OPERATOR]
    config = AugmentationConfig()
    path = audit_path if audit_path is not None else config.paths.coherence_audit
    verdicts = (
        pd.read_parquet(path).set_index("query_id")["verdict"].astype(bool)
        if path.exists() else pd.Series(dtype=bool)
    )
    judged = lane_rows["query_id"].map(verdicts)
    sample = judged.dropna()
    sample_cleared = (
        len(sample) >= config.pilot_n
        and float(sample.mean()) >= config.coherence_pass_rate
    )
    refusal = lane_rows["query"].astype(str).map(is_refusal)
    ok = judged.eq(True) | (
        sample_cleared & judged.isna() & ~refusal
    )
    rows = lane_rows[ok.fillna(False).astype(bool)]
    done = [pd.read_parquet(V2_LABELS, columns=["query_id"])]
    for labels in (
        V3_DIR / "labels.parquet",
        V3_DIR / "synthetic" / "labels.parquet",
        AUGMENTED_DIR / "labels.parquet",
    ):
        if labels.exists():
            done.append(pd.read_parquet(labels, columns=["query_id"]))
    seen = set(pd.concat(done, ignore_index=True)["query_id"].astype(str))
    fresh = rows[~rows["query_id"].isin(seen)].drop_duplicates("query_id")
    lane = fresh["home_lane"].astype(str)
    return fresh.assign(dataset=lane, home_lane=lane)[columns]


class V3LabelSweep:
    """Index + label the v3 selection into data/v3, reusing shared collections.
    `augmented` moves the output to data/v3/augmented and labels the admitted
    generated rows: supplemented qrels, and gated rows measured anyway."""

    def __init__(
        self,
        client: QdrantClient,
        selection: pd.DataFrame,
        *,
        augmented: bool = False,
        out_dir: Path | None = None,
    ) -> None:
        self._client = client
        self._selection = selection
        self._augmented = augmented
        self._out_dir = out_dir or (AUGMENTED_DIR if augmented else V3_DIR)
        self._dense = EmbeddingConfig(
            name="dense_base", model_id=DENSE_MODEL, kind="dense",
            size=DENSE_SIZE, distance=Distance.COSINE, parallel=4,
        )
        self._sparse = EmbeddingConfig(
            name="sparse_base", model_id=SPARSE_MODEL, kind="sparse",
        )

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
        # per lane, not per sweep: `item_id` hashes a bare doc_id and lanes
        # share doc_ids, so one cache across lanes serves the wrong vectors
        indexer = CorpusIndexer(
            self._client, collection,
            embeddings=[self._dense, self._sparse],
            cache=EmbeddingCache(namespace=_source_name(key)),
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
                self._selection,
                out_dir=self._out_dir,
                objective=RouterObjective(min_relevance=min_rel),
                scored_against="supplemented" if self._augmented else "natural",
            )
            args = (self._client, collection, self._dense, self._sparse)
            try:
                out = labels.label(
                    source,
                    DenseOnlyStrategy(*args), PureRRFStrategy(*args), SparseOnlyStrategy(*args),
                    dataset=key, force=force, include_gated=self._augmented,
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


def _label_v4_targets(
    *, plan: bool, force: bool, keys: tuple[str, ...] | None
) -> None:
    """Two sweeps over the v4 composition targets, split by provenance: natural
    rows label against their lane corpus (answer-covered only); generated rows
    use the supplemented/augmented regime. Synthetic rows are excluded by
    `_fresh_picks` (their answer docs live in the isolated collection). Already
    labelled targets fall out via the same dedup — reused, never relabelled."""
    if not V4_TARGETS.exists():
        print(f"no {V4_TARGETS} — run compose_v4.py first")
        return
    targets = pd.read_parquet(V4_TARGETS).astype({"dataset": str, "query_id": str})
    is_natural = targets["provenance"].astype(str) == "natural"
    natural = answer_covered(_fresh_picks(targets[is_natural]))
    generated = _fresh_picks(targets[~is_natural])
    print(
        f"v4 targets: {len(targets):,} selected | to label: "
        f"{len(natural):,} natural (answer-covered) + {len(generated):,} generated"
    )

    load_dotenv()
    client = QdrantClient(
        url=os.getenv("QDRANT_URL", "http://localhost:6333"),
        api_key=os.getenv("QDRANT_API_KEY"), timeout=60,
    )
    for selection, augmented, out_dir in (
        (natural, False, V4_DIR),
        (generated, True, V4_AUGMENTED_DIR),
    ):
        if selection.empty:
            continue
        sweep = V3LabelSweep(client, selection, augmented=augmented, out_dir=out_dir)
        label = "generated" if augmented else "natural"
        if plan:
            print(f"[{label} -> {out_dir}]")
            print(sweep.plan(keys).to_string(index=False))
            continue
        failed = sweep.run(keys, force=force)
        if failed:
            print(f"[{label}] skipped (no judged queries): {failed}")
        print(f"[{label}] labels -> {out_dir / 'labels.parquet'}")


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
    parser.add_argument(
        "--order", action="store_true",
        help="label the SelectionOrder rung's picks (selection_order.parquet) "
        "instead of the floor-gap selection",
    )
    parser.add_argument(
        "--cap", type=int, default=None,
        help="with --order: bound this batch, distributed across lanes "
        "proportionally to the order",
    )
    parser.add_argument(
        "--admitted", action="store_true",
        help="label the generated rows the composer admitted "
        "(data/v3/admitted.parquet) into data/v3/augmented",
    )
    parser.add_argument(
        "--v4", action="store_true",
        help="label the v4 rung's picks (data/v4/rung_a_picks.parquet) into "
        "data/v4/labels.parquet — reads and writes nothing under data/v3",
    )
    parser.add_argument(
        "--v4-targets", action="store_true", dest="v4_targets",
        help="label the v4 COMPOSITION targets "
        "(data/v4/composition_targets_v4.parquet): natural rows into "
        "data/v4/labels.parquet, generated rows into data/v4/augmented — the "
        "pre-label composition drives the selection, not a v3 order sheet",
    )
    args = parser.parse_args()

    if args.v4_targets:
        _label_v4_targets(
            plan=args.plan,
            force=args.force,
            keys=tuple(args.only) if args.only else None,
        )
        return

    if args.v4:
        picks = picks_selection(V4_PICKS)
        selection = answer_covered(picks)
        print(
            f"v4 picks: {len(picks):,} unlabelled, {len(selection):,} with a "
            "judged doc"
        )
    elif args.admitted:
        selection = picks_selection(V3_DIR / "admitted.parquet")
    elif args.order:
        selection = order_selection(cap=args.cap)
    elif args.supply:
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
    out_dir = V4_DIR if args.v4 else AUGMENTED_DIR if args.admitted else V3_DIR
    sweep = V3LabelSweep(
        client, selection, augmented=args.admitted, out_dir=out_dir
    )
    keys = tuple(args.only) if args.only else None
    if args.plan:
        print(sweep.plan(keys).to_string(index=False))
        return
    failed = sweep.run(keys, force=args.force)
    if failed:
        print(f"skipped (no judged queries): {failed}")
    print(f"labels -> {out_dir / 'labels.parquet'}")


if __name__ == "__main__":
    main()
