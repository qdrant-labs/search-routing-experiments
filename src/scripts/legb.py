"""Leg-2: relabel pilot lanes with an alternative dense encoder and diff the
result against leg-1. Two readouts from one pass — how many `all_zero` rows a
stronger stack rescues, and whether any decisive label is stack-specific.

The encoder is the ONLY thing that changes: separate `<lane>_legb_<model>_routes`
collections, labels in their own `data/legb_pilot/` dir, scores under the same
three route names in a different file — so a leg-2 score can never reach the
selection surface (guarded by test_wasted_recall_objective). Notebook
`leg2_encoder_pilot.ipynb` drives this; the machinery lives here so the flag
policy and the collection guard stay unit-tested.
"""

from __future__ import annotations

import os
from typing import Literal

import pandas as pd
from qdrant_client import QdrantClient
from qdrant_client.models import Distance
from tqdm.auto import tqdm

from composition.pool_v3 import CEILING, CLASSES, LabelledPool, native_mask
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
from hybrid_search_rrf_dataset.labels import (
    ALL_TIED,
    ALL_ZERO,
    ROUTES_DIFFER,
    RouteLabels,
)
from hybrid_search_rrf_dataset.lanes import LANES
from hybrid_search_rrf_dataset.objective import RouterObjective
from hybrid_search_rrf_dataset.retrieval import SnapshotDataset
from scripts.label_routes import DATA_DIR, _collection, _source_name

Supply = Literal["native", "v2", "both"]
"""Which rows to relabel: v3-native supply (the dataset), v2-origin control, or
both. The pool merges the two, so this must be chosen, not assumed."""

SHAPES = frozenset({ROUTES_DIFFER, ALL_TIED, ALL_ZERO})
"""`sample`'s population matches `shape` for these, `kind` otherwise. Note
`all_zero` is BOTH a shape and a kind, and they agree on it, so the branch is
unambiguous."""

Band = Literal["at_ceiling", "below"] | None
"""Restrict the draw by leg-1 `oracle` against `pool_v3.CEILING`. At ceiling
every route already ranks the judged doc first, so there is no score left for a
better encoder to win — the only movement available is dense LOSING the doc.
Measured on the 3 pilot lanes: 93.0% of native ties are at ceiling, so an
unbanded tie draw spends most of its budget on rows that cannot move up.
`below` targets the addressable slice; `at_ceiling` draws a regression control."""

LEGB_DIR = DATA_DIR / "legb_pilot"
PILOT_LANES = ("scirgen-geo-en", "crumb-legal-qa", "antique")
"""Native v3 supply only (native_mask), spanning the idf bands that native
supply actually has and three distinct domains: scirgen (mid, geoscience —
73% of all_zero), crumb-legal-qa (mid, legal), antique (high, non-factoid QA).
Banded over the full pool, gooaq read as high-idf and got picked — but it is
v2-origin control (4 native rows of 7,079), so relabelling it teaches nothing
about the shipped dataset. Native supply has no usable low-idf lane."""

SPARSE_CFG = EmbeddingConfig(name="sparse_base", model_id="Qdrant/bm25", kind="sparse")
"""Identical to leg-1's sparse leg — only the dense encoder is under test, so
BM25+IDF stays fixed and the fusion leg differs solely through its dense half."""

CERTIFIABLE = ("decisive", "genuine_tie")
"""A row is only evidence about a *label* when both legs certified one; a row
that was never supply in leg-1 cannot show a label became stack-specific."""


def e5_dense_cfg() -> EmbeddingConfig:
    """multilingual-e5-large: in fastembed, MIT, the popular open step up from
    bge-small. e5 needs its role prefixes or retrieval collapses (measured:
    fastembed does not add them).

    `parallel` deliberately left at its None default: e5-large is 2.24GB
    (bge-small is 0.067GB), and fastembed's `parallel=N` loads N FULL model
    copies in N worker processes — `parallel=4` here means ~9GB of model
    weights alone. None uses onnxruntime's own intra-op threading against the
    one already-loaded model instead."""
    return EmbeddingConfig(
        name="dense_legb",
        model_id="intfloat/multilingual-e5-large",
        kind="dense",
        size=1024,
        distance=Distance.COSINE,
        query_prompt="query: ",
        doc_prompt="passage: ",
    )


QWEN_MODEL_ID = "openrouter/qwen/qwen3-embedding-8b"
QWEN_DIM = 4096
QWEN_INSTRUCTION = (
    "Instruct: Given a web search query, retrieve relevant passages that "
    "answer the query.\nQuery: "
)
"""Qwen3-Embedding's own documented convention (its model card / usage guide):
task instruction on the query only, nothing on documents. This is the generic
default retrieval instruction — the three pilot lanes are different enough
domains (geoscience, legal, non-factoid QA) that a task-specific instruction
tuned to one would bias the comparison against the other two."""


def qwen_dense_cfg() -> EmbeddingConfig:
    """qwen/qwen3-embedding-8b via Qdrant Cloud Inference -> OpenRouter.
    Embeds server-side, so this has none of e5's local RAM/model-copy cost —
    the reason to run this leg first. Sparse stays SPARSE_CFG (local, unpaid
    round trip) — `cloud` is per-slot, so mixing is safe (see EmbeddingConfig).

    Requires `OPEN_ROUTER_API_KEY` in the environment and the Qdrant client
    constructed with `cloud_inference=True`."""
    key = os.environ.get("OPEN_ROUTER_API_KEY")
    if not key:
        raise RuntimeError(
            "OPEN_ROUTER_API_KEY is not set — add it to .env before building "
            "the Qwen3 leg-2 config."
        )
    return EmbeddingConfig(
        name="dense_legb",
        model_id=QWEN_MODEL_ID,
        kind="dense",
        size=QWEN_DIM,
        distance=Distance.COSINE,
        cloud=True,
        provider_options={"openrouter-api-key": key, "dimensions": QWEN_DIM},
        query_prompt=QWEN_INSTRUCTION,
    )


def stack_flags(merged: pd.DataFrame) -> pd.Series:
    """Per-row stack verdict from the two legs' kinds and classes. Flags on the
    class, not the route: dense_only->dense is a supply change, a within-class
    route wobble is not. `uncertifiable` unless BOTH legs certified a row."""
    certifiable = merged["kind_leg1"].isin(CERTIFIABLE) & merged["kind_legb"].isin(
        CERTIFIABLE
    )
    specific = merged["route_class_leg1"] != merged["route_class_legb"]
    return pd.Series(
        pd.NA, index=merged.index, dtype="object"
    ).mask(certifiable & specific, "stack_specific").mask(
        certifiable & ~specific, "stack_robust"
    ).fillna("uncertifiable")


class LegBPilot:
    """Owns one leg-2 encoder's pilot: index the pilot lanes into their own
    collections, relabel, and diff against leg-1. One instance per encoder."""

    def __init__(
        self,
        client: QdrantClient,
        dense_cfg: EmbeddingConfig,
        lanes: tuple[str, ...] = PILOT_LANES,
        out_dir=LEGB_DIR,
        supply: Supply = "native",
        max_workers: int | None = None,
        sample: tuple[str, int] | None = None,
        natural_only: bool = True,
        band: Band = None,
    ) -> None:
        self._client = client
        self._dense = dense_cfg
        self._lanes = lanes
        self._out_dir = out_dir
        self._supply = supply
        # (population, n): n rows PER LANE, fixed-seed so a rerun draws the
        # SAME rows. `population` matches either `shape` ("routes_differ" |
        # "all_tied" | "all_zero") or `kind` ("decisive" | "fake_tie" |
        # "genuine_tie" | "undecisive" | "all_zero") — the kind populations
        # cannot be reached through `shape` alone, and shape=="routes_differ"
        # is only ~19% decisive, so it buys ~4 useless rows per useful one.
        self._sample = sample
        self._natural_only = natural_only
        self._band = band
        self._pool = LabelledPool()
        self._labels: pd.DataFrame | None = None
        # network-bound (a cloud slot over OpenRouter) benefits from overlap;
        # local fastembed is CPU/GIL-bound and gains nothing from threads —
        # matches augmentation/config.py's own llm_workers default of 8
        self._max_workers = max_workers if max_workers is not None else (
            8 if dense_cfg.cloud else 1
        )

    def _leg1_labels(self) -> pd.DataFrame:
        """`pool.labels()` re-reads and re-merges several parquet files on
        every call; this pilot calls it once per lane plus once in `compare`,
        so cache the one merge each instance actually needs.

        Classified, not raw: `kind`/`route_class`/`oracle` are what the sample
        bands on, and `classify` is the only sanctioned way to derive them."""
        if self._labels is None:
            self._labels = self._pool.classify(self._pool.labels())
        return self._labels

    def _supply_mask(self, frame: pd.DataFrame) -> pd.Series:
        """Rows matching the chosen supply — default v3-native, the population
        the dataset is built from."""
        nat = native_mask(frame)
        if self._supply == "native":
            return nat
        if self._supply == "v2":
            return ~nat
        return pd.Series(True, index=frame.index)

    def collection(self, lane: str) -> str:
        """The isolated leg-2 collection — asserted distinct from the paid
        leg-1 one, because reusing it would overwrite labels already bought.

        Namespaced by the dense model too, not just the lane: `dense_legb` is
        one fixed-size vector slot, so switching `LegBPilot`'s config (e.g.
        e5's 1024-dim -> Qwen's 4096-dim) against an already-created
        `<lane>_legb_routes` used to no-op on `ensure_collection` (it only
        creates when the collection is absent) and fail at upload with
        Qdrant's own "expected dim: 1024, got 4096" — silently, since nothing
        here diffed the live schema. A different model now gets a different,
        untouched collection instead of colliding with the last one's."""
        slug = self._dense.model_id.rsplit("/", 1)[-1]
        name = f"{_source_name(lane)}_legb_{slug}_routes"
        assert name != _collection(lane), f"{lane}: would reuse the paid collection"
        return name

    def _selection(self, lane: str) -> pd.DataFrame:
        """leg-1's own rows for this lane — relabel the same query_ids so the
        diff is row-for-row.

        `natural_only` drops `supplemented` rows: they are scored against a
        corpus carrying constructed docs, so an encoder effect there is
        confounded with the augmentation. It also removes the augmented
        query_ids that `QuerySubset` silently discards (they are absent from
        the lane's `queries.parquet`), which is what made a 30-row draw label
        only 24. `band` and `sample` then narrow to the addressable slice.
        `n` is PER LANE — `_selection` is called once per lane."""
        labels = self._leg1_labels()
        mine = labels[(labels["dataset"] == lane) & self._supply_mask(labels)]
        if self._natural_only and "scored_against" in mine.columns:
            mine = mine[mine["scored_against"] == "natural"]
        if self._band is not None:
            at = mine["oracle"] >= CEILING
            mine = mine[at if self._band == "at_ceiling" else ~at]
        if self._sample is not None:
            population, n = self._sample
            column = "shape" if population in SHAPES else "kind"
            mine = mine[mine[column] == population]
            if len(mine) > n:
                mine = mine.sample(n, random_state=0)
            elif len(mine) < n:
                tqdm.write(
                    f"[{lane}] sample asked {n} {population} rows, drew "
                    f"{len(mine)} — that is the whole available population"
                )
        return mine[["dataset", "query_id", "query"]].astype(str)

    def plan(self) -> pd.DataFrame:
        live = {c.name for c in self._client.get_collections().collections}
        rows = []
        for lane in self._lanes:
            corpus = SnapshotDataset(_source_name(lane), path=str(DATA_DIR)).corpus()
            rows.append({
                "lane": lane,
                "to_label": len(self._selection(lane)),
                "corpus_docs": len(corpus),
                "collection": self.collection(lane),
                "indexed": self.collection(lane) in live,
            })
        return pd.DataFrame(rows)

    def _index(self, lane: str, corpus: pd.DataFrame) -> str:
        collection = self.collection(lane)
        indexer = CorpusIndexer(
            self._client, collection,
            embeddings=[self._dense, SPARSE_CFG],
            # bge-small and e5 keep separate cache files (keyed by model_id), so
            # sharing the lane namespace is safe and reuses nothing wrongly
            cache=EmbeddingCache(namespace=_source_name(lane)),
        )
        indexer.ensure_collection()
        if self._client.count(collection, exact=True).count < len(corpus):
            docs = [CorpusDocument(**r) for r in corpus.to_dict("records")]
            fresh = indexer.missing(docs)
            tqdm.write(f"[{lane}] indexing {len(fresh):,}/{len(docs):,} -> {collection}")
            indexer.upload(fresh, batch_size=64)
        return collection

    def index_and_label(
        self, lane: str | None = None, *, force: bool = False
    ) -> list[str]:
        """Index one lane's corpus into its leg-2 collection and relabel — or
        every configured lane when `lane` is omitted. Test on the cheapest
        lane first: `compare()` reads only the lanes actually present in
        `labels.parquet`, so a one-lane smoke test is never silently read as
        the full pilot's verdict. The compute+network step — user-initiated.
        Returns lanes that failed."""
        lanes = self._lanes if lane is None else (lane,)
        unknown = set(lanes) - set(self._lanes)
        if unknown:
            raise ValueError(
                f"{sorted(unknown)} not in this pilot's lanes {self._lanes}"
            )
        failed: list[str] = []
        for i, lane_idx in enumerate(lanes, 1):
            tqdm.write(
                f"=== [{i}/{len(lanes)}] {lane_idx} ({self._dense.model_id}, "
                f"max_workers={self._max_workers}) ==="
            )
            corpus = SnapshotDataset(_source_name(lane_idx), path=str(DATA_DIR)).corpus()
            collection = self._index(lane_idx, corpus)
            min_rel = LANES[lane_idx].min_relevance if lane_idx in LANES else 1
            labels = RouteLabels(
                self._selection(lane_idx), out_dir=self._out_dir,
                objective=RouterObjective(min_relevance=min_rel),
            )
            args = (self._client, collection, self._dense, SPARSE_CFG)
            source = SnapshotDataset(_source_name(lane_idx), path=str(DATA_DIR))
            try:
                out = labels.label(
                    source,
                    DenseOnlyStrategy(*args), PureRRFStrategy(*args),
                    SparseOnlyStrategy(*args),
                    dataset=lane_idx, force=force, max_workers=self._max_workers,
                )
            except ValueError as error:
                failed.append(lane_idx)
                tqdm.write(f"[{lane_idx}] SKIPPED: {error}")
                continue
            shape = out["shape"].value_counts() if not out.empty else {}
            tqdm.write(
                f"[{lane_idx}] +{len(out):,} leg-2 labels  "
                f"differ {shape.get('routes_differ', 0):,} | "
                f"tied {shape.get('all_tied', 0):,} | zero {shape.get('all_zero', 0):,}"
            )
        return failed

    def compare(self) -> pd.DataFrame:
        """Classify both legs, join row-for-row, and attach the stack flag +
        the dense-score delta. Writes `stack_flag.parquet`; also the input to
        the rescue and manipulation readouts.

        Scoped to the lanes actually IN `labels.parquet`, not to every
        configured lane — a one-lane `index_and_label` run must compare only
        that lane, never silently read the other, untested lanes as if they
        agreed or disagreed."""
        legb = pd.read_parquet(self._out_dir / "labels.parquet")
        tested = set(legb["dataset"].unique()) & set(self._lanes)
        leg1 = self._leg1_labels()
        leg1 = leg1[leg1["dataset"].isin(tested) & self._supply_mask(leg1)]

        # leg1 arrives already classified from `_leg1_labels`; classifying it
        # again collides on `depth` (both sides of `_qrels_depth`'s merge carry
        # it, so pandas suffixes it to depth_x/depth_y and the lookup raises).
        # legb is raw from its own labels.parquet, so it still needs the pass.
        a = leg1.astype({"query_id": str})
        b = self._pool.classify(legb).astype({"query_id": str})
        keep = ["dataset", "query_id", "kind", "route_class", "route", "margin",
                "score_dense_only"]
        merged = a[keep].merge(
            b[keep], on=["dataset", "query_id"], suffixes=("_leg1", "_legb")
        )
        merged["flag"] = stack_flags(merged)
        merged["dense_delta"] = (
            merged["score_dense_only_legb"] - merged["score_dense_only_leg1"]
        )
        merged["legb_model"] = self._dense.model_id
        self._out_dir.mkdir(parents=True, exist_ok=True)
        merged.to_parquet(self._out_dir / "stack_flag.parquet", index=False)
        return merged

    def readout(self, merged: pd.DataFrame) -> None:
        """Print the two decisions this pilot exists to inform."""
        tested = sorted(merged["dataset"].unique())
        untested = [lane for lane in self._lanes if lane not in tested]
        print(f"COVERAGE  tested: {tested}")
        if untested:
            print(f"          NOT YET TESTED: {untested} — this is a partial "
                  f"readout, not the full pilot's verdict.\n")
        else:
            print()
        az = merged[merged["kind_leg1"] == "all_zero"]
        rescued = az[az["kind_legb"] != "all_zero"]
        print(f"RESCUE  all_zero rows: {len(az):,}  ->  moved off zero: {len(rescued):,} "
              f"({len(rescued) / max(len(az), 1):.1%})")
        if len(rescued):
            print(rescued["kind_legb"].value_counts().to_string())
        certifiable = merged[merged["flag"] != "uncertifiable"]
        print(f"\nFLAG    certifiable rows: {len(certifiable):,}")
        print(certifiable["flag"].value_counts().to_string())
        certified_class = merged[merged["route_class_legb"].isin(CLASSES)]
        print("\nDELTA   dense_delta over rows certified by leg-2 "
              "(centred on 0 => encoder did nothing, a bug not a null):")
        print(certified_class["dense_delta"].describe()[["mean", "50%", "max"]].to_string())
