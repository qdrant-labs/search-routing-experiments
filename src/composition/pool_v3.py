"""The labelled pool, classified and strata-attached — the ONE frame every v3
consumer (composer, ablation, discriminator, sensitivity) reads instead of
re-deriving. Route classes at two nested bars (certified margin vs margin-0),
qrels-depth tie taxonomy, diversity strata, near-dup clusters and the frozen
eval reserve all live here."""

from __future__ import annotations

import numpy as np
import pandas as pd

from composition.cells import CELLS
from composition.cells_v3 import CELLS_V3
from composition.compose import DEFAULT_OUT_DIR
from composition.floors import CORRUPTION_SPANS, read_catalog, with_derived
from composition.recipe import Recipe

DATA = DEFAULT_OUT_DIR.parent

REUSED = "reused"
"""Selection stage whose rows were drawn BECAUSE they already won a route, so
their decisive rate is conditioned on the outcome and estimates nothing."""

SCORES = ["score_dense_only", "score_pure_rrf", "score_sparse_only"]
ROUTE_TO_CLASS = {"dense_only": "dense", "sparse_only": "sparse", "pure_rrf": "hybrid"}
CLASSES = ("dense", "sparse", "hybrid")
CORPUS_AXES = ("corpus_idf", "corpus_oov", "corpus_pmi")
"""Corpus-relative diversity, one MARGINAL axis each — never crossed with each
other, with cell, or with lane."""
CEILING = 0.999
"""Reporting bar only: splits each tie kind into at-ceiling (everyone found a
judged doc at rank 1) vs all-routes-missed. Classification is depth-based."""
TOL = 1e-9


def assign_classes(oracle, runner, low, winner, depth, recipe: Recipe):
    """Pure policy: score vectors + qrels depth -> (kind, route_class) arrays.
    kind in {decisive, genuine_tie, fake_tie, all_zero, undecisive};
    route_class in {dense, sparse, hybrid, ''}. No I/O — unit-tested directly."""
    zero = oracle <= TOL
    tied = (~zero) & (oracle - low <= TOL)
    differ = (~zero) & (~tied)
    decisive = differ & (oracle - runner >= recipe.class_margin)
    # a shallow tie is unreadable at ANY score: at ceiling everyone found the
    # one judged doc, below it everyone missed it — both are the qrels-depth
    # artifact. The old rule required the ceiling too, which misfiled 491
    # all-routes-missed rows as genuine hybrid supply.
    fake_tie = tied & (depth < recipe.genuine_tie_depth)
    genuine_tie = tied & ~fake_tie

    kind = np.full(len(oracle), "undecisive", dtype=object)
    kind[zero] = "all_zero"
    kind[fake_tie] = "fake_tie"
    kind[genuine_tie] = "genuine_tie"
    kind[decisive] = "decisive"

    cls = np.full(len(oracle), "", dtype=object)
    dec_class = np.array([ROUTE_TO_CLASS.get(w, "") for w in winner], dtype=object)
    cls[decisive] = dec_class[decisive]
    cls[genuine_tie] = "hybrid"
    return kind, cls


def native_mask(pool: pd.DataFrame) -> pd.Series:
    """Which rows are v3 supply; a frame without the column is all supply."""
    if "native" in pool.columns:
        return pool["native"].astype(bool)
    return pd.Series(True, index=pool.index)


class LabelledPool:
    """Owns loading, classification, strata attachment, clustering and the
    eval reserve over the labelled rows; consumers read, never re-derive."""

    def __init__(self, recipe: Recipe | None = None, data_dir=None) -> None:
        self._recipe = recipe if recipe is not None else Recipe()
        self._data = data_dir if data_dir is not None else DATA
        self._frame: pd.DataFrame | None = None

    @property
    def recipe(self) -> Recipe:
        return self._recipe

    @property
    def v3_catalog_path(self):
        return self._data / "v3" / "catalog_v3.parquet"

    # ------------------------------------------------------------- loading ---
    def labels(self) -> pd.DataFrame:
        """The re-derived pool (v2's labels rescored from the oracle caches at
        each lane's current min_relevance) PLUS the additive v3 labels PLUS the
        synthetic-rung labels, aligned on the base's columns. New (dataset,
        query_id) pairs are disjoint, so the dedup only guards a re-label."""
        rederived = self._data / "v3" / "labels_rederived.parquet"
        base_path = (
            rederived if rederived.exists()
            else self._data / "route_labels" / "labels.parquet"
        )
        base = pd.read_parquet(base_path).astype({"query_id": str})
        if "scored_against" not in base.columns:
            base["scored_against"] = "natural"
        # the provenance boundary (decided 2026-08-20): re-scored v2 rows
        # measure yields and priors but are never v3 SUPPLY — the v3 dataset
        # is composed only from material its own pipeline acquired
        base["native"] = False
        frames = [base]
        for path, scored_against in (
            (self._data / "v3" / "labels.parquet", "natural"),
            (self._data / "v3" / "synthetic" / "labels.parquet", "supplemented"),
        ):
            if not path.exists():
                continue
            extra = pd.read_parquet(path).astype({"query_id": str})
            if "scored_against" not in extra.columns:
                extra["scored_against"] = scored_against
            extra = extra.reindex(columns=base.columns)
            extra["native"] = True
            frames.append(extra)
        combined = pd.concat(frames, ignore_index=True)
        combined = combined.drop_duplicates(["dataset", "query_id"], keep="first")
        # checkable is a per-LANE registry fact; additive label frames arrive
        # without it, so backfill from lane siblings that know — a lane no row
        # can vouch for stays False, never assumed labelable
        known = (
            combined.dropna(subset=["checkable"])
            .drop_duplicates("dataset")
            .set_index("dataset")["checkable"]
        )
        combined["checkable"] = (
            combined["checkable"]
            .fillna(combined["dataset"].map(known))
            .fillna(False)
            .astype(bool)
        )
        return combined

    def _qrels_depth(self, labels: pd.DataFrame) -> np.ndarray:
        """Judged-relevant doc count per row at its own min_relevance — the
        fake-tie discriminator. Lane qrels first; rows a lane file does not
        know (the synthetic rung's constructed keys) fall back to
        augmentation qrels under the identical relevance filter."""
        out = []
        for dataset, grp in labels.groupby("dataset"):
            qrels_path = self._data / str(dataset) / "qrels.parquet"
            if not qrels_path.exists():
                continue
            qrels = pd.read_parquet(qrels_path).astype({"query_id": str})
            want = grp[["query_id", "min_relevance"]].drop_duplicates()
            hit = qrels.merge(want, on="query_id", how="inner")
            depth = (
                hit[hit["relevance"] >= hit["min_relevance"]]
                .groupby("query_id")
                .size()
                .rename("depth")
            )
            joined = grp[["query_id"]].merge(depth, on="query_id", how="left")
            joined["dataset"] = dataset
            out.append(joined)
        depths = pd.concat(out) if out else pd.DataFrame(
            columns=["dataset", "query_id", "depth"]
        )
        merged = labels.merge(depths, on=["dataset", "query_id"], how="left")
        missing = merged["depth"].isna()
        if missing.any():
            merged.loc[missing, "depth"] = self._augmentation_depth(
                merged.loc[missing, ["query_id", "min_relevance"]]
            )
        return merged["depth"].fillna(0).to_numpy()

    def _augmentation_depth(self, want: pd.DataFrame) -> np.ndarray:
        """Depth from `data/augmentation/qrels.parquet` for rows outside every
        lane file — query_id is globally unique there, so no dataset key."""
        path = self._data / "augmentation" / "qrels.parquet"
        if not path.exists():
            return np.full(len(want), np.nan)
        qrels = pd.read_parquet(path).astype({"query_id": str})
        hit = qrels.merge(want.drop_duplicates(), on="query_id", how="inner")
        depth = (
            hit[hit["relevance"] >= hit["min_relevance"]]
            .groupby("query_id")
            .size()
        )
        return want["query_id"].map(depth).to_numpy(dtype=float)

    # -------------------------------------------------------- classification ---
    def classify(self, labels: pd.DataFrame) -> pd.DataFrame:
        """Attach oracle/margin/depth and BOTH tiers' route classes:
        `route_class` at the certified margin, `route_class_any` at margin 0.
        The winner is the argmax either way, so a certified row's two classes
        agree — which makes the tiers nested rather than parallel."""
        ordered = np.sort(labels[SCORES].to_numpy(), axis=1)
        oracle, runner, low = ordered[:, -1], ordered[:, -2], ordered[:, 0]
        winner = labels[SCORES].idxmax(axis=1).str.replace("score_", "").to_numpy()
        depth = self._qrels_depth(labels)
        r = self._recipe
        kind, cls = assign_classes(oracle, runner, low, winner, depth, r)
        _, cls_any = assign_classes(
            oracle, runner, low, winner, depth,
            Recipe(class_margin=0.0, genuine_tie_depth=r.genuine_tie_depth),
        )
        return labels.assign(
            oracle=oracle, margin=oracle - runner, depth=depth,
            winner=winner, kind=kind, route_class=cls,
            route_class_any=cls_any,
            certified=np.isin(cls, CLASSES),
            is_decisive=np.isin(cls, CLASSES) & (kind != "genuine_tie"),
            is_hybrid=cls == "hybrid",
            is_waste=np.isin(kind, ["fake_tie", "all_zero"]),
        )

    # ---------------------------------------------------------------- strata ---
    def active_cells(self, catalog: pd.DataFrame | None = None):
        """v2 cells, plus the v3 cells when the v3 catalog (their columns)
        exists. Given a catalog, cells banding on a column it does not carry
        are dropped — a not-yet-backfilled column is a KeyError in
        `cell.select`, and the cell is BLOCKED until the build runs, not
        silently empty."""
        cells = (*CELLS, *CELLS_V3) if self.v3_catalog_path.exists() else CELLS
        if catalog is None:
            return cells
        return tuple(
            cell for cell in cells
            if all(band.column in catalog.columns for band in cell.bands)
        )

    def attach_strata(self, pool: pd.DataFrame) -> pd.DataFrame:
        """cell membership (multi) and corruption degree from the catalog,
        corpus axes from query_corpus_stats. Lane is `dataset`."""
        catalog_path = (
            self.v3_catalog_path if self.v3_catalog_path.exists()
            else self._data / "feature_table" / "catalog.parquet"
        )
        catalog = with_derived(read_catalog(catalog_path))
        idx = catalog.set_index(["dataset", "query_id"]).index
        per_row: list[set] = [set() for _ in range(len(catalog))]
        for cell in self.active_cells(catalog):
            for i in np.nonzero(cell.select(catalog).to_numpy())[0]:
                per_row[i].add(cell.name)
        cell_lookup = dict(zip(idx, per_row))

        # Corruption degree reads the catalog's own derived total — the
        # TOKENIZER-inclusive pass the v3 damage cells band on. Rows the
        # catalog does not carry are 'unknown', not 'clean'.
        spans = dict(zip(idx, catalog[CORRUPTION_SPANS]))
        pool_idx = list(zip(pool["dataset"], pool["query_id"]))
        total = np.array([spans.get(k, np.nan) for k in pool_idx], dtype=float)
        degree = pd.cut(total, [-np.inf, 0.5, 1.5, np.inf],
                        labels=["clean", "light", "heavy"]).astype(object)
        pool = pool.assign(
            cells=[frozenset(cell_lookup.get(k, set())) for k in pool_idx],
            corruption_spans=total,
            corruption_degree=pd.Series(degree).fillna("unknown").to_numpy(),
        )
        return self._attach_corpus_strata(pool)

    def _attach_corpus_strata(self, pool: pd.DataFrame) -> pd.DataFrame:
        """The three marginal corpus axes from query_corpus_stats.parquet:
        IDF quartile band (edges measured off the file, never hand numbers),
        OOV presence, PMI sentinel. Rows the file does not carry -> 'unknown'."""
        qcs_path = self._data / "route_labels" / "query_corpus_stats.parquet"
        if not qcs_path.exists():
            return pool.assign(**dict.fromkeys(CORPUS_AXES, "unknown"))
        qcs = (
            pd.read_parquet(qcs_path)
            .astype({"query_id": str})
            .drop_duplicates(["dataset", "query_id"])
            .set_index(["dataset", "query_id"])
        )
        key = pd.MultiIndex.from_arrays([pool["dataset"], pool["query_id"]])
        absent = ~key.isin(qcs.index)
        rows = qcs.reindex(key)
        p25, p75 = qcs["avg_idf"].quantile([0.25, 0.75])
        idf, oov, pmi = (
            rows[c].to_numpy() for c in ("avg_idf", "oov_share", "min_pmi")
        )
        return pool.assign(
            corpus_idf=np.where(
                absent | np.isnan(idf), "unknown",
                np.where(idf < p25, "low_idf",
                         np.where(idf >= p75, "high_idf", "mid_idf")),
            ),
            corpus_oov=np.where(
                absent | np.isnan(oov), "unknown",
                np.where(oov > 0, "has_oov", "in_vocab"),
            ),
            # -1.0 exactly is PMIBank's "never co-occurs" sentinel — measured.
            # NaN is the pair never being measurable at all; binning the two
            # together would call 698 unmeasured rows a structural miss.
            corpus_pmi=np.where(
                absent, "unknown",
                np.where(np.isnan(pmi), "unmeasured",
                         np.where(pmi == -1.0, "never_co_occurs", "co_occurring")),
            ),
        )

    # ------------------------------------------------------- clusters/reserve ---
    def cluster_ids(self, pool: pd.DataFrame) -> pd.Series:
        """Each row's near-dup cluster id: `dup_clusters.parquet`'s real id
        where known, else a unique per-row placeholder — rows the file does
        not cover have NO KNOWN relationship, which must never be conflated
        with knowing they are singletons."""
        path = self._data / "route_labels" / "dup_clusters.parquet"
        if not path.exists():
            return pd.Series(pool.index.astype(str), index=pool.index)
        dup = pd.read_parquet(path).astype({"query_id": str})
        lookup = {
            (d, q): c for d, q, c in
            zip(dup["dataset"], dup["query_id"], dup["cluster_id"])
        }
        keys = list(zip(pool["dataset"], pool["query_id"]))
        return pd.Series(
            [lookup.get(k, f"solo:{k[0]}:{k[1]}") for k in keys],
            index=pool.index,
        )

    def eval_reserve(self, pool: pd.DataFrame) -> pd.DataFrame:
        """The frozen ablation eval set, carved BEFORE any selection: a seeded
        stratified draw over (lane x certified route class), certified rows
        only because every proof runs on the certified tier."""
        certified = pool[pool["certified"] & native_mask(pool)]
        return certified.groupby(
            ["dataset", "route_class"], group_keys=False
        ).sample(
            frac=self._recipe.eval_reserve_frac, random_state=self._recipe.seed
        )

    def reserve_exclusion_keys(
        self, pool: pd.DataFrame, reserve: pd.DataFrame
    ) -> pd.Index:
        """Every row to exclude from selection: the reserve itself PLUS any
        row — certified or not — sharing a near-dup cluster with a reserved
        row; a cluster-mate left in the training pool is a train/eval leak
        regardless of its own certification."""
        cluster = self.cluster_ids(pool)
        reserved_clusters = set(cluster.loc[reserve.index])
        return pool.index[cluster.isin(reserved_clusters)]

    # ------------------------------------------------------------------ API ---
    def frame(self) -> pd.DataFrame:
        """The classified, strata-attached pool — computed once, then cached."""
        if self._frame is None:
            self._frame = self.attach_strata(self.classify(self.labels()))
        return self._frame

    def refresh(self) -> LabelledPool:
        """Drop the cached frame so the next read sees labels that landed
        after it was built — a mid-run recompose on a stale cache is a no-op
        that silently reports nothing changed."""
        self._frame = None
        return self

    def reserve(self) -> pd.DataFrame:
        return self.eval_reserve(self.frame())

    def selectable(self) -> pd.DataFrame:
        pool = self.frame()
        kept = pool.drop(index=self.reserve_exclusion_keys(pool, self.reserve()))
        return kept[native_mask(kept)]
