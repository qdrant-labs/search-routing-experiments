"""Route labels for the composition's rows — the golden set.

`TargetComposition` decides *which* queries the dataset contains; this decides
*which route* each of them should take. One row per labelled (dataset,
query_id), appended dataset by dataset as each source's corpus and qrels become
available. Rows whose dataset has no usable corpus stay unlabelled and are
reported by `coverage()` — never silently dropped, because the gap between
"labelled" and "in the composition" is the number that decides how many
trainable rows a 50K pool actually yields.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Literal

import numpy as np
import pandas as pd

from hybrid_search_rrf_dataset.fusion import (
    SERVING_COST,
    TIE_TOLERANCE,
    FusionStrategy,
    StrategyName,
    derive_route,
)
from hybrid_search_rrf_dataset.golden import GoldenRoutingBuilder
from hybrid_search_rrf_dataset.lanes import LANES
from hybrid_search_rrf_dataset.objective import Objective, RouterObjective
from hybrid_search_rrf_dataset.qrels import QrelStore
from hybrid_search_rrf_dataset.retrieval import (
    QuerySubset,
    QuerySupplement,
    RetrievalDataset,
)

if TYPE_CHECKING:
    from augmentation.config import AugmentationPaths

DEFAULT_OUT_DIR = Path(__file__).resolve().parent.parent / "data" / "route_labels"

Generation = Literal["floor_based", "cell_based"]

ROUTES_DIFFER = "routes_differ"
ALL_TIED = "all_tied"
ALL_ZERO = "all_zero"


def outcome_shape(scores: dict[str, float], tolerance: float = TIE_TOLERANCE) -> str:
    """Classify a query by whether its routes disagree.

    `ROUTES_DIFFER` is the only shape that teaches the router about
    dense-vs-sparse. `ALL_TIED` says any route works, so the cheapest is
    served — signal for the speed requirement, not the quality one. `ALL_ZERO`
    means no route surfaced anything relevant, so **no valid label exists**
    and `route` is stored null (SPEC d41b).
    """
    if not scores:
        return ALL_ZERO
    values = list(scores.values())
    if max(values) <= tolerance:
        return ALL_ZERO
    if max(values) - min(values) <= tolerance:
        return ALL_TIED
    return ROUTES_DIFFER


def route_label(scores: dict[str, float]) -> str | None:
    """The stored label for one row: `derive_route` over the score vector,
    null when no route retrieved anything (SPEC d41 — the scores are the
    record, this is the derived serving decision)."""
    if outcome_shape(scores) == ALL_ZERO:
        return None
    return str(derive_route(scores))


def _augmented_rows(
    pool: pd.DataFrame, wanted: pd.DataFrame, home_lane: str, generation: Generation,
) -> pd.DataFrame:
    """This lane's augmentation-pool rows admissible for evaluation: `floor_based`
    unconditionally once ungated, `cell_based` only once `wanted` already admits
    them (SPEC d61). A gated row has no human-audited credit yet regardless of
    generation — the d42h gate applies to both (2026-08-10 fix: `floor_based`
    originally skipped this check, leaking 240 unaudited rows)."""
    # lazy: composition/__init__ pulls in hybrid_search_rrf_dataset.router, which
    # imports this module — a top-level import here would be circular (same
    # reason augmentation/supply.py's lane_dirs() defers its own).
    from composition.cells import CELLS_BY_NAME

    rows = pool[pool["home_lane"] == home_lane]
    rows = rows[rows["credit_gate"].fillna("none") == "none"]
    if generation == "floor_based":
        return rows[~rows["floor"].isin(CELLS_BY_NAME)]
    admitted = set(wanted["query_id"].astype(str))
    return rows[rows["query_id"].astype(str).isin(admitted)]


class RouteLabels:
    """Builds and owns `data/route_labels/labels.parquet`."""

    CARRIED = ["slice", "checkable", "label_lane", "cell", "stage", "route"]
    """Selection columns copied onto every label where the selection has them,
    so outcome shapes can be read per group rather than only in aggregate."""

    @property
    def carried(self) -> list[str]:
        """The carried columns this selection actually has."""
        return [column for column in self.CARRIED if column in self.selection.columns]

    def __init__(
        self,
        selection: pd.DataFrame,
        out_dir: Path | None = None,
        objective: Objective | None = None,
        *,
        augmentation_paths: AugmentationPaths | None = None,
    ) -> None:
        from augmentation.config import AugmentationPaths  # lazy: see _augmented_rows

        self.selection = selection
        self.objective = objective or RouterObjective()
        self._out_dir = out_dir if out_dir is not None else DEFAULT_OUT_DIR
        self._aug_paths = augmentation_paths or AugmentationPaths()

    @property
    def labels_path(self) -> Path:
        return self._out_dir / "labels.parquet"

    def rows_for(self, dataset: str) -> pd.DataFrame:
        return self.selection[self.selection["dataset"] == dataset]

    def load(self) -> pd.DataFrame:
        if not self.labels_path.exists():
            return pd.DataFrame(columns=["dataset", "query_id", "route"])
        return pd.read_parquet(self.labels_path)

    def label(
        self,
        source: RetrievalDataset,
        dense: FusionStrategy,
        hybrid: FusionStrategy,
        sparse: FusionStrategy,
        *,
        dataset: str | None = None,
        qrels: QrelStore | None = None,
        force: bool = False,
        generation: Generation = "cell_based",
    ) -> pd.DataFrame:
        """Label this dataset's query_ids not already in the artifact, and
        append. Already-labelled query_ids (natural or augmented) are never
        rescored — `force` is the only way to redo one that already has a
        row. `dataset` names the selection's key for `source` when the two
        differ — the composition calls BEIR NFCorpus `beir-nfcorpus` while
        the retrieval class calls it `nfcorpus`.
        """
        key = dataset or source.name
        wanted = self.rows_for(key)
        if wanted.empty:
            raise ValueError(f"No selection rows for dataset {key!r}.")

        existing = self.load()
        already = (
            set()
            if force
            else set(
                existing.loc[existing.get("dataset") == key, "query_id"].astype(str)
            )
        )

        subset = QuerySubset(source, wanted["query_id"])
        excluded_df = subset.excluded()
        exclusions = (
            {
                str(query_id): frozenset(group["doc_id"].astype(str))
                for query_id, group in excluded_df.groupby("query_id")
            }
            if not excluded_df.empty
            else None
        )

        from augmentation.pool import GeneratedPool  # lazy: see _augmented_rows
        from augmentation.qrels import AugmentationQrels

        aug_rows = _augmented_rows(
            GeneratedPool(self._aug_paths).load(), wanted, key, generation
        )
        if aug_rows.empty:
            eval_dataset, eval_qrels = subset, qrels
        else:
            aug_ids = set(aug_rows["query_id"].astype(str))
            matched = AugmentationQrels(self._aug_paths).load()
            matched = matched[matched["query_id"].astype(str).isin(aug_ids)]
            eval_dataset = QuerySupplement(
                subset,
                aug_rows[["query_id", "query", "provenance"]].rename(
                    columns={"query": "text"}
                ),
                matched[["query_id", "doc_id", "relevance"]],
            )
            eval_qrels = QrelStore.concat([
                qrels or QrelStore.from_dataset(source),
                QrelStore(matched.assign(dataset=key)[QrelStore.COLUMNS]),
            ])

        missing = set(eval_dataset.queries()["query_id"].astype(str)) - already
        if not missing:
            return existing.iloc[:0]
        if already:
            eval_dataset = QuerySubset(eval_dataset, missing)

        rows = GoldenRoutingBuilder(
            dense, hybrid, sparse, objective=self.objective, excluded=exclusions
        ).build(eval_dataset, qrels=eval_qrels)

        labelled = pd.DataFrame(
            [
                {
                    "dataset": key,
                    "query_id": row.query_id,
                    "query": row.query,
                    "route": route_label(row.route_scores),
                    "score": row.metric,
                    **{f"score_{name}": value for name, value in row.route_scores.items()},
                    "shape": outcome_shape(row.route_scores),
                    "metric_name": row.metric_name,
                    "min_relevance": self.objective.min_relevance,
                    "provenance": row.provenance,
                }
                for row in rows
            ]
        )
        if labelled.empty:
            raise ValueError(
                f"{key!r}: no rows produced for the {len(missing):,} new "
                f"query_ids — check that the qrels cover them."
            )
        # one label per (dataset, query_id): a query in several cells appears
        # once in wanted per cell, so dedup before the merge or it fans out.
        # Per-cell readouts join labels back to the selection on query_id.
        carry = (
            wanted[["query_id", *self.carried]]
            .astype({"query_id": str})
            .drop_duplicates("query_id")
        )
        # a carried selection column can share a name with a label column
        # (cell_selection's planned `route` vs the labelled `route`): keep the
        # label authoritative and suffix the selection's copy.
        clash = {c: f"{c}_selected" for c in self.carried if c in labelled.columns}
        labelled = labelled.merge(
            carry.rename(columns=clash), on="query_id", how="left"
        )

        # keep every row this call didn't touch — force only replaces the
        # query_ids it actually rescored, never the rest of the dataset
        stale = (existing.get("dataset") == key) & (
            existing["query_id"].astype(str).isin(missing)
        )
        merged = pd.concat([existing[~stale], labelled], ignore_index=True)
        self.labels_path.parent.mkdir(parents=True, exist_ok=True)
        merged.to_parquet(self.labels_path, index=False)
        return labelled

    def rederive(self) -> pd.DataFrame:
        """Recompute `route` and `shape` for every stored row from the score
        columns and rewrite the artifact — the d41 migration, and the standing
        repair after any rule change (a cost-order flip, a tolerance change).
        A re-derivation, never a re-run: no retrieval is involved.
        """
        labels = self.load()
        if labels.empty:
            return labels
        names = [
            c.removeprefix("score_")
            for c in labels.columns
            if c.startswith("score_")
        ]

        def derived(row: pd.Series) -> pd.Series:
            scores = {name: row[f"score_{name}"] for name in names}
            return pd.Series(
                {"route": route_label(scores), "shape": outcome_shape(scores)}
            )

        labels[["route", "shape"]] = labels.apply(derived, axis=1)
        labels.to_parquet(self.labels_path, index=False)
        return labels

    def _scored(self) -> tuple[pd.DataFrame, list[str]]:
        """Stored labels plus two derived columns: `oracle` (best of the three
        route scores) and `margin` (oracle minus runner-up). Returns the frame
        and the score column names."""
        labels = self.load()
        score_cols = [c for c in labels.columns if c.startswith("score_")]
        if labels.empty or not score_cols:
            raise ValueError("No labelled rows on disk — nothing to read out.")
        ordered = np.sort(labels[score_cols].to_numpy(), axis=1)
        labels = labels.assign(
            oracle=ordered[:, -1], margin=ordered[:, -1] - ordered[:, -2]
        )
        return labels, score_cols

    def headroom_decomposition(self) -> pd.DataFrame:
        """The headline three-level readout (SPEC d44a): what a single global
        route earns, what picking the best route per collection earns, and
        what a perfect per-query choice (the oracle) earns — each level's gain
        relative to the previous one. The two gaps are the value of
        collection-level and per-query routing respectively.
        """
        labels, score_cols = self._scored()
        global_constant = max(labels[c].mean() for c in score_cols)
        global_name = max(score_cols, key=lambda c: labels[c].mean())
        lane_best = labels.groupby("dataset")[score_cols].mean().max(axis=1)
        lane_n = labels.groupby("dataset").size()
        per_collection = float((lane_best * lane_n).sum() / lane_n.sum())
        oracle = float(labels["oracle"].mean())

        levels = pd.DataFrame(
            {
                "level": [
                    f"one global constant ({global_name.removeprefix('score_')})",
                    "best constant per collection",
                    "per-query oracle (ceiling)",
                ],
                "score": [float(global_constant), per_collection, oracle],
            }
        )
        levels["gain_vs_previous_pct"] = (
            levels["score"].pct_change().mul(100).round(1)
        )
        return levels

    def headroom(self) -> pd.DataFrame:
        """Per-lane routing-value ceiling (SPEC d44a), POOLED row last.

        `headroom` = mean oracle minus the lane's best constant route — what a
        perfect router would add over never routing at all. A ceiling, not an
        achievement; the mandatory caveats live beside the table wherever it
        is shown.
        """
        labels, score_cols = self._scored()
        decisive = self.objective.decisive_margin

        def one(group: pd.DataFrame, name: str) -> dict[str, object]:
            means = {c: group[c].mean() for c in score_cols}
            best = max(means, key=means.get)  # type: ignore[arg-type]
            constant = float(means[best])
            oracle = float(group["oracle"].mean())
            gap = oracle - constant
            return {
                "dataset": name,
                "labelled": len(group),
                "oracle": round(oracle, 3),
                "best_constant": best.removeprefix("score_"),
                "constant": round(constant, 3),
                "headroom": round(gap, 3),
                "headroom_pct": round(100 * gap / constant, 1) if constant else 0.0,
                "decisive_share": round((group["margin"] >= decisive).mean(), 3),
                "all_zero_share": round((group["oracle"] <= 0).mean(), 3),
            }

        rows = [one(group, str(name)) for name, group in labels.groupby("dataset")]
        rows.append(one(labels, "POOLED"))
        return pd.DataFrame(rows)

    def decisive_winners(self) -> pd.DataFrame:
        """Winner counts over decisive rows only (winner hit rank 1, runner-up
        missed — SPEC d41d), per lane plus a POOLED row. One-sided lanes show
        why decisive share alone is not headroom: a single route can win every
        decisive row and leave the constant nothing to improve on.
        """
        labels, score_cols = self._scored()
        rows = labels[labels["margin"] >= self.objective.decisive_margin]
        winners = (
            rows[score_cols]
            .idxmax(axis=1)
            .str.removeprefix("score_")
            .rename("winner")
        )
        table = pd.crosstab(rows["dataset"], winners)
        table = table.reindex(
            columns=sorted(c.removeprefix("score_") for c in score_cols),
            fill_value=0,
        )
        table.loc["POOLED"] = table.sum()
        return table.reset_index().rename(columns={"index": "dataset"})

    def coverage(self) -> pd.DataFrame:
        """Per-dataset progress: selected rows vs labelled, and the shape split.

        Every column counts SELECTED rows only, via a key join —
        labels.parquet keeps rows from earlier, larger selections (measured
        scores are never deleted when a row leaves the selection), so a bare
        per-dataset label count overstates progress and once drove
        `unlabelled` negative. `qrels_ready` counts selected rows whose lane
        has qrels on disk but no label yet — the corpus-pending state between
        pass 1 and pass 2 (SPEC d39b). A row is unlabelled because its lane's
        qrels are not fetched, its corpus is not indexed, or the source never
        judged it.
        """
        labels = self.load()
        shapes = (
            labels[["dataset", "query_id", "shape"]]
            .drop_duplicates(["dataset", "query_id"])
            if "shape" in labels.columns and not labels.empty
            else pd.DataFrame(columns=["dataset", "query_id", "shape"])
        )
        merged = self.selection[["dataset", "query_id"]].merge(
            shapes, on=["dataset", "query_id"], how="left"
        )
        rows = []
        for dataset, picked in merged.groupby("dataset", sort=False):
            done = int(picked["shape"].notna().sum())
            counts = picked["shape"].value_counts()
            rows.append(
                {
                    "dataset": dataset,
                    "selected": len(picked),
                    "labelled": done,
                    "qrels_ready": self._qrels_ready(dataset, done),
                    "unlabelled": len(picked) - done,
                    ROUTES_DIFFER: int(counts.get(ROUTES_DIFFER, 0)),
                    ALL_TIED: int(counts.get(ALL_TIED, 0)),
                    ALL_ZERO: int(counts.get(ALL_ZERO, 0)),
                }
            )
        return pd.DataFrame(rows).sort_values(
            ["labelled", "selected"], ascending=False, ignore_index=True
        )

    def acceptability(self, tolerance: float | None = None) -> AcceptabilityLabels:
        """The d60 view over the stored labels."""
        return AcceptabilityLabels(self.load(), tolerance=tolerance)

    def _qrels_ready(self, dataset: str, labelled: int) -> int:
        """Selected rows joinable against the lane's on-disk qrels, minus the
        already-labelled ones. 0 when the lane is unknown or not yet fetched."""
        lane = LANES.get(dataset)
        if lane is None:
            return 0
        qrels_path = self._out_dir.parent / lane.source.name / "qrels.parquet"
        if not qrels_path.exists():
            return 0
        judged = set(
            pd.read_parquet(qrels_path, columns=["query_id"])["query_id"].astype(str)
        )
        selected_ids = set(
            self.selection.loc[
                self.selection["dataset"] == dataset, "query_id"
            ].astype(str)
        )
        return max(len(selected_ids & judged) - labelled, 0)


class AcceptabilityLabels:
    """The d60 view over a labelled frame: per-route `ok_*` booleans plus the
    cost-aware `serve` decision, derived from the stored score vector at read
    time and never materialized. Default tolerance is hit parity — the
    objective's own `ndcg_weight`, the widest gap that cannot involve a
    top-1 flip."""

    def __init__(
        self, labels: pd.DataFrame, tolerance: float | None = None
    ) -> None:
        self.labels = labels
        self.tolerance = (
            RouterObjective().ndcg_weight if tolerance is None else tolerance
        )

    def frame(self) -> pd.DataFrame:
        """The input frame plus `ok_<route>` (nullable boolean) and `serve`;
        all-zero rows carry nulls in every added column (d41 upheld)."""
        out = self.labels.copy()
        score_cols = [c for c in out.columns if c.startswith("score_")]
        routes = [c.removeprefix("score_") for c in score_cols]
        scores = out[score_cols].to_numpy(dtype=np.float64)
        oracle = scores.max(axis=1)
        answerable = oracle > TIE_TOLERANCE
        # floored at TIE_TOLERANCE so tolerance=0 means "exact ties", exactly
        # as derive_route counts them — the must-pass reproduction property.
        effective = max(self.tolerance, TIE_TOLERANCE)
        ok = scores >= (oracle - effective)[:, None]

        for i, route in enumerate(routes):
            column = pd.array(ok[:, i], dtype="boolean")
            column[~answerable] = pd.NA
            out[f"ok_{route}"] = column

        cost = np.array([SERVING_COST[StrategyName(r)] for r in routes])
        cheapest_ok = np.where(ok, cost[None, :], np.inf).argmin(axis=1)
        out["serve"] = pd.Series(
            [routes[i] for i in cheapest_ok], index=out.index
        ).where(answerable)
        return out
