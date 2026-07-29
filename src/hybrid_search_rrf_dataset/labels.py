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

import pandas as pd

from hybrid_search_rrf_dataset.fusion import FusionStrategy
from hybrid_search_rrf_dataset.golden import GoldenRoutingBuilder
from hybrid_search_rrf_dataset.lanes import LANES
from hybrid_search_rrf_dataset.objective import Objective, RouterObjective
from hybrid_search_rrf_dataset.qrels import QrelStore
from hybrid_search_rrf_dataset.retrieval import QuerySubset, RetrievalDataset

DEFAULT_OUT_DIR = Path(__file__).resolve().parent.parent / "data" / "route_labels"

ROUTES_DIFFER = "routes_differ"
ALL_TIED = "all_tied"
ALL_ZERO = "all_zero"


def outcome_shape(scores: dict[str, float], tolerance: float = 1e-9) -> str:
    """Classify a query by whether its routes disagree.

    `ROUTES_DIFFER` is the only shape that teaches the router about
    dense-vs-sparse. `ALL_TIED` says any route works, so the cheapest should be
    served — signal for the speed requirement, not the quality one. `ALL_ZERO`
    means no route surfaced anything relevant, so **no valid label exists**: the
    builder's argmax falls through to tie-break order and reports whichever
    route is listed first, which is a fabricated label.
    """
    if not scores:
        return ALL_ZERO
    values = list(scores.values())
    if max(values) <= tolerance:
        return ALL_ZERO
    if max(values) - min(values) <= tolerance:
        return ALL_TIED
    return ROUTES_DIFFER


class RouteLabels:
    """Builds and owns `data/route_labels/labels.parquet`."""

    CARRIED = ["slice", "checkable", "label_lane"]
    """Selection columns copied onto every label, so outcome shapes can be read
    per composition slice rather than only in aggregate."""

    def __init__(
        self,
        selection: pd.DataFrame,
        out_dir: Path | None = None,
        objective: Objective | None = None,
    ) -> None:
        self.selection = selection
        self.objective = objective or RouterObjective()
        self._out_dir = out_dir if out_dir is not None else DEFAULT_OUT_DIR

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
    ) -> pd.DataFrame:
        """Label this dataset's selection rows and merge into the artifact.

        `dataset` names the selection's key for `source` when the two differ —
        the composition calls BEIR NFCorpus `beir-nfcorpus` while the retrieval
        class calls it `nfcorpus`. Re-labelling replaces that dataset's rows
        only; every other dataset's labels are left untouched.
        """
        key = dataset or source.name
        wanted = self.rows_for(key)
        if wanted.empty:
            raise ValueError(f"No selection rows for dataset {key!r}.")

        existing = self.load()
        if not force and (existing.get("dataset") == key).any():
            return existing[existing["dataset"] == key]

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
        rows = GoldenRoutingBuilder(
            dense, hybrid, sparse, objective=self.objective, excluded=exclusions
        ).build(subset, qrels=qrels)

        labelled = pd.DataFrame(
            [
                {
                    "dataset": key,
                    "query_id": row.query_id,
                    "query": row.query,
                    "route": str(row.strategy_name),
                    "score": row.metric,
                    **{f"score_{name}": value for name, value in row.route_scores.items()},
                    "shape": outcome_shape(row.route_scores),
                    "metric_name": row.metric_name,
                    "min_relevance": self.objective.min_relevance,
                }
                for row in rows
            ]
        )
        if labelled.empty:
            raise ValueError(
                f"{key!r}: no rows produced. Every selected query lacked "
                f"judgments — check that the qrels cover this selection."
            )
        labelled = labelled.merge(
            wanted[["query_id", *self.CARRIED]].astype({"query_id": str}),
            on="query_id",
            how="left",
        )

        merged = pd.concat(
            [existing[existing.get("dataset") != key], labelled], ignore_index=True
        )
        self.labels_path.parent.mkdir(parents=True, exist_ok=True)
        merged.to_parquet(self.labels_path, index=False)
        return labelled

    def coverage(self) -> pd.DataFrame:
        """Per-dataset progress: selected rows vs labelled, and the shape split.

        `qrels_ready` counts selected rows whose lane has qrels on disk but no
        label yet — the corpus-pending state between pass 1 and pass 2 (SPEC
        d39b). `unlabelled` is just `selected - labelled`; it asserts no
        cause. In practice a row is unlabelled because its lane's qrels are
        not fetched, its corpus is not indexed, or the source never judged it.
        """
        labels = self.load()
        counts = (
            labels.groupby(["dataset", "shape"]).size().unstack(fill_value=0)
            if "shape" in labels.columns and not labels.empty
            else pd.DataFrame()
        )
        none_yet: pd.Series = pd.Series(dtype=int)
        rows = []
        for dataset, selected in self.selection["dataset"].value_counts().items():
            shapes = counts.loc[dataset] if dataset in counts.index else none_yet
            done = int(shapes.sum())
            rows.append(
                {
                    "dataset": dataset,
                    "selected": int(selected),
                    "labelled": done,
                    "qrels_ready": self._qrels_ready(dataset, done),
                    "unlabelled": int(selected) - done,
                    ROUTES_DIFFER: int(shapes.get(ROUTES_DIFFER, 0)),
                    ALL_TIED: int(shapes.get(ALL_TIED, 0)),
                    ALL_ZERO: int(shapes.get(ALL_ZERO, 0)),
                }
            )
        return pd.DataFrame(rows).sort_values(
            ["labelled", "selected"], ascending=False, ignore_index=True
        )

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
