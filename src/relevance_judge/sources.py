"""Read-only data access for the judge: oracle route_rankings, corpus text, and
manifest gold. Everything here is a lookup against artifacts other stages own —
nothing is computed or written."""

from __future__ import annotations

from functools import cached_property
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

from relevance_judge.config import RelevanceJudgeConfig

ROUTES = ("dense_only", "sparse_only", "pure_rrf")


class Sources:
    """One handle over the on-disk truth the queue and scorer read."""

    def __init__(self, config: RelevanceJudgeConfig | None = None) -> None:
        self.config = config or RelevanceJudgeConfig()
        self._rankings: dict[str, dict[str, dict[str, list[str]]]] = {}
        self._gold: dict[str, dict[str, set[str]]] = {}

    def _cache_paths(self, dataset: str) -> list[Path]:
        """Oracle caches that may hold this lane's rankings, precedence first —
        the v2-100K labelling caches match the target, route_labels backstops."""
        base = self.config.data_dir
        candidates = [
            base / "rungs" / "100k-v2" / "labeling" / "natural" / f"{dataset}_oracle" / "rows.parquet",
            base / "rungs" / "100k-v2" / "labeling" / "supplemented" / f"{dataset}_oracle" / "rows.parquet",
            base / "route_labels" / f"{dataset}_oracle" / "rows.parquet",
        ]
        return [p for p in candidates if p.exists()]

    def rankings(self, dataset: str) -> dict[str, dict[str, list[str]]]:
        """`{query_id: {route: [doc_id, ...top-10]}}` for a lane, merged across
        caches (first cache wins). Empty when no cache carries the lane — those
        rows cannot be rescored and the caller must drop them (invariant §5a-3)."""
        if dataset not in self._rankings:
            merged: dict[str, dict[str, list[str]]] = {}
            for path in self._cache_paths(dataset):
                frame = pd.read_parquet(path, columns=["query_id", "route_rankings"])
                for query_id, rr in zip(
                    frame["query_id"].astype(str), frame["route_rankings"], strict=True
                ):
                    if query_id not in merged and rr is not None:
                        merged[query_id] = {r: [str(d) for d in rr[r]] for r in rr}
            self._rankings[dataset] = merged
        return self._rankings[dataset]

    @cached_property
    def _manifest(self) -> pd.DataFrame:
        path = self.config.v2_100k / "candidate_manifests.parquet"
        return pd.read_parquet(path).astype({"query_id": str, "doc_id": str})

    def manifest_gold(self, dataset: str) -> dict[str, set[str]]:
        """`{query_id: {gold_doc_id, ...}}` at `min_relevance` — the answer key
        arch5k's l2 triples were scored against (not base qrels; §gold-source)."""
        if dataset not in self._gold:
            lane = self._manifest[
                (self._manifest["dataset"] == dataset)
                & (self._manifest["relevance"] >= self.config.min_relevance)
            ]
            self._gold[dataset] = {
                query_id: set(group["doc_id"])
                for query_id, group in lane.groupby("query_id")
            }
        return self._gold[dataset]

    def corpus_text(self, dataset: str, doc_ids: set[str]) -> dict[str, str]:
        """`{doc_id: title+text}` for the wanted docs only — a filtered read, so
        a 257MB corpus never lands in memory whole (CLAUDE.md memory rule). Some
        lanes ship no `title` column, so only existing columns are requested."""
        path = self.config.data_dir / dataset / "corpus.parquet"
        if not path.exists() or not doc_ids:
            return {}
        available = set(pq.read_schema(path).names)
        columns = [c for c in ("doc_id", "title", "text") if c in available]
        frame = pd.read_parquet(
            path, columns=columns, filters=[("doc_id", "in", list(doc_ids))]
        ).astype({"doc_id": str})
        for missing in ("title", "text"):
            if missing not in frame.columns:
                frame[missing] = ""
        out: dict[str, str] = {}
        for row in frame.itertuples(index=False):
            title = row.title if isinstance(row.title, str) else ""
            text = row.text if isinstance(row.text, str) else ""
            out[str(row.doc_id)] = f"{title}\n\n{text}".strip() if title else text.strip()
        return out

    def query_text(self, dataset: str, query_ids: set[str]) -> dict[str, str]:
        """`{query_id: text}` for the wanted queries — a filtered read."""
        path = self.config.data_dir / dataset / "queries.parquet"
        if not path.exists() or not query_ids:
            return {}
        frame = pd.read_parquet(
            path, columns=["query_id", "text"],
            filters=[("query_id", "in", list(query_ids))],
        ).astype({"query_id": str})
        return {str(q): (t or "") for q, t in zip(frame["query_id"], frame["text"], strict=True)}

    def base_qrels(self, dataset: str) -> pd.DataFrame:
        """Human judgments shipped with the lane — the validation referee. Columns
        query_id, doc_id, relevance (positive-only on most lanes)."""
        path = self.config.data_dir / dataset / "qrels.parquet"
        if not path.exists():
            return pd.DataFrame(columns=["query_id", "doc_id", "relevance"])
        return pd.read_parquet(path).astype({"query_id": str, "doc_id": str})

    def lanes_with_negatives(self) -> list[str]:
        """Registered human lanes with judged-IRRELEVANT docs (grade 0) — the only
        valid precision referees. A hard negative must be human-judged irrelevant,
        NOT merely below a strict min_relevance: a grade-1 doc in a min_relevance=2
        lane (e.g. nfcorpus, all grade 1/2) is relevant-but-marginal, and counting
        it as a negative manufactures false positives. Lanes not in `LANES` (e.g.
        synthetic `augmentation`) are excluded too. One-column read per lane."""
        from hybrid_search_rrf_dataset.lanes import LANES

        found = []
        for path in sorted(self.config.data_dir.glob("*/qrels.parquet")):
            lane = path.parent.name
            if lane not in LANES or lane in self.config.excluded_referee_lanes:
                continue
            relevance = pd.read_parquet(path, columns=["relevance"])["relevance"]
            if (relevance < 1).any():
                found.append(lane)
        return found
