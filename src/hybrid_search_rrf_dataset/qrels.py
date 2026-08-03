"""Normalized relevance judgments.

One row per (dataset, query_id, doc_id), replacing the `gold_qrel` dict that
used to ride on every result row: parquet encodes a doc_id-keyed dict as a
struct with one field per distinct doc_id in the whole file — 30,000 fields at
76 rows — which does not survive composition scale.

`source` keeps assessor judgments, clicks, and LLM judgments in one table, so
scoring a dataset against any lane is a filter rather than a second pipeline.
"""

from __future__ import annotations

from collections.abc import Iterable
from enum import StrEnum
from pathlib import Path

import pandas as pd

from hybrid_search_rrf_dataset.retrieval import RetrievalDataset


class QrelSource(StrEnum):
    """Provenance of a judgment. **Declaration order is precedence** — earlier
    members win when two lanes judge the same (query, doc). Insert new members
    at the right rank rather than appending."""

    HUMAN = "human"
    """Assessor judgments shipped with the dataset. Inherit-path augmented
    children carry copies of these under their own query_id — the judgment
    is still a human's, only the query changed under a declared operator
    (d43d; the transfer is recorded in the augmentation qrels artifact)."""

    CONSTRUCTED = "constructed"
    """The answer key exists by construction (d40b): Inject rows minted
    against their grounding doc, synthetic closed-world rows. Definitional
    rather than judged — below HUMAN, above behavioral CLICK."""

    CLICK = "click"
    """A user clicked this doc for this query — ORCAS's 18.8M pairs over
    msmarco-document. Weaker than an assessor judgment (position bias, no true
    negatives) but it is a real relevance signal, and it is why no dataset in
    the composition needs manufactured qrels (SPEC d37k)."""

    LLM = "llm"
    """Judgments produced by an LLM to fill holes the assessors left — docs a
    retriever surfaced that nobody graded. Deferred per SPEC d37(k) until the
    per-route hole rate is measured, so nothing writes this lane yet."""


_PRIORITY = {source: rank for rank, source in enumerate(QrelSource)}
"""Derived from declaration order so a new `QrelSource` cannot desync it."""


class QrelStore:
    """Long-format qrels across any number of datasets."""

    COLUMNS = ["dataset", "query_id", "doc_id", "relevance", "source"]

    def __init__(self, frame: pd.DataFrame) -> None:
        missing = set(self.COLUMNS) - set(frame.columns)
        if missing:
            raise ValueError(
                f"QrelStore needs {self.COLUMNS}; missing {sorted(missing)}."
            )
        subset = frame[self.COLUMNS]
        if subset.isna().any().any():
            raise ValueError("QrelStore rejects nulls; every column is required.")
        self.frame = subset.astype(
            {
                "dataset": str,
                "query_id": str,
                "doc_id": str,
                "relevance": int,
                "source": str,
            }
        ).reset_index(drop=True)

    def __len__(self) -> int:
        return len(self.frame)

    @classmethod
    def from_dataset(
        cls,
        dataset: RetrievalDataset,
        source: QrelSource = QrelSource.HUMAN,
    ) -> QrelStore:
        """Wrap a `RetrievalDataset`'s own qrels, tagged with `dataset.name`."""
        return cls(
            dataset.qrels().assign(dataset=dataset.name, source=str(source))
        )

    @classmethod
    def concat(cls, stores: Iterable[QrelStore]) -> QrelStore:
        frames = [s.frame for s in stores]
        if not frames:
            return cls(pd.DataFrame(columns=cls.COLUMNS))
        return cls(pd.concat(frames, ignore_index=True))

    def for_source(self, source: QrelSource) -> QrelStore:
        return QrelStore(self.frame[self.frame["source"] == str(source)])

    def lookup(
        self,
        dataset: str,
        source: QrelSource | None = None,
    ) -> dict[str, dict[str, int]]:
        """`{query_id: {doc_id: relevance}}` for one dataset — the shape the
        objectives score against.

        With `source` set, only that lane is returned. Without it, lanes are
        merged and the highest-precedence source wins any (query, doc) that
        more than one lane judged.
        """
        rows = self.frame[self.frame["dataset"] == dataset]
        if source is not None:
            rows = rows[rows["source"] == str(source)]
        else:
            rows = self._resolve_conflicts(rows)

        out: dict[str, dict[str, int]] = {}
        for query_id, doc_id, relevance in zip(
            rows["query_id"], rows["doc_id"], rows["relevance"], strict=True
        ):
            out.setdefault(query_id, {})[doc_id] = int(relevance)
        return out

    @staticmethod
    def _resolve_conflicts(rows: pd.DataFrame) -> pd.DataFrame:
        """Keep one judgment per (query, doc), preferring the highest-precedence
        source. No-op when only one lane is present."""
        if rows["source"].nunique() <= 1:
            return rows
        ranked = rows.assign(
            _precedence=[_PRIORITY[QrelSource(s)] for s in rows["source"]]
        ).sort_values("_precedence", kind="stable")
        return ranked.drop_duplicates(["query_id", "doc_id"], keep="first")

    def save(self, path: Path | str) -> Path:
        file = Path(path)
        file.parent.mkdir(parents=True, exist_ok=True)
        self.frame.to_parquet(file, index=False)
        return file

    @classmethod
    def load(cls, path: Path | str) -> QrelStore:
        return cls(pd.read_parquet(path))
