"""Generic label cache — an infrastructure service, not a composition input.

A reusable entry is keyed on the full fingerprint tuple in spec:143-148. An
identity-only match is a cache miss: changing the corpus or the retrieval
stack changes the route outcome, and reusing a stale row would silently
lie about what the label measures. Absent fingerprints on legacy label
files are therefore misses too (spec:151-153).
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA = REPO_ROOT / "src" / "data"

FP_COLUMNS = ("query_fp", "corpus_fp", "qrels_fp", "retrieval_stack_fp")


@dataclass(frozen=True)
class CacheKey:
    """Full-fingerprint identity of a labeled row (spec:143-148)."""

    dataset: str
    query_id: str
    query_fp: str
    corpus_fp: str
    qrels_fp: str
    retrieval_stack_fp: str

    def digest(self) -> str:
        material = "\x1f".join((
            self.dataset,
            self.query_id,
            self.query_fp,
            self.corpus_fp,
            self.qrels_fp,
            self.retrieval_stack_fp,
        ))
        return hashlib.sha256(material.encode()).hexdigest()


class LabelCache:
    """Read-only view over label files that carry the four fingerprints. Rows
    whose fingerprint columns are absent or blank are ignored — a strict
    reading of spec:151-153. Writes belong to whichever process labels."""

    def __init__(self, sources: Iterable[Path] | None = None) -> None:
        self._sources: list[Path] = list(sources) if sources is not None else []
        self._index: dict[str, pd.Series] | None = None

    def add_source(self, path: Path) -> "LabelCache":
        self._sources.append(path)
        self._index = None
        return self

    def _load(self) -> dict[str, pd.Series]:
        if self._index is not None:
            return self._index
        rows: dict[str, pd.Series] = {}
        for path in self._sources:
            if not path.exists():
                continue
            frame = pd.read_parquet(path)
            missing = [c for c in FP_COLUMNS if c not in frame.columns]
            if missing:
                # a legacy label file without fingerprints is not reusable — spec:151-153
                continue
            frame = frame.astype({
                "dataset": str, "query_id": str,
                "query_fp": str, "corpus_fp": str,
                "qrels_fp": str, "retrieval_stack_fp": str,
            })
            for _, row in frame.iterrows():
                if not all(row[c] for c in FP_COLUMNS):
                    continue
                key = CacheKey(
                    dataset=row["dataset"],
                    query_id=row["query_id"],
                    query_fp=row["query_fp"],
                    corpus_fp=row["corpus_fp"],
                    qrels_fp=row["qrels_fp"],
                    retrieval_stack_fp=row["retrieval_stack_fp"],
                ).digest()
                rows[key] = row
        self._index = rows
        return rows

    def lookup(self, key: CacheKey) -> pd.Series | None:
        """Return the labeled row for `key` or None. A row present under the
        same (dataset, query_id) but different fingerprints is a miss."""
        return self._load().get(key.digest())

    def has(self, key: CacheKey) -> bool:
        return key.digest() in self._load()

    def size(self) -> int:
        return len(self._load())
