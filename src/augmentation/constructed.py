"""The constructed-docs lane: documents the synthetic rung writes because no
corpus supplies them.

Its own store, never an existing lane's corpus — one added document can steal
rank-1 and silently falsify labels that lane has already paid for. Only the
synthetic rung writes here; every other operator is query-side only.
"""

from __future__ import annotations

from pathlib import Path
from typing import Final

import pandas as pd

from augmentation.config import AugmentationPaths

_COLUMNS: Final[tuple[str, ...]] = (
    "doc_id",          # the constructed document's own id
    "source_dataset",  # lane whose corpus lends this doc its distractors
    "for_query",       # query_id this document was written to answer
    "text",            # the document itself
)


class ConstructedDocs:
    """Owns `paths.constructed_docs` — idempotent on doc_id."""

    def __init__(self, paths: AugmentationPaths | None = None) -> None:
        self._paths = paths or AugmentationPaths()

    @property
    def path(self) -> Path:
        return self._paths.constructed_docs

    @staticmethod
    def doc_id(query_id: str, ordinal: int = 1) -> str:
        """Derivable rather than counted — a rerun that regenerates a row
        overwrites nothing; the ordinal exists because one doc is depth 1,
        which classify() files as a fake tie."""
        return f"constructed-{query_id}-{ordinal}"

    def load(self) -> pd.DataFrame:
        if not self.path.exists():
            return pd.DataFrame(columns=_COLUMNS)
        return pd.read_parquet(self.path)

    def written_for(self) -> set[str]:
        """Queries that already have their document — a rerun neither
        regenerates nor re-pays for them."""
        return set(self.load()["for_query"].astype(str))

    def add(
        self, *, query_id: str, source_dataset: str, text: str, ordinal: int = 1
    ) -> str:
        """Bank one document and return its id."""
        doc_id = self.doc_id(query_id, ordinal)
        existing = self.load()
        if doc_id in set(existing["doc_id"]):
            return doc_id
        row = pd.DataFrame(
            [{
                "doc_id": doc_id,
                "source_dataset": source_dataset,
                "for_query": query_id,
                "text": text,
            }],
            columns=_COLUMNS,
        )
        self.path.parent.mkdir(parents=True, exist_ok=True)
        pd.concat([existing, row], ignore_index=True).to_parquet(
            self.path, index=False
        )
        return doc_id
