"""Answer keys born with the row (d43d): `data/augmentation/qrels.parquet`.

Minted the moment a candidate is accepted. Two paths, per the operator's
declared answer key: MINTED (Inject, synthetic) writes one
source='constructed' row against the grounding doc; INHERIT copies the
parent's judgments under the child's query_id keeping source='human' —
the judgment is still a human's, only the query changed under a declared
operator — with `inherited_from` recording the transfer. QrelStore merges
by its declaration-order precedence at labeling.
"""

from __future__ import annotations

from pathlib import Path
from typing import Final

import pandas as pd

from augmentation.config import AugmentationPaths
from augmentation.core import AnswerKeyPath, AugmentedCandidate
from augmentation.supply import lane_dirs

_COLUMNS: Final[tuple[str, ...]] = (
    "query_id",        # the augmented child this key belongs to
    "doc_id",          # the judged document
    "relevance",       # graded judgment; minted rows write 1
    "source",          # 'constructed' when minted, 'human' when copied over
    "inherited_from",  # parent whose judgment this copies; None when minted
)


class AugmentationQrels:
    """Owns `paths.qrels` — idempotent on query_id."""

    def __init__(self, paths: AugmentationPaths | None = None) -> None:
        self._paths = paths or AugmentationPaths()
        self._lane_qrels: dict[str, pd.DataFrame] = {}

    @property
    def path(self) -> Path:
        return self._paths.qrels

    def load(self) -> pd.DataFrame:
        if not self.path.exists():
            return pd.DataFrame(columns=_COLUMNS)
        return pd.read_parquet(self.path)

    def mint(self, candidate: AugmentedCandidate) -> int:
        """Write the candidate's answer key; returns rows written (0 when
        the key already exists, or when an inherit-path parent has no
        judgments on disk — such a child is as unlabelable as its parent
        was, honestly)."""
        existing = self.load()
        if candidate.query_id in set(existing["query_id"]):
            return 0
        if candidate.answer_key is AnswerKeyPath.MINTED:
            rows = pd.DataFrame([{
                "query_id": candidate.query_id,
                "doc_id": str(candidate.grounding_doc_id),
                "relevance": 1,
                "source": "constructed",
                "inherited_from": None,
            }])
        else:
            parent_rows = self._parent_judgments(
                candidate.home_lane, candidate.generated_from
            )
            if parent_rows.empty:
                return 0
            rows = pd.DataFrame({
                "query_id": candidate.query_id,
                "doc_id": parent_rows["doc_id"].astype(str),
                "relevance": parent_rows["relevance"].astype(int),
                "source": "human",
                "inherited_from": candidate.generated_from,
            })
        merged = pd.concat([existing, rows[list(_COLUMNS)]], ignore_index=True)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        merged.to_parquet(self.path, index=False)
        return len(rows)

    def _parent_judgments(self, home_lane: str, parent_id: str) -> pd.DataFrame:
        lane = lane_dirs().get(home_lane)
        if lane is None:
            return pd.DataFrame(columns=["doc_id", "relevance"])
        if lane not in self._lane_qrels:
            path = self._paths.lane_qrels(lane)
            self._lane_qrels[lane] = (
                pd.read_parquet(path).astype({"query_id": str})
                if path.exists()
                else pd.DataFrame(columns=["query_id", "doc_id", "relevance"])
            )
        qrels = self._lane_qrels[lane]
        return qrels[qrels["query_id"] == str(parent_id)]
