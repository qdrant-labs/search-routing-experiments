"""Answer keys born with the row (d43d): `data/augmentation/qrels.parquet`.

Minted the moment a candidate is accepted. Two paths, per the operator's
declared answer key: MINTED (Inject) writes source='constructed' rows against
every judged doc the injected surface still occurs in, at the parent's own
grades — a narrowed query keeps its parent's depth (d43d fix); INHERIT copies
the parent's judgments under the child's query_id keeping source='human' —
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
            rows = self._minted_rows(candidate)
        else:
            rows = self._inherit_rows(
                candidate.query_id, candidate.home_lane, candidate.generated_from
            )
        if rows.empty:
            return 0
        merged = pd.concat([existing, rows[list(_COLUMNS)]], ignore_index=True)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        merged.to_parquet(self.path, index=False)
        return len(rows)

    def mint_constructed(self, query_id: str, doc_id: str) -> int:
        """The synthetic rung's key: the one document written to answer this
        query. Depth of one is honest here — nobody judged anything else
        against it — and it does not collapse to a ceiling tie the way a
        minted Inject key would, because the constructed-docs collection
        carries borrowed distractors for the routes to disagree over."""
        existing = self.load()
        if query_id in set(existing["query_id"]):
            return 0
        row = pd.DataFrame(
            [{
                "query_id": query_id, "doc_id": doc_id, "relevance": 1,
                "source": "constructed", "inherited_from": None,
            }],
            columns=_COLUMNS,
        )
        self.path.parent.mkdir(parents=True, exist_ok=True)
        pd.concat([existing, row], ignore_index=True).to_parquet(
            self.path, index=False
        )
        return 1

    def backfill(self, pool: pd.DataFrame) -> pd.DataFrame:
        """Retry the inherit-path lookup for every pool row with no qrels
        entry yet — a parent whose lane wasn't fully materialized at mint
        time (d43d) may have judgments now. Pure retry: no new judgment
        source, so a row whose parent still has none stays exactly as
        unlabelable as before. A MINTED row should never be orphaned (its
        answer key has no external dependency at mint time) — reported
        separately rather than silently mishandled. Returns the query ids
        still unlabelable after this pass."""
        self._lane_qrels.clear()   # re-read every lane fresh — that's the point
        existing = self.load()
        orphaned = pool[~pool["query_id"].astype(str).isin(set(existing["query_id"]))]
        inherited = orphaned[orphaned["answer_key"] == str(AnswerKeyPath.INHERIT)]
        unexpected = orphaned[orphaned["answer_key"] != str(AnswerKeyPath.INHERIT)]
        if not unexpected.empty:
            print(
                f"backfill: {len(unexpected)} orphaned MINTED row(s) — "
                f"unexpected, not retried: {list(unexpected['query_id'])}"
            )

        recovered: list[pd.DataFrame] = []
        unlabelable: list[str] = []
        for _, row in inherited.iterrows():
            rows = self._inherit_rows(
                str(row["query_id"]), str(row["home_lane"]), str(row["generated_from"]),
            )
            if rows.empty:
                unlabelable.append(str(row["query_id"]))
            else:
                recovered.append(rows)
        if recovered:
            merged = pd.concat([existing, *recovered], ignore_index=True)
            self.path.parent.mkdir(parents=True, exist_ok=True)
            merged.to_parquet(self.path, index=False)
        print(
            f"backfill: {len(recovered)}/{len(inherited)} recovered, "
            f"{len(unlabelable)} still have no judgment anywhere"
        )
        return pd.DataFrame({"query_id": unlabelable})

    def _minted_rows(self, candidate: AugmentedCandidate) -> pd.DataFrame:
        """Inject's answer key: the parent's own human grades for every judged
        doc the injected surface still occurs in. Eligibility guaranteed >= 2
        such docs, so the narrowed child keeps its parent's depth instead of
        collapsing to a single doc that scores 1.0 for every route (a fake tie
        at ceiling — d43d fix). source stays 'constructed' — the query/doc
        PAIRING is Inject's; only the grade is the parent's real judgment, not
        the old synthetic 1 that graded lanes then thresholded to all_zero."""
        keep = {str(d) for d in candidate.grounding_doc_ids} or (
            {str(candidate.grounding_doc_id)}
            if candidate.grounding_doc_id is not None
            else set()
        )
        parent = self._parent_judgments(candidate.home_lane, candidate.generated_from)
        rows = parent[parent["doc_id"].astype(str).isin(keep)]
        if rows.empty:
            return pd.DataFrame(columns=_COLUMNS)
        return pd.DataFrame({
            "query_id": candidate.query_id,
            "doc_id": rows["doc_id"].astype(str),
            "relevance": rows["relevance"].astype(int),
            "source": "constructed",
            "inherited_from": None,
        })

    def _inherit_rows(
        self, query_id: str, home_lane: str, generated_from: str
    ) -> pd.DataFrame:
        """The qrels rows an inherit-path child copies from its parent's own
        judgments — empty when the parent's lane has none (yet)."""
        parent_rows = self._parent_judgments(home_lane, generated_from)
        if parent_rows.empty:
            return pd.DataFrame(columns=_COLUMNS)
        return pd.DataFrame({
            "query_id": query_id,
            "doc_id": parent_rows["doc_id"].astype(str),
            "relevance": parent_rows["relevance"].astype(int),
            "source": "human",
            "inherited_from": generated_from,
        })

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
