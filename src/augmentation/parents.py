"""The parent pool (d51g): catalog rows the dataset has never spent. Excluding
every selected query — and reserving each one already used as a parent — is
what makes a parent/child near-duplicate pair impossible rather than unlikely.
"""

from __future__ import annotations

from collections.abc import Mapping
from functools import lru_cache
from pathlib import Path

import pandas as pd

from augmentation.config import AugmentationPaths
from augmentation.pool import GeneratedPool
from augmentation.supply import lane_dirs
from composition.compose import RegistryDataset, join_text
from composition.floors import identifier_floor_key
from query_taxonomy.taxonomy import FeatureGroup

_IDENTIFIERS = f"{FeatureGroup.STRUCTURED_IDENTIFIERS.value}."


def _key(frame: pd.DataFrame) -> pd.Series:
    return frame["dataset"].astype(str) + "\x00" + frame["query_id"].astype(str)


@lru_cache(maxsize=8)
def _lane_qrels(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=["query_id", "doc_id", "relevance"])
    return pd.read_parquet(path).astype({"query_id": str, "doc_id": str})


def _corpus_text(path: Path, doc_id: str) -> str:
    """One document's text, read without loading the corpus — the lane parquets
    reach 133M rows, so this is a pushdown filter, never a full read."""
    if not path.exists():
        return ""
    rows = pd.read_parquet(path, filters=[("doc_id", "==", doc_id)])
    if rows.empty:
        return ""
    row = rows.iloc[0]
    title = str(row["title"]) if "title" in rows.columns else ""
    return f"{title}\n{row['text']}".strip()


def identifier_floors(catalog: pd.DataFrame) -> pd.Series:
    """Each row's id-floor keys, read off the catalog's span columns — the
    `floors` column Inject's structural check compares against."""
    columns = [c for c in catalog.columns if c.startswith(_IDENTIFIERS)]
    keys = [identifier_floor_key(c.rpartition(".")[2]) for c in columns]
    present = catalog[columns].to_numpy() > 0
    return pd.Series(
        [
            sorted({k for k, on in zip(keys, row, strict=True) if on})
            for row in present
        ],
        index=catalog.index,
    )


class ParentPool:
    """The unspent side of the catalog. Owns no artifact — it reads the
    catalog, the selection and the generated pool."""

    def __init__(
        self,
        catalog: pd.DataFrame,
        selection: pd.DataFrame,
        datasets: Mapping[str, RegistryDataset],
        *,
        pool: GeneratedPool | None = None,
        paths: AugmentationPaths | None = None,
    ) -> None:
        self._catalog = catalog
        self._selection = selection
        self._datasets = datasets
        self._paths = paths or AugmentationPaths()
        self._pool = pool or GeneratedPool(self._paths)

    def reserved(self) -> set[str]:
        """Queries already used as a parent — spent for good, so a later fill
        cannot select one and recreate the pair it was chosen to avoid."""
        pool = self._pool.load()
        if pool.empty:
            return set()
        used = pool.dropna(subset=["generated_from"])
        return set(
            used["parent_dataset"].astype(str)
            + "\x00"
            + used["generated_from"].astype(str)
        )

    def available(self) -> pd.DataFrame:
        """Checkable catalog rows the selection has not touched and no child
        already claims. Every row is natural by construction — the catalog
        holds only source queries — which is the first-generation guard,
        structurally rather than by a column test."""
        spent = set(_key(self._selection)) | self.reserved()
        return self._catalog[
            self._catalog["checkable"] & ~_key(self._catalog).isin(spent)
        ]

    def gold_text(self, parent: pd.Series, *, chars: int = 1200) -> str:
        """The document the parent was judged against, truncated. A destructive
        rewrite has to keep that document answering, so the cut needs to read
        it (d53); an unresolvable doc returns '' and the caller degrades to a
        corpus-blind instruction."""
        lane = lane_dirs().get(str(parent.get("dataset")))
        # a NaN grounding_doc_id is TRUTHY, so `or` alone would take it, stringify
        # it to 'nan', match nothing, and lose a perfectly good qrel fallback
        minted = parent.get("grounding_doc_id")
        doc_id = (
            minted
            if minted is not None and not pd.isna(minted)
            else self._judged_doc(parent, lane)
        )
        if lane is None or not doc_id:
            return ""
        return _corpus_text(self._paths.lane_corpus(lane), str(doc_id))[:chars]

    def _judged_doc(self, parent: pd.Series, lane: str | None) -> str | None:
        """The parent's own top-graded doc, for a cut with no minted surface to
        inherit a grounding doc from."""
        if lane is None:
            return None
        qrels = _lane_qrels(self._paths.lane_qrels(lane))
        rows = qrels[qrels["query_id"] == str(parent.get("query_id"))]
        if rows.empty:
            return None
        return str(rows.sort_values("relevance", ascending=False)["doc_id"].iloc[0])

    def hydrate(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Attach the columns an operator reads — query text and id floors —
        to the rows dispatch actually chose, dropping the textless ones. Kept
        off `available()` because the text join hits one parquet per lane, and
        the emptiness only becomes visible once the text is joined: some source
        caches carry blank query rows, and a blank parent spends an LLM call to
        rewrite nothing."""
        rows = frame.copy()
        rows["floors"] = identifier_floors(rows)
        rows["query"] = join_text(rows, self._datasets)
        text = rows["query"].fillna("").astype(str).str.strip()
        blank = text == ""
        if blank.any():
            print(f"parents: dropped {int(blank.sum())} row(s) with no query text")
        return rows[~blank]
