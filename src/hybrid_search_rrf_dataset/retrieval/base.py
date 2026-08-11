"""Retrieval-side contracts: what a lane must provide, how a snapshot is
read back, and the 20/80 rule that sizes a corpus from its own qrels."""

from __future__ import annotations

import random
from abc import ABC, abstractmethod
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any, ClassVar, NamedTuple

import pandas as pd
from tqdm.auto import tqdm

CORPUS_COLUMNS = ["doc_id", "title", "text"]
QUERY_COLUMNS = ["query_id", "text"]
QREL_COLUMNS = ["query_id", "doc_id", "relevance"]


class RetrievalDataset(ABC):
    name: str

    @abstractmethod
    def corpus(self) -> pd.DataFrame:
        """Return corpus with columns: doc_id, title, text."""

    @abstractmethod
    def queries(self) -> pd.DataFrame:
        """Return queries with columns: query_id, text."""

    @abstractmethod
    def qrels(self) -> pd.DataFrame:
        """Return relevance judgments with columns: query_id, doc_id, relevance."""

    def excluded(self) -> pd.DataFrame:
        """Per-query doc exclusions (query_id, doc_id) — docs that must not be
        scored against that query (BRIGHT leakage lists, SPEC d39). Empty for
        most datasets; applied at scoring time, never by dropping corpus docs."""
        return pd.DataFrame(columns=["query_id", "doc_id"])

    def provenance(self) -> pd.DataFrame:
        """Per-query origin (query_id, provenance) for queries this dataset
        knows are NOT its own natural rows. Empty for most datasets — a query
        absent here is natural by convention (`golden.py`'s `_iter_queries`
        default), so ordinary sources never need to enumerate the obvious."""
        return pd.DataFrame(columns=["query_id", "provenance"])

    @staticmethod
    def _queries_frame(dataset: Any) -> pd.DataFrame:
        return pd.DataFrame(
            [{"query_id": q.query_id, "text": q.text} for q in dataset.queries_iter()],
            columns=QUERY_COLUMNS,
        )

    @staticmethod
    def _qrels_frame(dataset: Any) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "query_id": r.query_id,
                    "doc_id": r.doc_id,
                    "relevance": r.relevance,
                }
                for r in dataset.qrels_iter()
            ],
            columns=QREL_COLUMNS,
        )

    def materialize(self) -> None:
        """Pre-filter into a self-consistent snapshot before save();
        default: nothing to do."""

    def save_metadata(self, path: Path | str = Path("data")) -> None:
        """Write queries + qrels only — the corpus-pending snapshot (SPEC
        d39b). Pass 1 of a lane; `save()` completes it in pass 2."""
        out = Path(path) / self.name
        out.mkdir(parents=True, exist_ok=True)
        self.queries().to_parquet(out / "queries.parquet", index=False)
        self.qrels().to_parquet(out / "qrels.parquet", index=False)

    def save(self, path: Path | str = Path("data")) -> None:
        corpus = self.corpus()
        if corpus.empty:
            raise ValueError(
                f"{self.name}: corpus is empty — materialize() first, or use "
                "save_metadata() for a corpus-pending snapshot."
            )
        self.save_metadata(path)
        corpus.to_parquet(Path(path) / self.name / "corpus.parquet", index=False)


class SnapshotDataset(RetrievalDataset):
    """Reads back a snapshot written by `RetrievalDataset.save()` or
    `save_metadata()`.

    Queries + qrels are required; the corpus may still be pending (SPEC
    d39b) — then `corpus()` fails loud and `has_corpus` is False, while
    qrels-side consumers (coverage, QrelStore) proceed.
    """

    def __init__(self, name: str, path: Path | str = Path("data")) -> None:
        self.name = name
        self._dir = Path(path) / name
        missing = [
            f"{part}.parquet"
            for part in ("queries", "qrels")
            if not (self._dir / f"{part}.parquet").exists()
        ]
        if missing:
            raise FileNotFoundError(f"{self._dir} is missing {missing}.")

    @property
    def has_corpus(self) -> bool:
        return (self._dir / "corpus.parquet").exists()

    def corpus(self) -> pd.DataFrame:
        if not self.has_corpus:
            raise FileNotFoundError(
                f"{self._dir} is corpus-pending (pass 1 only) — index it in "
                "pass 2 before retrieval."
            )
        return pd.read_parquet(self._dir / "corpus.parquet")

    def excluded(self) -> pd.DataFrame:
        path = self._dir / "excluded.parquet"
        if not path.exists():
            return super().excluded()
        return pd.read_parquet(path)

    def queries(self) -> pd.DataFrame:
        return pd.read_parquet(self._dir / "queries.parquet")

    def qrels(self) -> pd.DataFrame:
        return pd.read_parquet(self._dir / "qrels.parquet")


class QuerySubset(RetrievalDataset):
    """A dataset narrowed to specific query ids, keeping the source's corpus.

    The composition picked particular rows out of each source dataset, so
    labeling must run over *those* queries rather than the dataset's full query
    set. Keeps `source.name` so a `QrelStore` lookup still resolves.
    """

    def __init__(self, source: RetrievalDataset, query_ids: Iterable[str]) -> None:
        self.name = source.name
        self._source = source
        self._ids = {str(q) for q in query_ids}

    def corpus(self) -> pd.DataFrame:
        return self._source.corpus()

    def queries(self) -> pd.DataFrame:
        queries = self._source.queries()
        return queries[queries["query_id"].astype(str).isin(self._ids)].reset_index(
            drop=True
        )

    def qrels(self) -> pd.DataFrame:
        qrels = self._source.qrels()
        return qrels[qrels["query_id"].astype(str).isin(self._ids)].reset_index(
            drop=True
        )

    def excluded(self) -> pd.DataFrame:
        excluded = self._source.excluded()
        return excluded[excluded["query_id"].astype(str).isin(self._ids)].reset_index(
            drop=True
        )


class QuerySupplement(RetrievalDataset):
    """A dataset with extra queries and judgments layered on top of a source,
    keeping the source's own queries/qrels exactly as they are.

    Mirrors `QuerySubset`'s constructor shape but ADDS rows instead of
    narrowing them — named "supplement" rather than "overlay" (SPEC d61): an
    overlay reads as covering what's underneath, a supplement only adds
    alongside it, so the source's own snapshot stays exactly reproducible
    from scratch. `extra_queries`/`extra_qrels` use the same column contract
    as `queries()`/`qrels()` respectively.
    """

    def __init__(
        self,
        source: RetrievalDataset,
        extra_queries: pd.DataFrame,
        extra_qrels: pd.DataFrame,
    ) -> None:
        self.name = source.name
        self._source = source
        self._extra_queries = extra_queries
        self._extra_qrels = extra_qrels

    def corpus(self) -> pd.DataFrame:
        return self._source.corpus()

    def queries(self) -> pd.DataFrame:
        added = pd.concat(
            [self._source.queries(), self._extra_queries[QUERY_COLUMNS]],
            ignore_index=True,
        )
        return added.drop_duplicates("query_id", keep="first").reset_index(drop=True)

    def qrels(self) -> pd.DataFrame:
        added = pd.concat(
            [self._source.qrels(), self._extra_qrels[QREL_COLUMNS]],
            ignore_index=True,
        )
        return added.drop_duplicates(
            ["query_id", "doc_id"], keep="first"
        ).reset_index(drop=True)

    def excluded(self) -> pd.DataFrame:
        return self._source.excluded()

    def provenance(self) -> pd.DataFrame:
        if "provenance" not in self._extra_queries.columns:
            return pd.DataFrame(columns=["query_id", "provenance"])
        return self._extra_queries[["query_id", "provenance"]]


class CorpusRecipe(NamedTuple):
    """The d39(e) 20/80 corpus-sizing rule, as three named parameters.

    `target()` computes a lane's corpus size from its own measured
    judged-relevant count — nothing is hand-picked per lane. `answer_share`
    is the intended fraction of the corpus that answers *some* query;
    `floor` keeps ranking non-trivial against the strategies' fetch depth;
    `ceiling` keeps answer-heavy lanes (clinical would want 198K) from
    ballooning the embedding bill.
    """

    answer_share: float = 0.2
    floor: int = 10_000
    ceiling: int = 100_000

    def target(self, relevant: int) -> int:
        return min(max(round(relevant / self.answer_share), self.floor), self.ceiling)


class MaterializedDataset(RetrievalDataset, ABC):
    """A dataset whose corpus is too large to build on demand.

    `corpus`/`queries`/`qrels` serve in-memory frames that stay empty until
    `load_metadata()` (queries + qrels — the pass-1 fill) or `materialize()`
    (full snapshot) fills them. Source-agnostic: subclasses decide where the
    frames come from.
    """

    recipe: ClassVar[CorpusRecipe] = CorpusRecipe()

    def __init__(self) -> None:
        self._corpus_df = pd.DataFrame(columns=CORPUS_COLUMNS)
        self._queries_df = pd.DataFrame(columns=QUERY_COLUMNS)
        self._qrels_df = pd.DataFrame(columns=QREL_COLUMNS)
        self.corpus_target: int | None = None
        """Explicit target override — only for the recorded exceptions
        (a lane whose relevant docs exceed the recipe ceiling, or one whose
        design requires the full corpus). None = computed by `recipe`."""
        self.query_ids: set[str] | None = None
        """The query ids this snapshot must serve, pinned from the current
        composition selection before load_metadata(). Lanes whose upstream is
        too big to keep whole consult this instead of their seeded self-sample;
        a snapshot sampled independently of the selection silently drops
        selected queries from the labelled set. None = lane-default behavior."""

    def corpus(self) -> pd.DataFrame:
        return self._corpus_df

    def queries(self) -> pd.DataFrame:
        return self._queries_df

    def qrels(self) -> pd.DataFrame:
        return self._qrels_df

    @abstractmethod
    def load_metadata(self) -> None:
        """Load queries + qrels only. Cheap — no corpus touch."""

    @abstractmethod
    def _iter_corpus(self) -> Iterator[dict[str, str]]:
        """Stream the full source corpus as {doc_id, title, text} dicts."""

    def hydrate(self, path: Path | str = Path("data")) -> bool:
        """Fill queries + qrels from an on-disk snapshot instead of the
        network; True when the snapshot exists. The pass-2 loop calls this
        first so pass-1 fetches are never repeated."""
        out = Path(path) / self.name
        if not (out / "qrels.parquet").exists():
            return False
        self._queries_df = pd.read_parquet(out / "queries.parquet")
        self._qrels_df = pd.read_parquet(out / "qrels.parquet")
        return True

    def restrict(self, query_ids: Iterable[str]) -> None:
        """Drop every query outside `query_ids`, so `materialize()` sizes the
        corpus from the remaining qrels alone."""
        ids = {str(q) for q in query_ids}
        queries, qrels = self._queries_df, self._qrels_df
        self._queries_df = queries[
            queries["query_id"].astype(str).isin(ids)
        ].reset_index(drop=True)
        self._qrels_df = qrels[qrels["query_id"].astype(str).isin(ids)].reset_index(
            drop=True
        )

    def materialize(self) -> None:
        """The d39(e) 20/80 recipe in one pass over the corpus stream.

        The corpus target is *computed* from this lane's own qrels by
        `CorpusRecipe` (or pinned via `corpus_target` for the recorded
        exceptions). Judged-relevant docs are force-included (d38c); every
        other doc goes through a seeded uniform reservoir of size
        `target - relevant`. When the corpus fits the target the reservoir
        keeps everything, so the full-index branch is the same code path,
        not a special case.
        """
        if self._qrels_df.empty:
            raise ValueError(
                f"{self.name}: hydrate() or load_metadata() before materialize()."
            )
        relevant = set(
            self._qrels_df.loc[self._qrels_df["relevance"] >= 1, "doc_id"].astype(str)
        )
        target = self.corpus_target or self.recipe.target(len(relevant))
        budget = target - len(relevant)
        if budget < 0:
            raise ValueError(
                f"{self.name}: {len(relevant):,} judged-relevant docs exceed "
                f"the {target:,} target — the recipe ceiling is too low for "
                "this lane; pin an explicit corpus_target deliberately."
            )
        rng = random.Random(0)
        forced: list[dict[str, str]] = []
        pool: list[dict[str, str]] = []
        others = 0
        # Source corpora can repeat rows (crumb code_retrieval ships 3,702
        # exact-duplicate passages); duplicates would collapse onto one uuid5
        # point at upload, so first occurrence wins here.
        seen: set[str] = set()
        for doc in tqdm(
            self._iter_corpus(), desc=f"materialize:{self.name}", unit="doc"
        ):
            if doc["doc_id"] in seen:
                continue
            seen.add(doc["doc_id"])
            if doc["doc_id"] in relevant:
                forced.append(doc)
            else:
                others += 1
                if len(pool) < budget:
                    pool.append(doc)
                else:
                    slot = rng.randrange(others)
                    if slot < budget:
                        pool[slot] = doc
        self._corpus_df = pd.DataFrame([*forced, *pool], columns=CORPUS_COLUMNS)
