from __future__ import annotations

import random
from abc import ABC, abstractmethod
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import ir_datasets
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

    def save(self, path: Path | str = Path("data")) -> None:
        out = Path(path) / self.name
        out.mkdir(parents=True, exist_ok=True)
        self.corpus().to_parquet(out / "corpus.parquet", index=False)
        self.queries().to_parquet(out / "queries.parquet", index=False)
        self.qrels().to_parquet(out / "qrels.parquet", index=False)


class SnapshotDataset(RetrievalDataset):
    """Reads back a snapshot written by `RetrievalDataset.save()`.

    The counterpart of `save()`, so a pipeline can consume a materialized
    dataset without re-touching ir_datasets or the network.
    """

    def __init__(self, name: str, path: Path | str = Path("data")) -> None:
        self.name = name
        self._dir = Path(path) / name
        missing = [
            f"{part}.parquet"
            for part in ("corpus", "queries", "qrels")
            if not (self._dir / f"{part}.parquet").exists()
        ]
        if missing:
            raise FileNotFoundError(f"{self._dir} is missing {missing}.")

    def corpus(self) -> pd.DataFrame:
        return pd.read_parquet(self._dir / "corpus.parquet")

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


class MaterializedDataset(RetrievalDataset, ABC):
    """A dataset whose corpus is too large to build on demand.

    `corpus`/`queries`/`qrels` serve in-memory frames that stay empty until
    `materialize()` (full snapshot) or `load_metadata()` + `fetch_docs()`
    (targeted lookup) fills them.
    """

    def __init__(self) -> None:
        self._corpus_df = pd.DataFrame(columns=CORPUS_COLUMNS)
        self._queries_df = pd.DataFrame(columns=QUERY_COLUMNS)
        self._qrels_df = pd.DataFrame(columns=QREL_COLUMNS)

    def corpus(self) -> pd.DataFrame:
        return self._corpus_df

    def queries(self) -> pd.DataFrame:
        return self._queries_df

    def qrels(self) -> pd.DataFrame:
        return self._qrels_df

    def load_metadata(self) -> None:
        """Load queries + qrels only. Cheap (seconds) — no corpus touch.
        For pipelines that need specific docs, follow with `fetch_docs`."""
        self._qrels_df = self._qrels_frame(self._test_ds)
        self._queries_df = self._queries_frame(self._test_ds)

    @property
    @abstractmethod
    def _test_ds(self) -> Any: ...

    @abstractmethod
    def fetch_docs(self, doc_ids: Iterable[str]) -> dict[str, str]:
        """Return `{doc_id: text}` for a specific list of doc_ids."""


class TrecDL2022(MaterializedDataset):
    """TREC Deep Learning 2022 passage retrieval task (MS MARCO passage v2).

    ~138M passages with no title field, so `corpus()` returns title="" for
    every row.
    """

    name = "trec-dl-2022"

    _CORPUS_ID = "msmarco-passage-v2"
    _TEST_ID = "msmarco-passage-v2/trec-dl-2022"

    def __init__(self, corpus_limit: int | None = None) -> None:
        """`corpus_limit` is only required by `materialize()`; callers that only
        need queries + qrels via `load_metadata()` can leave it None."""
        super().__init__()
        self.corpus_limit = corpus_limit
        self._corpus_ds = ir_datasets.load(self._CORPUS_ID)
        self._test_dataset = ir_datasets.load(self._TEST_ID)

    @property
    def _test_ds(self) -> Any:
        return self._test_dataset

    def fetch_docs(self, doc_ids: Iterable[str]) -> dict[str, str]:
        """Return `{doc_id: text}`, iterating the corpus with early termination.

        Deliberately *not* `docs_store()` as `MSMarcoDev` uses: at v2's 138M
        passages the store's one-time index build is the larger cost, so a scan
        that stops as soon as every requested id is found wins for the handful
        of lookups this path serves. Wall-clock depends on where the ids land in
        ir_datasets' iteration order.
        """
        target = set(doc_ids)
        found: dict[str, str] = {}
        with tqdm(total=len(target), desc=f"docs:{self.name}") as bar:
            for d in self._corpus_ds.docs_iter():
                if d.doc_id in target and d.doc_id not in found:
                    found[d.doc_id] = d.text
                    bar.update(1)
                    if len(found) == len(target):
                        break
        return found

    def materialize(self) -> None:
        if self.corpus_limit is None:
            raise ValueError(
                "materialize() needs a corpus_limit; use "
                "load_metadata() + fetch_docs() for targeted lookup"
            )
        qrels_df = self._qrels_frame(self._test_ds)
        queries_df = self._queries_frame(self._test_ds)
        judged_ids = set(qrels_df["doc_id"])
        limit = min(self.corpus_limit, len(judged_ids))

        rows: list[dict[str, str]] = []
        saved_ids: set[str] = set()
        for d in tqdm(
            self._corpus_ds.docs_iter(),
            total=self._corpus_ds.docs_count(),
            desc=f"materialize:{self.name}",
        ):
            if len(rows) >= limit:
                break
            if d.doc_id in judged_ids:
                rows.append({"doc_id": d.doc_id, "title": "", "text": d.text})
                saved_ids.add(d.doc_id)

        self._corpus_df = pd.DataFrame(rows, columns=CORPUS_COLUMNS)
        self._qrels_df = qrels_df[qrels_df["doc_id"].isin(saved_ids)].reset_index(
            drop=True
        )
        self._queries_df = queries_df[
            queries_df["query_id"].isin(set(self._qrels_df["query_id"]))
        ].reset_index(drop=True)


class MSMarcoDev(MaterializedDataset):
    """MS MARCO passage dev, materialized for a fixed query-id list on a
    realistic index (SPEC d38).

    The ids come from the composition — never sampled here. Their dev qrels
    are kept exactly, the judged passages are force-included, and the corpus
    is padded to `corpus_size` with uniform-random distractor passages
    (seeded). Uniform sampling preserves the collection's vocabulary/IDF
    profile: a judged-docs-only corpus is near-all answers, which inflates
    dense and starves sparse (d37g).

    First docs_store access downloads and indexes the msmarco-passage
    collection once; after that, materialization is random-access lookups.
    """

    name = "msmarco-passage-dev"

    _CORPUS_ID = "msmarco-passage"
    _TEST_ID = "msmarco-passage/dev/judged"

    def __init__(
        self,
        query_ids: Iterable[str] | None = None,
        corpus_size: int | None = None,
        seed: int = 0,
    ) -> None:
        """`query_ids` and `corpus_size` are only required by `materialize()`;
        callers that only need queries + qrels via `load_metadata()` can leave
        them None."""
        super().__init__()
        self.query_ids = None if query_ids is None else {str(q) for q in query_ids}
        self.corpus_size = corpus_size
        self.seed = seed
        self._corpus_ds = ir_datasets.load(self._CORPUS_ID)
        self._test_dataset = ir_datasets.load(self._TEST_ID)

    @property
    def _test_ds(self) -> Any:
        return self._test_dataset

    def fetch_docs(self, doc_ids: Iterable[str]) -> dict[str, str]:
        """Return `{doc_id: text}` via the corpus docs_store — random access,
        O(1) per lookup once the store is built."""
        store = self._corpus_ds.docs_store()
        return {
            did: store.get(did).text
            for did in tqdm(list(doc_ids), desc=f"docs:{self.name}", unit="doc")
        }

    def _distractor_ids(self, exclude: set[str], count: int) -> list[str]:
        """Uniform seeded draw over the collection's id space.

        v1 passage ids are the integers 0..docs_count()-1 as strings, so a
        draw over that range is a draw over the collection. Oversamples by
        `len(exclude)` — the worst case for collisions — then filters.
        """
        universe = self._corpus_ds.docs_count()
        drawn = random.Random(self.seed).sample(range(universe), count + len(exclude))
        kept = [str(i) for i in drawn if str(i) not in exclude]
        return kept[:count]

    def materialize(self) -> None:
        if self.query_ids is None or self.corpus_size is None:
            raise ValueError(
                "materialize() needs query_ids and corpus_size; use "
                "load_metadata() + fetch_docs() for targeted lookup"
            )
        queries_df = self._queries_frame(self._test_ds)
        queries_df = queries_df[
            queries_df["query_id"].astype(str).isin(self.query_ids)
        ].reset_index(drop=True)
        qrels_df = self._qrels_frame(self._test_ds)
        qrels_df = qrels_df[
            qrels_df["query_id"].isin(set(queries_df["query_id"]))
        ].reset_index(drop=True)

        relevant = list(dict.fromkeys(qrels_df["doc_id"]))
        distractors = self._distractor_ids(
            set(relevant), self.corpus_size - len(relevant)
        )
        texts = self.fetch_docs([*relevant, *distractors])
        # msmarco-passage docs carry no title field
        self._corpus_df = pd.DataFrame(
            [
                {"doc_id": doc_id, "title": "", "text": text}
                for doc_id, text in texts.items()
            ],
            columns=CORPUS_COLUMNS,
        )
        self._queries_df = queries_df
        self._qrels_df = qrels_df


class BeirDataset(RetrievalDataset):
    """A BEIR subset served on demand from `beir/<name>` + `beir/<name>/test`.

    Small enough that `corpus()` builds the frame per call — no materialize
    step and no in-memory snapshot.
    """

    def __init__(self) -> None:
        self._corpus_ds = ir_datasets.load(f"beir/{self.name}")
        self._test_ds = ir_datasets.load(f"beir/{self.name}/test")

    def corpus(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {"doc_id": d.doc_id, "title": d.title, "text": d.text}
                for d in tqdm(
                    self._corpus_ds.docs_iter(),
                    total=self._corpus_ds.docs_count(),
                    desc=f"corpus:{self.name}",
                    unit="doc",
                )
            ],
            columns=CORPUS_COLUMNS,
        )

    def queries(self) -> pd.DataFrame:
        return self._queries_frame(self._test_ds)

    def qrels(self) -> pd.DataFrame:
        return self._qrels_frame(self._test_ds)


class NFCorpus(BeirDataset):
    name = "nfcorpus"


class SciFact(BeirDataset):
    name = "scifact"
