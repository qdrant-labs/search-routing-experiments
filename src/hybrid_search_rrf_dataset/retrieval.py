from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

import ir_datasets
import pandas as pd
from tqdm.auto import tqdm


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

    def save(self, path: Path | str = Path("data")) -> None:
        out = Path(path) / self.name
        out.mkdir(parents=True, exist_ok=True)
        self.corpus().to_parquet(out / "corpus.parquet", index=False)
        self.queries().to_parquet(out / "queries.parquet", index=False)
        self.qrels().to_parquet(out / "qrels.parquet", index=False)


class TrecDL2022(RetrievalDataset):
    """TREC Deep Learning 2022 passage retrieval task (MS MARCO passage v2).

    The corpus is ~138M passages with no title field — corpus() returns
    title="" for every row. Loading the full corpus into memory is expensive;
    prefer iterating docs_iter() directly for large-scale indexing.
    """

    name = "trec-dl-2022"

    _CORPUS_ID = "msmarco-passage-v2"
    _TEST_ID = "msmarco-passage-v2/trec-dl-2022"

    def __init__(self, corpus_limit: int) -> None:
        self.corpus_limit = corpus_limit
        self._corpus_ds = ir_datasets.load(self._CORPUS_ID)
        self._test_ds = ir_datasets.load(self._TEST_ID)
        self._corpus_df: pd.DataFrame = pd.DataFrame(
            columns=["doc_id", "title", "text"]
        )
        self._queries_df: pd.DataFrame = pd.DataFrame(
            columns=["query_id", "text"]
        )
        self._qrels_df: pd.DataFrame = pd.DataFrame(
            columns=["query_id", "doc_id", "relevance"]
        )

    def corpus(self) -> pd.DataFrame:
        return self._corpus_df

    def queries(self) -> pd.DataFrame:
        return self._queries_df

    def qrels(self) -> pd.DataFrame:
        return self._qrels_df

    def materialize(self):
        qrels_df = pd.DataFrame(
            [{"query_id": r.query_id, "doc_id": r.doc_id, "relevance": r.relevance}
             for r in self._test_ds.qrels_iter()]
        )
        queries_df = pd.DataFrame(
            [{"query_id": q.query_id, "text": q.text}
             for q in self._test_ds.queries_iter()]
        )
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
                rows.append(d)
                saved_ids.add(d.doc_id)

        self._corpus_df = pd.DataFrame(rows)
        
        qrels_df = qrels_df[qrels_df["doc_id"].isin(list(saved_ids))].reset_index(drop=True)
        kept_query_ids = list(qrels_df["query_id"])
        queries_df = queries_df[
            queries_df["query_id"].isin(kept_query_ids)
        ].reset_index(drop=True)

        if not isinstance(qrels_df, pd.DataFrame) or not isinstance(queries_df, pd.DataFrame):
            raise RuntimeError("Query relationshup and queries to answers should be pandas dataframess")
        self._qrels_df = qrels_df
        self._queries_df = queries_df



    def save(self, path: Path | str = Path("data")) -> None:
        """Save a self-consistent corpus/queries/qrels snapshot.
        """
        out = Path(path) / self.name
        out.mkdir(parents=True, exist_ok=True)
        self._corpus_df.to_parquet(out / "corpus.parquet", index=False)
        self._queries_df.to_parquet(out / "queries.parquet", index=False)
        self._qrels_df.to_parquet(out / "qrels.parquet", index=False)


class NFCorpus(RetrievalDataset):
    name = "nfcorpus"

    _CORPUS_ID = "beir/nfcorpus"
    _TEST_ID = "beir/nfcorpus/test"

    def __init__(self) -> None:
        self._corpus_ds = ir_datasets.load(self._CORPUS_ID)
        self._test_ds = ir_datasets.load(self._TEST_ID)

    def corpus(self) -> pd.DataFrame:
        return pd.DataFrame(
            [{"doc_id": d.doc_id, "title": d.title, "text": d.text}
             for d in self._corpus_ds.docs_iter()]
        )

    def queries(self) -> pd.DataFrame:
        return pd.DataFrame(
            [{"query_id": q.query_id, "text": q.text}
             for q in self._test_ds.queries_iter()]
        )

    def qrels(self) -> pd.DataFrame:
        return pd.DataFrame(
            [{"query_id": r.query_id, "doc_id": r.doc_id, "relevance": r.relevance}
             for r in self._test_ds.qrels_iter()]
        )


class SciFact(RetrievalDataset):
    name = "scifact"

    _CORPUS_ID = "beir/scifact"
    _TEST_ID = "beir/scifact/test"

    def __init__(self) -> None:
        self._corpus_ds = ir_datasets.load(self._CORPUS_ID)
        self._test_ds = ir_datasets.load(self._TEST_ID)

    def corpus(self) -> pd.DataFrame:
        return pd.DataFrame(
            [{"doc_id": d.doc_id, "title": d.title, "text": d.text}
             for d in self._corpus_ds.docs_iter()]
        )

    def queries(self) -> pd.DataFrame:
        return pd.DataFrame(
            [{"query_id": q.query_id, "text": q.text}
             for q in self._test_ds.queries_iter()]
        )

    def qrels(self) -> pd.DataFrame:
        return pd.DataFrame(
            [{"query_id": r.query_id, "doc_id": r.doc_id, "relevance": r.relevance}
             for r in self._test_ds.qrels_iter()]
        )
