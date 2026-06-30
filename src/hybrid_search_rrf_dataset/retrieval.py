from __future__ import annotations

from pathlib import Path

import ir_datasets
import pandas as pd


class SciFact:
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

    def save(self, path: Path | str = Path("data")) -> None:
        out = Path(path) / "scifact"
        out.mkdir(parents=True, exist_ok=True)
        self.corpus().to_parquet(out / "corpus.parquet", index=False)
        self.queries().to_parquet(out / "queries.parquet", index=False)
        self.qrels().to_parquet(out / "qrels.parquet", index=False)
