"""Offline target artifacts for the corpus branch: a per-lane corpus profile
(scan stats + sampled-document extraction) and a per-query gold-document
profile (qrel-doc stats + query<->gold lexical overlap). Route-outcome rates
are fold-local and live in the training pipeline, not here."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from composition.floors import IDENTIFIER_SPANS, with_derived
from composition.mini_catalog import feature_columns
from hybrid_search_rrf_dataset.lanes import LANES
from hybrid_search_rrf_dataset.retrieval import SnapshotDataset

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
OUT_DIR = DATA_DIR / "encoder_router"

NL_SHARE = "natural_language_signal.natural_language_share"
DOC_STATS = {
    IDENTIFIER_SPANS: "identifier_spans",
    NL_SHARE: "nl_share",
}

EXTRACT_CHARS = 2_000
"""Documents are profiled by their head: spaCy caps input at 1M chars, and
the extracted signals are densities, stable under truncation."""

_TOKEN = re.compile(r"[a-z0-9]+")


def _source_name(lane: str) -> str:
    return LANES[lane].source.name if lane in LANES else lane


def _tokens(text: str) -> set[str]:
    return set(_TOKEN.findall(text.lower()))


def _doc_feature_rows(texts: list[str], extractor) -> pd.DataFrame:
    rows = [
        feature_columns(extractor.resolve(text[:EXTRACT_CHARS]))
        for text in texts
    ]
    frame = pd.DataFrame(rows).reindex(
        columns=sorted({column for row in rows for column in row})
    )
    return with_derived(frame.fillna(0.0))


def _doc_stat_means(features: pd.DataFrame, prefix: str) -> dict[str, float]:
    present = features.reindex(columns=list(DOC_STATS)).fillna(0.0)
    return {
        f"{prefix}.{name}": float(present[column].mean())
        for column, name in DOC_STATS.items()
    }


class CorpusProfile:
    """Per-lane corpus block: cheap scan stats over the whole corpus plus
    extractor stats over a fixed-seed document sample."""

    def __init__(
        self,
        lanes: tuple[str, ...],
        *,
        data_dir: Path = DATA_DIR,
        out_dir: Path = OUT_DIR,
        sample_size: int = 5_000,
        seed: int = 0,
    ) -> None:
        self._lanes = tuple(dict.fromkeys(lanes))
        self._data_dir = data_dir
        self._out_dir = out_dir
        self._sample_size = sample_size
        self._seed = seed

    @property
    def path(self) -> Path:
        return self._out_dir / "corpus_profile.parquet"

    def load(self) -> pd.DataFrame:
        return pd.read_parquet(self.path)

    def build(self, *, force: bool = False) -> pd.DataFrame:
        done = self.load() if self.path.exists() and not force else None
        have = set() if done is None else set(done["dataset"])
        todo = [lane for lane in self._lanes if lane not in have]
        if not todo:
            return done
        extractor = self._extractor()
        rows = [self._profile(lane, extractor) for lane in tqdm(todo)]
        merged = pd.concat(
            ([done] if done is not None else []) + [pd.DataFrame(rows)],
            ignore_index=True,
        )
        self._out_dir.mkdir(parents=True, exist_ok=True)
        merged.to_parquet(self.path, index=False)
        return merged

    def _extractor(self):
        from query_taxonomy.features import FeatureExtractor

        return FeatureExtractor(engines=None)

    def _profile(self, lane: str, extractor) -> dict[str, float | str]:
        corpus = SnapshotDataset(_source_name(lane), path=str(self._data_dir))
        texts = corpus.corpus()["text"].astype(str)
        rng = np.random.default_rng(self._seed)
        take = min(self._sample_size, len(texts))
        sample = texts.iloc[rng.permutation(len(texts))[:take]].tolist()
        words = [len(text.split()) for text in sample]
        all_tokens = [token for text in sample for token in _tokens(text)]
        row: dict[str, float | str] = {
            "dataset": lane,
            "corpus.doc_count_log10": float(np.log10(max(len(texts), 1))),
            "corpus.mean_words": float(np.mean(words)),
            "corpus.p90_words": float(np.percentile(words, 90)),
            "corpus.ttr": len(set(all_tokens)) / max(len(all_tokens), 1),
        }
        row.update(
            _doc_stat_means(_doc_feature_rows(sample, extractor), "corpus")
        )
        return row


class GoldDocProfile:
    """Per-query gold-doc block: extractor stats of the row's qrel documents
    at the lane's min_relevance, plus query<->gold lexical overlap."""

    def __init__(
        self,
        rows: pd.DataFrame,
        *,
        data_dir: Path = DATA_DIR,
        out_dir: Path = OUT_DIR,
    ) -> None:
        self._rows = rows[["dataset", "query_id", "query"]].drop_duplicates()
        self._data_dir = data_dir
        self._out_dir = out_dir

    @property
    def path(self) -> Path:
        return self._out_dir / "gold_doc_profile.parquet"

    def load(self) -> pd.DataFrame:
        return pd.read_parquet(self.path)

    def build(self, *, force: bool = False) -> pd.DataFrame:
        done = self.load() if self.path.exists() and not force else None
        have = set() if done is None else set(done["dataset"])
        todo = sorted(set(self._rows["dataset"]) - have)
        if not todo:
            return done
        extractor = self._extractor()
        parts = [self._profile_lane(lane, extractor) for lane in tqdm(todo)]
        merged = pd.concat(
            ([done] if done is not None else []) + parts, ignore_index=True
        )
        self._out_dir.mkdir(parents=True, exist_ok=True)
        merged.to_parquet(self.path, index=False)
        return merged

    def _extractor(self):
        from query_taxonomy.features import FeatureExtractor

        return FeatureExtractor(engines=None)

    def _profile_lane(self, lane: str, extractor) -> pd.DataFrame:
        wanted = self._rows[self._rows["dataset"] == lane]
        source = SnapshotDataset(_source_name(lane), path=str(self._data_dir))
        min_rel = LANES[lane].min_relevance if lane in LANES else 1
        qrels = source.qrels()
        qrels = qrels[
            (qrels["relevance"] >= min_rel)
            & qrels["query_id"].isin(set(wanted["query_id"]))
        ]
        texts = self._doc_texts(source, tuple(qrels["doc_id"].unique()))
        features = _doc_feature_rows(list(texts.values()), extractor)
        features.index = list(texts)
        records = [
            self._profile_query(row, qrels, texts, features)
            for row in wanted.itertuples(index=False)
        ]
        missing = sum(record["gold.doc_count"] == 0 for record in records)
        if missing:
            tqdm.write(f"[{lane}] {missing} queries without gold docs")
        return pd.DataFrame(records)

    def _doc_texts(
        self, source: SnapshotDataset, doc_ids: tuple[str, ...]
    ) -> dict[str, str]:
        corpus = source.corpus().set_index("doc_id")["text"]
        found = corpus.index.intersection(doc_ids)
        return corpus.loc[found].astype(str).to_dict()

    def _profile_query(
        self,
        row,
        qrels: pd.DataFrame,
        texts: dict[str, str],
        features: pd.DataFrame,
    ) -> dict[str, float | str]:
        doc_ids = [
            doc_id
            for doc_id in qrels.loc[
                qrels["query_id"] == row.query_id, "doc_id"
            ]
            if doc_id in texts
        ]
        record: dict[str, float | str] = {
            "dataset": row.dataset,
            "query_id": row.query_id,
            "gold.doc_count": float(len(doc_ids)),
        }
        if not doc_ids:
            record.update(dict.fromkeys(self._stat_keys(), np.nan))
            return record
        record.update(_doc_stat_means(features.loc[doc_ids], "gold"))
        record["gold.words"] = float(
            np.mean([len(texts[d].split()) for d in doc_ids])
        )
        query_tokens = _tokens(str(row.query))
        doc_tokens = set().union(*(_tokens(texts[d]) for d in doc_ids))
        record["gold.overlap"] = (
            len(query_tokens & doc_tokens) / len(query_tokens)
            if query_tokens
            else np.nan
        )
        return record

    def _stat_keys(self) -> list[str]:
        return [f"gold.{name}" for name in DOC_STATS.values()] + [
            "gold.words",
            "gold.overlap",
        ]
