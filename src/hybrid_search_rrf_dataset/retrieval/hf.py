"""Lanes loaded through `datasets.load_dataset` — HF repos, raw jsonl over
HTTP, and one local file. Guessed layouts fail loud via `_expect`."""

from __future__ import annotations

import random
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

import pandas as pd
from datasets import load_dataset

from dataset_registry.wave2 import (
    GOOAQ_JSONL,
    SCIRGEN_REPO,
    is_english,
    scirgen_questions,
    scirgen_rows,
)

from hybrid_search_rrf_dataset.retrieval.base import (
    QREL_COLUMNS,
    QUERY_COLUMNS,
    MaterializedDataset,
)


def _expect(row: dict[Any, Any], keys: tuple[str, ...], source: str) -> None:
    """Fail loud on schema drift in a guessed HF layout (SPEC d39: qrels
    dialects are verified at first fetch), naming the real schema."""
    missing = [key for key in keys if key not in row]
    if missing:
        raise ValueError(f"{source}: missing {missing}; row has {sorted(row)}")


def _beir_qrels(data_files: str, reader: str = "csv") -> pd.DataFrame:
    """BEIR-layout qrels (`query-id` / `corpus-id` / `score`) — TSV by
    default, `reader="json"` for repos that ship them as JSONL (LIMIT)."""
    kwargs = {"delimiter": "\t"} if reader == "csv" else {}
    rows = load_dataset(
        reader, data_files=data_files, split="train", streaming=True, **kwargs
    )
    records = []
    for row in rows:
        _expect(row, ("query-id", "corpus-id", "score"), data_files)
        records.append(
            {
                "query_id": str(row["query-id"]),
                "doc_id": str(row["corpus-id"]),
                "relevance": int(row["score"]),
            }
        )
    return pd.DataFrame(records, columns=QREL_COLUMNS)


class RarbLane(MaterializedDataset):
    """One RAR-b pooled lane, streamed raw from its BEIR-layout repo (the
    repos carry a legacy loading script that datasets >= 4 refuses to
    run — same workaround as the registry's query side)."""

    _POOLS = {"math": "math-pooled", "code": "humanevalpack-mbpp-pooled"}

    def __init__(self, pool: str) -> None:
        super().__init__()
        self.name = f"rarb-{pool}"
        self._base = f"hf://datasets/RAR-b/{self._POOLS[pool]}"

    def load_metadata(self) -> None:
        queries = load_dataset(
            "json",
            data_files=f"{self._base}/queries.jsonl",
            split="train",
            streaming=True,
        )
        self._queries_df = pd.DataFrame(
            [{"query_id": str(r["_id"]), "text": r["text"]} for r in queries],
            columns=QUERY_COLUMNS,
        )
        self._qrels_df = _beir_qrels(f"{self._base}/qrels/test.tsv")

    def _iter_corpus(self) -> Iterator[dict[str, str]]:
        rows = load_dataset(
            "json",
            data_files=f"{self._base}/corpus.jsonl",
            split="train",
            streaming=True,
        )
        for row in rows:
            yield {
                "doc_id": str(row["_id"]),
                "title": row.get("title") or "",
                "text": row["text"],
            }


class BrightLane(MaterializedDataset):
    """One BRIGHT split. `examples` carries per-query `gold_ids` (the
    binary qrels) and `excluded_ids` — documents that must not be scored
    against that query (near-duplicates/leakage per the BRIGHT paper).

    Excluded pairs persist to `excluded.parquet` beside the qrels; the
    labeling stage applies them at scoring time. Corpus-level dropping
    would be wrong: a doc excluded for one query can be gold for another.
    """

    def __init__(self, split: str) -> None:
        super().__init__()
        self.split = split
        self.name = f"bright-{split.replace('_', '-')}"
        self._excluded_df = pd.DataFrame(columns=["query_id", "doc_id"])

    def load_metadata(self) -> None:
        rows = load_dataset(
            "xlangai/BRIGHT", "examples", split=self.split, streaming=True
        )
        queries: list[dict[str, str]] = []
        gold: list[dict[str, Any]] = []
        excluded: list[dict[str, str]] = []
        for row in rows:
            _expect(
                row,
                ("id", "query", "gold_ids", "excluded_ids"),
                f"BRIGHT examples/{self.split}",
            )
            qid = str(row["id"])
            queries.append({"query_id": qid, "text": row["query"]})
            gold.extend(
                {"query_id": qid, "doc_id": str(d), "relevance": 1}
                for d in row["gold_ids"]
            )
            # BRIGHT pads some splits' excluded_ids with "N/A" placeholders
            excluded.extend(
                {"query_id": qid, "doc_id": str(d)}
                for d in row["excluded_ids"]
                if d and d != "N/A"
            )
        self._queries_df = pd.DataFrame(queries, columns=QUERY_COLUMNS)
        self._qrels_df = pd.DataFrame(gold, columns=QREL_COLUMNS)
        self._excluded_df = pd.DataFrame(excluded, columns=["query_id", "doc_id"])

    def save_metadata(self, path: Path | str = Path("data")) -> None:
        super().save_metadata(path)
        self._excluded_df.to_parquet(
            Path(path) / self.name / "excluded.parquet", index=False
        )

    def excluded(self) -> pd.DataFrame:
        return self._excluded_df

    def _iter_corpus(self) -> Iterator[dict[str, str]]:
        rows = load_dataset(
            "xlangai/BRIGHT", "documents", split=self.split, streaming=True
        )
        for row in rows:
            yield {"doc_id": str(row["id"]), "title": "", "text": row["content"]}


class CrumbLane(MaterializedDataset):
    """One CRUMB task. No qrels config exists — judgments ride on each
    `evaluation_queries` row (verified 2026-07-29: `passage_qrels` holds
    `{'id': 'NCT00002569:0', 'label': 1.0}` dicts, including judged-0
    entries; `passage_binary_qrels` is the variant some tasks use). Ids key
    into the `passage_corpus` config — we stay at passage granularity like
    every other lane, ignoring `full_document_qrels`/`use_max_p` (CRUMB's
    own doc-level MaxP eval)."""

    def __init__(self, task: str) -> None:
        super().__init__()
        self.task = task
        self.name = f"crumb-{task.replace('_', '-')}"

    def load_metadata(self) -> None:
        rows = load_dataset(
            "jfkback/crumb", "evaluation_queries", split=self.task, streaming=True
        )
        queries: list[dict[str, str]] = []
        qrels: list[dict[str, Any]] = []
        for row in rows:
            _expect(
                row,
                ("query_id", "query_content", "passage_qrels", "passage_binary_qrels"),
                f"crumb evaluation_queries/{self.task}",
            )
            qid = str(row["query_id"])
            queries.append({"query_id": qid, "text": row["query_content"]})
            judgments = row["passage_qrels"] or row["passage_binary_qrels"]
            if not judgments:
                raise ValueError(
                    f"crumb {self.task}: query {qid} carries no passage-level "
                    "judgments — decide granularity before labeling this task."
                )
            qrels.extend(
                {"query_id": qid, "doc_id": str(j["id"]), "relevance": int(j["label"])}
                for j in judgments
            )
        self._queries_df = pd.DataFrame(queries, columns=QUERY_COLUMNS)
        self._qrels_df = pd.DataFrame(qrels, columns=QREL_COLUMNS)

    def _iter_corpus(self) -> Iterator[dict[str, str]]:
        rows = load_dataset(
            "jfkback/crumb", "passage_corpus", split=self.task, streaming=True
        )
        for row in rows:
            yield {
                "doc_id": str(row["document_id"]),
                "title": "",
                "text": row["document_content"],
            }


class QuestLane(MaterializedDataset):
    """QUEST validation+test. Rows carry no id — ids are synthesized as
    `{split}-{index}` in stream order, matching the registry's convention
    exactly so the composition's ids join. Relevance is the row's
    attributed wiki pages; QUEST doc_ids are page titles.

    The corpus is not in the HF repo (configs: `main` only) — it streams
    from the paper's canonical GCS release (~325K wiki docs, so the d39e
    cap fires)."""

    name = "quest"
    _SPLITS = ("validation", "test")
    _DOCS_URL = "https://storage.googleapis.com/gresearch/quest/documents.jsonl"

    def load_metadata(self) -> None:
        queries: list[dict[str, str]] = []
        qrels: list[dict[str, Any]] = []
        for split in self._SPLITS:
            rows = load_dataset("cmalaviya/quest", "main", split=split, streaming=True)
            for index, row in enumerate(rows):
                _expect(row, ("query", "docs"), f"quest main/{split}")
                qid = f"{split}-{index}"
                queries.append({"query_id": qid, "text": row["query"]})
                qrels.extend(
                    {"query_id": qid, "doc_id": str(title), "relevance": 1}
                    for title in row["docs"]
                )
        self._queries_df = pd.DataFrame(queries, columns=QUERY_COLUMNS)
        self._qrels_df = pd.DataFrame(qrels, columns=QREL_COLUMNS)

    def _iter_corpus(self) -> Iterator[dict[str, str]]:
        rows = load_dataset(
            "json", data_files=self._DOCS_URL, split="train", streaming=True
        )
        for row in rows:
            title = str(row["title"])
            yield {"doc_id": title, "title": title, "text": row["text"]}


class LimitLane(MaterializedDataset):
    """LIMIT stress set from the google-deepmind repo's raw BEIR-layout
    files — the lane where dense fails by construction."""

    name = "limit"
    _BASE = "https://raw.githubusercontent.com/google-deepmind/limit/main/data/limit"

    def load_metadata(self) -> None:
        queries = load_dataset(
            "json",
            data_files=f"{self._BASE}/queries.jsonl",
            split="train",
            streaming=True,
        )
        self._queries_df = pd.DataFrame(
            [{"query_id": str(r["_id"]), "text": r["text"]} for r in queries],
            columns=QUERY_COLUMNS,
        )
        self._qrels_df = _beir_qrels(f"{self._BASE}/qrels.jsonl", reader="json")

    def _iter_corpus(self) -> Iterator[dict[str, str]]:
        rows = load_dataset(
            "json",
            data_files=f"{self._BASE}/corpus.jsonl",
            split="train",
            streaming=True,
        )
        for row in rows:
            yield {
                "doc_id": str(row["_id"]),
                "title": row.get("title") or "",
                "text": row["text"],
            }


# --- wave 2 (docs/datasets.md) -------------------------------------------

FRESHSTACK_QUERIES_REPO = "freshstack/queries-oct-2024"
FRESHSTACK_CORPUS_REPO = "freshstack/corpus-oct-2024"
WEBFAQ_REPO = "PaDaS-Lab/webfaq-retrieval"
WEBFAQ_LANGUAGE = "eng"
CLERC_TRAIN = "hf://datasets/jhu-clsp/CLERC/teva_train_dir/train_data.jsonl.gz"

def _sampled(rows: Iterator[dict[str, Any]], size: int, seed: int = 0) -> list[dict]:
    """Seeded uniform reservoir over a stream of unknown length — the
    query-side twin of `materialize()`'s corpus reservoir."""
    rng = random.Random(seed)
    kept: list[dict] = []
    for seen, row in enumerate(rows):
        if len(kept) < size:
            kept.append(row)
            continue
        slot = rng.randrange(seen + 1)
        if slot < size:
            kept[slot] = row
    return kept


class FreshStackLane(MaterializedDataset):
    """One FreshStack topic: StackOverflow questions over that framework's doc
    chunks. Judgments ride on the query rows as per-nugget document lists, so
    a query's qrels are the union of its nuggets' verdicts."""

    def __init__(self, topic: str) -> None:
        super().__init__()
        self.topic = topic
        self.name = f"freshstack-{topic}"

    # the two repos disagree on split name: queries ship as `test`, corpus as
    # `train` — same topic configs on both
    def _config_rows(self, repo: str, split: str) -> Iterator[dict[str, Any]]:
        yield from load_dataset(repo, self.topic, split=split, streaming=True)

    def load_metadata(self) -> None:
        queries: list[dict[str, str]] = []
        judged: dict[tuple[str, str], int] = {}
        for row in self._config_rows(FRESHSTACK_QUERIES_REPO, "test"):
            _expect(
                row,
                ("query_id", "query_title", "query_text", "nuggets"),
                f"freshstack queries/{self.topic}",
            )
            qid = str(row["query_id"])
            title = (row["query_title"] or "").strip()
            body = (row["query_text"] or "").strip()
            queries.append({"query_id": qid, "text": f"{title}\n{body}".strip()})
            for nugget in row["nuggets"]:
                # a doc supporting any nugget is relevant to the query; judged-0
                # pairs are kept because few lanes ship true negatives at all
                for doc_id in nugget["non_relevant_corpus_ids"]:
                    judged.setdefault((qid, str(doc_id)), 0)
                for doc_id in nugget["relevant_corpus_ids"]:
                    judged[(qid, str(doc_id))] = 1
        self._queries_df = pd.DataFrame(queries, columns=QUERY_COLUMNS)
        self._qrels_df = pd.DataFrame(
            [
                {"query_id": qid, "doc_id": doc_id, "relevance": score}
                for (qid, doc_id), score in judged.items()
            ],
            columns=QREL_COLUMNS,
        )

    def _iter_corpus(self) -> Iterator[dict[str, str]]:
        for row in self._config_rows(FRESHSTACK_CORPUS_REPO, "train"):
            _expect(row, ("_id", "text"), f"freshstack corpus/{self.topic}")
            yield {
                "doc_id": str(row["_id"]),
                "title": row.get("title") or "",
                "text": row["text"],
            }


class WebFaqLane(MaterializedDataset):
    """WebFAQ English retrieval slice: FAQ questions with the answer
    passages they were mined beside. The `<lang>-queries` / `-corpus` /
    `-qrels` config naming and the BEIR field names are unverified."""

    name = f"webfaq-{WEBFAQ_LANGUAGE}"

    def _config_rows(self, kind: str) -> Iterator[dict[str, Any]]:
        config = f"{WEBFAQ_LANGUAGE}-{kind}"
        for rows in load_dataset(WEBFAQ_REPO, config, streaming=True).values():
            yield from rows

    def load_metadata(self) -> None:
        queries: list[dict[str, str]] = []
        for row in self._config_rows("queries"):
            _expect(row, ("_id", "text"), f"webfaq {WEBFAQ_LANGUAGE}-queries")
            qid = str(row["_id"])
            if self.query_ids is not None and qid not in self.query_ids:
                continue
            queries.append({"query_id": qid, "text": row["text"]})
        wanted = {q["query_id"] for q in queries}
        qrels: list[dict[str, Any]] = []
        for row in self._config_rows("qrels"):
            _expect(
                row,
                ("query-id", "corpus-id", "score"),
                f"webfaq {WEBFAQ_LANGUAGE}-qrels",
            )
            qid = str(row["query-id"])
            if qid not in wanted:
                continue
            qrels.append(
                {
                    "query_id": qid,
                    "doc_id": str(row["corpus-id"]),
                    "relevance": int(row["score"]),
                }
            )
        self._queries_df = pd.DataFrame(queries, columns=QUERY_COLUMNS)
        self._qrels_df = pd.DataFrame(qrels, columns=QREL_COLUMNS)

    def _iter_corpus(self) -> Iterator[dict[str, str]]:
        for row in self._config_rows("corpus"):
            _expect(row, ("_id", "text"), f"webfaq {WEBFAQ_LANGUAGE}-corpus")
            yield {
                "doc_id": str(row["_id"]),
                "title": row.get("title") or "",
                "text": row["text"],
            }


class ScirgenGeoLane(MaterializedDataset):
    """ScIRGen-Geo English slice: dataset-seeking questions whose gold document
    is the geoscience metadata record they were written from. A record carries
    several question categories, so it is one document to many queries."""

    name = "scirgen-geo-en"

    def _rows(self) -> Iterator[dict[str, Any]]:
        for row in scirgen_rows():
            _expect(row, ("context",), SCIRGEN_REPO)
            yield row

    def load_metadata(self) -> None:
        queries: list[dict[str, str]] = []
        qrels: list[dict[str, Any]] = []
        for row in self._rows():
            record = str(row["id"])
            for query_id, text in scirgen_questions(row):
                queries.append({"query_id": query_id, "text": text})
                qrels.append(
                    {"query_id": query_id, "doc_id": record, "relevance": 1}
                )
        self._queries_df = pd.DataFrame(queries, columns=QUERY_COLUMNS)
        self._qrels_df = pd.DataFrame(qrels, columns=QREL_COLUMNS)

    def _iter_corpus(self) -> Iterator[dict[str, str]]:
        for row in self._rows():
            metadata = (row["context"] or {}).get("metadata") or {}
            yield {
                "doc_id": str(row["id"]),
                "title": (metadata.get("titleEn") or "").strip(),
                "text": (metadata.get("description") or "").strip(),
            }


class ClercLane(MaterializedDataset):
    """CLERC legal case retrieval from its Tevatron-format training file:
    every row carries the query, its cited-case passages and 20 hard
    negatives, so queries, qrels and corpus all come from one stream.

    The 327,414 rows would force more judged docs than any corpus budget
    allows, so `sample` caps the query set (seeded, uniform) unless the
    composition passes the ids it actually wants.
    """

    name = "clerc"

    def __init__(
        self,
        sample: int = 50_000,
        query_ids: Iterable[str] | None = None,
        seed: int = 0,
    ) -> None:
        super().__init__()
        self.sample = sample
        self.query_ids = None if query_ids is None else {str(q) for q in query_ids}
        self.seed = seed

    def _rows(self) -> Iterator[dict[str, Any]]:
        rows = load_dataset(
            "json", data_files=CLERC_TRAIN, split="train", streaming=True
        )
        for row in rows:
            _expect(row, ("query_id", "query", "positive_passages"), CLERC_TRAIN)
            yield row

    def _judgments(self) -> Iterator[dict[str, Any]]:
        """Query plus its gold doc_ids, dropping the passage texts — a
        reservoir of whole rows would hold gigabytes of legal prose."""
        for row in self._rows():
            yield {
                "query_id": str(row["query_id"]),
                "text": row["query"],
                "doc_ids": [str(p["docid"]) for p in row["positive_passages"]],
            }

    def load_metadata(self) -> None:
        wanted = (
            [row for row in self._judgments() if row["query_id"] in self.query_ids]
            if self.query_ids is not None
            else _sampled(self._judgments(), self.sample, self.seed)
        )
        self._queries_df = pd.DataFrame(
            [{"query_id": r["query_id"], "text": r["text"]} for r in wanted],
            columns=QUERY_COLUMNS,
        )
        self._qrels_df = pd.DataFrame(
            [
                {"query_id": row["query_id"], "doc_id": doc_id, "relevance": 1}
                for row in wanted
                for doc_id in row["doc_ids"]
            ],
            columns=QREL_COLUMNS,
        )

    def _iter_corpus(self) -> Iterator[dict[str, str]]:
        for row in self._rows():
            passages = [
                *row["positive_passages"],
                *row.get("negative_passages", ()),
            ]
            for passage in passages:
                yield {
                    "doc_id": str(passage["docid"]),
                    "title": passage.get("title") or "",
                    "text": passage["text"],
                }


class GooaqLane(MaterializedDataset):
    """GooAQ as a retrieval lane: the row's snippet `answer` IS its gold
    document (msmarco's regime), so qrels are one self-pair per query.

    Usable rows are the 3,032,114 English ones carrying an `answer`; the
    query set is capped like CLERC's, and the other ~3M answers are the
    distractor pool the recipe draws from.
    """

    name = "gooaq"

    def __init__(
        self,
        sample: int = 50_000,
        query_ids: Iterable[str] | None = None,
        seed: int = 0,
    ) -> None:
        super().__init__()
        self.sample = sample
        self.query_ids = None if query_ids is None else {str(q) for q in query_ids}
        self.seed = seed

    def _usable_rows(self) -> Iterator[dict[str, Any]]:
        rows = load_dataset(
            "json", data_files=str(GOOAQ_JSONL), split="train", streaming=True
        )
        for row in rows:
            if row["answer"] and is_english(row["question"]):
                yield row

    def _questions(self) -> Iterator[dict[str, Any]]:
        for row in self._usable_rows():
            yield {"query_id": str(row["id"]), "text": row["question"]}

    def load_metadata(self) -> None:
        wanted = (
            [row for row in self._questions() if row["query_id"] in self.query_ids]
            if self.query_ids is not None
            else _sampled(self._questions(), self.sample, self.seed)
        )
        self._queries_df = pd.DataFrame(wanted, columns=QUERY_COLUMNS)
        self._qrels_df = pd.DataFrame(
            [
                {
                    "query_id": row["query_id"],
                    "doc_id": row["query_id"],
                    "relevance": 1,
                }
                for row in wanted
            ],
            columns=QREL_COLUMNS,
        )

    def _iter_corpus(self) -> Iterator[dict[str, str]]:
        for row in self._usable_rows():
            yield {"doc_id": str(row["id"]), "title": "", "text": row["answer"]}
