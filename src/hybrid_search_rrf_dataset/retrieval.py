from __future__ import annotations

import random
from abc import ABC, abstractmethod
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any, ClassVar, NamedTuple

import ir_datasets
import pandas as pd
from datasets import load_dataset
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
        for doc in tqdm(
            self._iter_corpus(), desc=f"materialize:{self.name}", unit="doc"
        ):
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


class IRDatasetsMaterialized(MaterializedDataset, ABC):
    """Materialized lanes served by the ir_datasets catalog.

    Subclasses set `_CORPUS_ID` / `_TEST_ID`; metadata comes from the test
    dataset, documents from the corpus dataset's `docs_store` (random
    access, O(1) per lookup once the store is built).
    """

    _CORPUS_ID: str
    _TEST_ID: str

    def __init__(self) -> None:
        super().__init__()
        self._corpus_ds = ir_datasets.load(self._CORPUS_ID)
        self._test_dataset = ir_datasets.load(self._TEST_ID)

    @property
    def _test_ds(self) -> Any:
        return self._test_dataset

    def load_metadata(self) -> None:
        self._qrels_df = self._qrels_frame(self._test_ds)
        self._queries_df = self._queries_frame(self._test_ds)

    def fetch_docs(self, doc_ids: Iterable[str]) -> dict[str, str]:
        """Return `{doc_id: text}` for a specific list of doc_ids."""
        store = self._corpus_ds.docs_store()
        return {
            did: store.get(did).text
            for did in tqdm(list(doc_ids), desc=f"docs:{self.name}", unit="doc")
        }

    def _iter_corpus(self) -> Iterator[dict[str, str]]:
        for doc in self._corpus_ds.docs_iter():
            yield {
                "doc_id": str(doc.doc_id),
                "title": getattr(doc, "title", "") or "",
                "text": doc.text,
            }


class TrecDL2022(IRDatasetsMaterialized):
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

    def fetch_docs(self, doc_ids: Iterable[str]) -> dict[str, str]:
        """Return `{doc_id: text}`, iterating the corpus with early termination.

        Deliberately *not* the base `docs_store()` path: at v2's 138M passages
        the store's one-time index build is the larger cost, so a scan that
        stops as soon as every requested id is found wins for the handful of
        lookups this path serves. Wall-clock depends on where the ids land in
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


class MSMarcoDev(IRDatasetsMaterialized):
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


def _expect(row: dict, keys: tuple[str, ...], source: str) -> None:
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


class DBPediaLane(IRDatasetsMaterialized):
    """BEIR DBpedia-Entity test slice — graded 0/1/2 qrels (d39d keeps
    min_relevance=1; the zero-retrieval sensitivity cell is the escape
    hatch). First metadata access downloads the BEIR dbpedia zip (~2GB,
    corpus included — pass 2 pays nothing new)."""

    name = "dbpedia-entity"
    _CORPUS_ID = "beir/dbpedia-entity"
    _TEST_ID = "beir/dbpedia-entity/test"


class MiraclLane(IRDatasetsMaterialized):
    """MIRACL en dev. Qrels are a small standalone download; the ~33M-doc
    corpus stays untouched until pass 2's capped recipe."""

    name = "miracl-en-dev"
    _CORPUS_ID = "miracl/en"
    _TEST_ID = "miracl/en/dev"


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
