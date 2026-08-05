"""Lanes served by the ir_datasets catalog. Subclasses name their corpus
and test dataset ids; documents come from a docs_store or a stream."""

from __future__ import annotations

import random
from abc import ABC
from collections.abc import Iterable, Iterator
from typing import Any, ClassVar

import ir_datasets
import pandas as pd
from tqdm.auto import tqdm

from dataset_registry.core import DatasetName

from hybrid_search_rrf_dataset.retrieval.base import (
    CORPUS_COLUMNS,
    QREL_COLUMNS,
    QUERY_COLUMNS,
    MaterializedDataset,
    RetrievalDataset,
)


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

    subset: ClassVar[str] = ""
    """Upstream BEIR slug, when it differs from `name`."""

    def __init__(self) -> None:
        subset = self.subset or self.name
        self._corpus_ds = ir_datasets.load(f"beir/{subset}")
        self._test_ds = ir_datasets.load(f"beir/{subset}/test")

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
    name = DatasetName.BEIR_NFCORPUS.value
    subset = "nfcorpus"


class SciFact(BeirDataset):
    name = "scifact"


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


class AntiqueLane(IRDatasetsMaterialized):
    """ANTIQUE test: 200 non-factoid questions with 6,589 graded judgments
    over 403,666 Yahoo Answers passages (LANES pins min_relevance=3 — level
    2 is "does not answer the question"). Docs carry no title."""

    name = "antique"
    _CORPUS_ID = "antique"
    _TEST_ID = "antique/test"


class LotteLane(IRDatasetsMaterialized):
    """One LoTTE domain's `search` or `forum` test queries over that
    domain's corpus; qrels are the upvoted/accepted answer posts, binary.
    First access downloads the single LoTTE tarball (all domains)."""

    def __init__(self, domain: str, query_set: str) -> None:
        self._CORPUS_ID = f"lotte/{domain}/test"
        self._TEST_ID = f"lotte/{domain}/test/{query_set}"
        super().__init__()
        self.name = f"lotte-{domain}-{query_set}"


class OrcasLane(MaterializedDataset):
    """ORCAS clicks over the msmarco-document corpus — the click lane (SPEC
    d37k), and the only source that breaks the single-corpus concentration on
    the biggest archetype cells.

    Its judgments are clicks, so they are **positive-only** (every one of the
    18,823,602 pairs is relevance 1, "user click" — no judged-0 rows, no true
    negatives), **position-biased** (a click records what Bing already ranked
    high enough to be seen), and give **~1.8 judged docs per query** (18.8M
    pairs over 10,405,342 queries). That is the accepted cost of the one real
    query log we can label, not a defect to engineer around.

    Queries are the registry's cached ORCAS sample (100K, seed 0 — exactly the
    rows the feature table profiled, so the composition's ids join); qrels are
    streamed pair by pair and kept only for those ids, so the 18.8M-pair file
    is never held in memory. `query_ids` narrows further, like `MSMarcoDev`:
    pass the composition's ORCAS rows before `materialize()`, because the full
    100K sample's clicked docs (~165K) exceed the recipe's 100K ceiling and
    would leave the force-include set no distractor budget at all.
    """

    name = "orcas"  # must stay equal to DatasetName.ORCAS

    _IRDS_ID = "msmarco-document/orcas"

    def __init__(self, query_ids: Iterable[str] | None = None) -> None:
        super().__init__()
        self._ids = None if query_ids is None else {str(q) for q in query_ids}
        self._dataset = ir_datasets.load(self._IRDS_ID)

    def load_metadata(self) -> None:
        """Cached queries + their click qrels. The qrels file downloads once
        (separate from the queries file the registry already fetched) and is
        consumed as a stream."""
        # the registry owns the cached sample and its seed; imported here so
        # the retrieval stack keeps no module-level dependency on it
        from dataset_registry.irds import Orcas

        queries = [
            {"query_id": query.query_id, "text": query.text}
            for query in Orcas().sample_queries()
            if self._ids is None or query.query_id in self._ids
        ]
        if self._ids is not None and len(queries) < len(self._ids):
            raise ValueError(
                f"{self.name}: {len(self._ids) - len(queries):,} of "
                f"{len(self._ids):,} requested query_ids are outside the "
                "registry's cached sample — labeling them would silently "
                "drop rows."
            )
        wanted = {row["query_id"] for row in queries}
        self._queries_df = pd.DataFrame(queries, columns=QUERY_COLUMNS)
        self._qrels_df = pd.DataFrame(
            [
                {
                    "query_id": qrel.query_id,
                    "doc_id": str(qrel.doc_id),
                    "relevance": int(qrel.relevance),
                }
                for qrel in tqdm(
                    self._dataset.qrels_iter(),
                    desc=f"qrels:{self.name}",
                    unit="pair",
                )
                if qrel.query_id in wanted
            ],
            columns=QREL_COLUMNS,
        )

    def _iter_corpus(self) -> Iterator[dict[str, str]]:
        # msmarco-document docs carry url/title/body rather than `text`, and
        # they are full documents, not passages (first pass downloads ~8GB)
        for doc in self._dataset.docs_iter():
            yield {
                "doc_id": str(doc.doc_id),
                "title": doc.title or "",
                "text": doc.body,
            }
