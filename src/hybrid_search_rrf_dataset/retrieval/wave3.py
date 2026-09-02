"""Wave-3 retrieval lanes with source-specific schema normalization."""

from __future__ import annotations

import hashlib
from collections.abc import Iterator
from typing import Any

import pandas as pd
from datasets import load_dataset

from dataset_registry.wave3 import (
    CAST_2020,
    TECHQA_REPO,
    TECHQA_SPLITS,
    TOUCHE_2020,
    serialize_cast_history,
)

from hybrid_search_rrf_dataset.retrieval.base import (
    QREL_COLUMNS,
    QUERY_COLUMNS,
    MaterializedDataset,
)
from hybrid_search_rrf_dataset.retrieval.irds import IRDatasetsMaterialized

ESCI_BASE = (
    "https://media.githubusercontent.com/media/amazon-science/esci-data/"
    "main/shopping_queries_dataset"
)
ESCI_EXAMPLES = f"{ESCI_BASE}/shopping_queries_dataset_examples.parquet"
ESCI_PRODUCTS = f"{ESCI_BASE}/shopping_queries_dataset_products.parquet"
WANDS_BASE = "https://raw.githubusercontent.com/wayfair/WANDS/main/dataset"
FINDER_REPO = "Linq-AI-Research/FinDER"


def _expect(row: dict[str, Any], keys: tuple[str, ...], source: str) -> None:
    missing = [key for key in keys if key not in row]
    if missing:
        raise ValueError(f"{source}: missing {missing}; row has {sorted(row)}")


def _field(label: str, value: Any) -> str | None:
    text = str(value).strip() if value is not None else ""
    return f"{label}: {text}" if text else None


class AmazonEsciLane(MaterializedDataset):
    """English reduced ESCI task-1 queries over the US product catalog."""

    name = "amazon-esci-en-hard"
    _RELEVANCE = {"E": 2, "S": 1, "C": 0, "I": 0}

    def _examples(self) -> Iterator[dict[str, Any]]:
        yield from load_dataset(
            "parquet",
            data_files=ESCI_EXAMPLES,
            split="train",
            streaming=True,
        )

    def _products(self) -> Iterator[dict[str, Any]]:
        yield from load_dataset(
            "parquet",
            data_files=ESCI_PRODUCTS,
            split="train",
            streaming=True,
        )

    def load_metadata(self) -> None:
        queries: dict[str, str] = {}
        qrels: list[dict[str, str | int]] = []
        required = (
            "query_id",
            "query",
            "product_id",
            "product_locale",
            "esci_label",
            "small_version",
        )
        for row in self._examples():
            _expect(row, required, ESCI_EXAMPLES)
            if row["product_locale"] != "us" or int(row["small_version"]) != 1:
                continue
            query_id = str(row["query_id"])
            if self.query_ids is not None and query_id not in self.query_ids:
                continue
            label = str(row["esci_label"])
            if label not in self._RELEVANCE:
                raise ValueError(f"{ESCI_EXAMPLES}: unknown ESCI label {label!r}")
            text = str(row["query"])
            previous = queries.setdefault(query_id, text)
            if previous != text:
                raise ValueError(
                    f"{ESCI_EXAMPLES}: query {query_id} has conflicting text"
                )
            qrels.append(
                {
                    "query_id": query_id,
                    "doc_id": str(row["product_id"]),
                    "relevance": self._RELEVANCE[label],
                }
            )
        self._queries_df = pd.DataFrame(
            [
                {"query_id": query_id, "text": text}
                for query_id, text in queries.items()
            ],
            columns=QUERY_COLUMNS,
        )
        self._qrels_df = pd.DataFrame(qrels, columns=QREL_COLUMNS)

    def _iter_corpus(self) -> Iterator[dict[str, str]]:
        required = (
            "product_id",
            "product_locale",
            "product_title",
            "product_brand",
            "product_color",
            "product_bullet_point",
            "product_description",
        )
        for row in self._products():
            _expect(row, required, ESCI_PRODUCTS)
            if row["product_locale"] != "us":
                continue
            parts = [
                _field("Brand", row["product_brand"]),
                _field("Color", row["product_color"]),
                _field("Bullet points", row["product_bullet_point"]),
                _field("Description", row["product_description"]),
            ]
            yield {
                "doc_id": str(row["product_id"]),
                "title": str(row["product_title"] or "").strip(),
                "text": "\n".join(part for part in parts if part),
            }

class WandsLane(MaterializedDataset):
    """Wayfair product-search queries with deep human judgments."""

    name = "wands"
    _RELEVANCE = {"Exact": 2, "Partial": 1, "Irrelevant": 0}

    def _rows(self, filename: str) -> Iterator[dict[str, Any]]:
        yield from load_dataset(
            "csv",
            data_files=f"{WANDS_BASE}/{filename}",
            delimiter="\t",
            split="train",
            streaming=True,
        )

    def load_metadata(self) -> None:
        queries: list[dict[str, str]] = []
        wanted: set[str] = set()
        for row in self._rows("query.csv"):
            _expect(row, ("query_id", "query"), f"{WANDS_BASE}/query.csv")
            query_id = str(row["query_id"])
            if self.query_ids is not None and query_id not in self.query_ids:
                continue
            queries.append({"query_id": query_id, "text": str(row["query"])})
            wanted.add(query_id)

        qrels: list[dict[str, str | int]] = []
        for row in self._rows("label.csv"):
            _expect(
                row,
                ("query_id", "product_id", "label"),
                f"{WANDS_BASE}/label.csv",
            )
            query_id = str(row["query_id"])
            if query_id not in wanted:
                continue
            label = str(row["label"])
            if label not in self._RELEVANCE:
                raise ValueError(f"WANDS: unknown relevance label {label!r}")
            qrels.append(
                {
                    "query_id": query_id,
                    "doc_id": str(row["product_id"]),
                    "relevance": self._RELEVANCE[label],
                }
            )
        self._queries_df = pd.DataFrame(queries, columns=QUERY_COLUMNS)
        self._qrels_df = pd.DataFrame(qrels, columns=QREL_COLUMNS)

    def _iter_corpus(self) -> Iterator[dict[str, str]]:
        required = (
            "product_id",
            "product_name",
            "product_class",
            "category hierarchy",
            "product_description",
            "product_features",
        )
        for row in self._rows("product.csv"):
            _expect(row, required, f"{WANDS_BASE}/product.csv")
            parts = [
                _field("Product class", row["product_class"]),
                _field("Category hierarchy", row["category hierarchy"]),
                _field("Description", row["product_description"]),
                _field("Features", row["product_features"]),
            ]
            yield {
                "doc_id": str(row["product_id"]),
                "title": str(row["product_name"] or "").strip(),
                "text": "\n".join(part for part in parts if part),
            }


def _finder_passage(text: Any) -> tuple[str, str]:
    normalized = " ".join(str(text).split())
    if not normalized:
        raise ValueError("FinDER: reference passage is empty")
    digest = hashlib.sha256(normalized.encode()).hexdigest()[:24]
    return f"finder-{digest}", normalized


class FinderLane(MaterializedDataset):
    """FinDER finance queries with reference passages as the gold corpus."""

    name = "finder"

    def _rows(self) -> Iterator[dict[str, Any]]:
        yield from load_dataset(FINDER_REPO, split="train", streaming=True)

    def load_metadata(self) -> None:
        queries: list[dict[str, str]] = []
        qrels: list[dict[str, str | int]] = []
        seen_qrels: set[tuple[str, str]] = set()
        for row in self._rows():
            _expect(
                row,
                ("_id", "text", "reasoning", "category", "references"),
                FINDER_REPO,
            )
            query_id = str(row["_id"])
            if self.query_ids is not None and query_id not in self.query_ids:
                continue
            queries.append({"query_id": query_id, "text": str(row["text"])})
            for reference in row["references"]:
                doc_id, _ = _finder_passage(reference)
                pair = (query_id, doc_id)
                if pair in seen_qrels:
                    continue
                seen_qrels.add(pair)
                qrels.append(
                    {"query_id": query_id, "doc_id": doc_id, "relevance": 1}
                )
        self._queries_df = pd.DataFrame(queries, columns=QUERY_COLUMNS)
        self._qrels_df = pd.DataFrame(qrels, columns=QREL_COLUMNS)

    def _iter_corpus(self) -> Iterator[dict[str, str]]:
        seen: set[str] = set()
        for row in self._rows():
            _expect(row, ("references",), FINDER_REPO)
            for reference in row["references"]:
                doc_id, text = _finder_passage(reference)
                if doc_id in seen:
                    continue
                seen.add(doc_id)
                yield {"doc_id": doc_id, "title": "", "text": text}


class TrecCast2020HistoryLane(IRDatasetsMaterialized):
    """CAsT 2020 judged turns with prior utterances serialized into the query."""

    name = "trec-cast-2020-history"
    _CORPUS_ID = "trec-cast/v1/2020"
    _TEST_ID = CAST_2020

    def load_metadata(self) -> None:
        rows = [
            {"query_id": q.query_id, "text": q.text}
            for q in serialize_cast_history(self._test_ds.queries_iter())
        ]
        self._queries_df = pd.DataFrame(rows, columns=QUERY_COLUMNS)
        self._qrels_df = self._qrels_frame(self._test_ds)


class Touche2020Lane(IRDatasetsMaterialized):
    """Touché 2020 comparative decision topics over ~383K argument passages."""

    name = "beir-touche-2020"
    _CORPUS_ID = TOUCHE_2020
    _TEST_ID = TOUCHE_2020


def _techqa_doc(text: Any) -> tuple[str, str]:
    normalized = " ".join(str(text).split())
    if not normalized:
        raise ValueError("TechQA: linked document is empty")
    digest = hashlib.sha256(normalized.encode()).hexdigest()[:24]
    return f"techqa-{digest}", normalized


class TechQaLane(MaterializedDataset):
    """IBM technical-support questions linked to Technote-derived documents."""

    name = "techqa"

    def _rows(self) -> Iterator[dict[str, Any]]:
        for split in TECHQA_SPLITS:
            yield from load_dataset(TECHQA_REPO, split=split, streaming=True)

    def load_metadata(self) -> None:
        queries: list[dict[str, str]] = []
        qrels: list[dict[str, str | int]] = []
        seen_qrels: set[tuple[str, str]] = set()
        for row in self._rows():
            required = ("id", "question", "document")
            missing = [key for key in required if key not in row]
            if missing:
                raise ValueError(f"{TECHQA_REPO}: missing {missing}")
            query_id = str(row["id"])
            if self.query_ids is not None and query_id not in self.query_ids:
                continue
            doc_id, _ = _techqa_doc(row["document"])
            queries.append({"query_id": query_id, "text": str(row["question"])})
            pair = (query_id, doc_id)
            if pair in seen_qrels:
                continue
            seen_qrels.add(pair)
            qrels.append(
                {"query_id": query_id, "doc_id": doc_id, "relevance": 1}
            )
        self._queries_df = pd.DataFrame(queries, columns=QUERY_COLUMNS)
        self._qrels_df = pd.DataFrame(qrels, columns=QREL_COLUMNS)

    def _iter_corpus(self) -> Iterator[dict[str, str]]:
        seen: set[str] = set()
        for row in self._rows():
            if "document" not in row:
                raise ValueError(f"{TECHQA_REPO}: row missing 'document'")
            doc_id, text = _techqa_doc(row["document"])
            if doc_id in seen:
                continue
            seen.add(doc_id)
            yield {"doc_id": doc_id, "title": "", "text": text}
