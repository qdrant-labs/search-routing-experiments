"""Wave-3 query sources: commerce, finance, conversational, enterprise
support and argument retrieval distributions."""

from collections.abc import Iterator

import ir_datasets
from datasets import load_dataset

from dataset_registry.core import (
    Availability,
    DatasetCard,
    DatasetName,
    Grounding,
    Query,
    QueryProvenance,
    RegistryDataset,
    Scope,
    SourceKind,
)
from dataset_registry.irds import IRDatasetsBacked

ESCI_EXAMPLES = (
    "https://media.githubusercontent.com/media/amazon-science/esci-data/"
    "main/shopping_queries_dataset/shopping_queries_dataset_examples.parquet"
)
WANDS_QUERIES = (
    "https://raw.githubusercontent.com/wayfair/WANDS/main/dataset/query.csv"
)
FINDER_REPO = "Linq-AI-Research/FinDER"
CAST_2020 = "trec-cast/v1/2020/judged"
TECHQA_REPO = "rojagtap/tech-qa"
TECHQA_SPLITS = ("train", "validation", "test")
TOUCHE_2020 = "beir/webis-touche2020/v2"


def serialize_cast_history(queries: Iterator) -> Iterator[Query]:
    """Yield each CAsT turn as `prior_turns [CURRENT] utterance`, grouped by
    topic_number and ordered by turn_number."""
    by_topic: dict[int, list] = {}
    for query in queries:
        by_topic.setdefault(query.topic_number, []).append(query)
    for turns in by_topic.values():
        turns.sort(key=lambda q: q.turn_number)
        history: list[str] = []
        for turn in turns:
            text = " ".join([*history, "[CURRENT]", turn.raw_utterance])
            yield Query(str(turn.query_id), text)
            history.append(turn.raw_utterance)


class AmazonEsci(RegistryDataset):
    """Authentic English customer queries from ESCI's reduced hard split."""

    @property
    def card(self) -> DatasetCard:
        return DatasetCard(
            name=DatasetName.AMAZON_ESCI_EN_HARD,
            source=SourceKind.URL,
            grounding=Grounding.QQ,
            llm_target=False,
            query_provenance=QueryProvenance.HUMAN,
            scope=Scope.SPECIFIC,
            non_trivial=True,
            multilingual=False,
            multimodal=False,
            availability=Availability.OPEN,
            homepage="https://github.com/amazon-science/esci-data",
        )

    def describe_source(self) -> str:
        return ESCI_EXAMPLES

    def _fetch_queries(self) -> Iterator[Query]:
        rows = load_dataset(
            "parquet",
            data_files=ESCI_EXAMPLES,
            split="train",
            streaming=True,
        )
        seen: set[str] = set()
        for row in rows:
            if row["product_locale"] != "us" or int(row["small_version"]) != 1:
                continue
            query_id = str(row["query_id"])
            if query_id in seen:
                continue
            seen.add(query_id)
            yield Query(query_id, row["query"])


class Wands(RegistryDataset):
    """Genuine historical Wayfair product-search queries."""

    @property
    def card(self) -> DatasetCard:
        return DatasetCard(
            name=DatasetName.WANDS,
            source=SourceKind.URL,
            grounding=Grounding.QQ,
            llm_target=False,
            query_provenance=QueryProvenance.HUMAN,
            scope=Scope.SPECIFIC,
            non_trivial=True,
            multilingual=False,
            multimodal=False,
            availability=Availability.OPEN,
            homepage="https://github.com/wayfair/WANDS",
        )

    def describe_source(self) -> str:
        return WANDS_QUERIES

    def _fetch_queries(self) -> Iterator[Query]:
        rows = load_dataset(
            "csv",
            data_files=WANDS_QUERIES,
            delimiter="\t",
            split="train",
            streaming=True,
        )
        for row in rows:
            yield Query(str(row["query_id"]), row["query"])


class Finder(RegistryDataset):
    """Professional finance queries grounded in 10-K evidence passages."""

    @property
    def card(self) -> DatasetCard:
        return DatasetCard(
            name=DatasetName.FINDER,
            source=SourceKind.HUGGINGFACE,
            grounding=Grounding.QQ,
            llm_target=False,
            query_provenance=QueryProvenance.UNKNOWN,
            scope=Scope.SPECIFIC,
            non_trivial=True,
            multilingual=False,
            multimodal=False,
            availability=Availability.OPEN,
            homepage=f"https://huggingface.co/datasets/{FINDER_REPO}",
        )

    def describe_source(self) -> str:
        return f"hf://datasets/{FINDER_REPO} (train)"

    def _fetch_queries(self) -> Iterator[Query]:
        rows = load_dataset(FINDER_REPO, split="train", streaming=True)
        for row in rows:
            yield Query(str(row["_id"]), row["text"])


class TrecCast2020History(RegistryDataset):
    """CAsT 2020 judged turns with dialogue history serialized inline."""

    @property
    def card(self) -> DatasetCard:
        return DatasetCard(
            name=DatasetName.TREC_CAST_2020_HISTORY,
            source=SourceKind.IR_DATASETS,
            grounding=Grounding.QQ,
            llm_target=False,
            query_provenance=QueryProvenance.HUMAN,
            scope=Scope.GENERAL,
            non_trivial=True,
            multilingual=False,
            multimodal=False,
            availability=Availability.OPEN,
            homepage="https://ir-datasets.com/trec-cast.html",
        )

    def describe_source(self) -> str:
        return f"ir_datasets:{CAST_2020}"

    def _fetch_queries(self) -> Iterator[Query]:
        yield from serialize_cast_history(
            ir_datasets.load(CAST_2020).queries_iter()
        )


class TechQa(RegistryDataset):
    """IBM technical-support forum questions linked to Technote passages."""

    @property
    def card(self) -> DatasetCard:
        return DatasetCard(
            name=DatasetName.TECHQA,
            source=SourceKind.HUGGINGFACE,
            grounding=Grounding.QQ,
            llm_target=False,
            query_provenance=QueryProvenance.HUMAN,
            scope=Scope.SPECIFIC,
            non_trivial=True,
            multilingual=False,
            multimodal=False,
            availability=Availability.OPEN,
            homepage=f"https://huggingface.co/datasets/{TECHQA_REPO}",
        )

    def describe_source(self) -> str:
        return f"hf://datasets/{TECHQA_REPO} ({'+'.join(TECHQA_SPLITS)})"

    def _fetch_queries(self) -> Iterator[Query]:
        for split in TECHQA_SPLITS:
            rows = load_dataset(TECHQA_REPO, split=split, streaming=True)
            for row in rows:
                yield Query(str(row["id"]), row["question"])


class BeirTouche2020(IRDatasetsBacked):
    """Touché 2020 comparative decision topics over 383K argument passages."""

    irds_id = TOUCHE_2020

    @property
    def card(self) -> DatasetCard:
        return DatasetCard(
            name=DatasetName.BEIR_TOUCHE_2020,
            source=SourceKind.IR_DATASETS,
            grounding=Grounding.QQ,
            llm_target=False,
            query_provenance=QueryProvenance.HUMAN,
            scope=Scope.GENERAL,
            non_trivial=True,
            multilingual=False,
            multimodal=False,
            availability=Availability.OPEN,
            homepage="https://ir-datasets.com/beir.html#beir/webis-touche2020",
        )
