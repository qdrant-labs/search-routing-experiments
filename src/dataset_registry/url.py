from collections.abc import Iterator

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

LIMIT_QUERIES_URL = (
    "https://raw.githubusercontent.com/google-deepmind/limit/main/data/limit/queries.jsonl"
)


class Limit(RegistryDataset):
    """LIMIT embedding stress set (arXiv 2508.21038): 1,000 deliberately
    trivial queries whose relevant top-k document combinations exceed what
    a fixed-dimension embedding can return — dense fails by construction.
    Registered for the strategy-labeling stage; excluded from harvest
    targets (SPEC d21)."""

    @property
    def card(self) -> DatasetCard:
        return DatasetCard(
            name=DatasetName.LIMIT,
            source=SourceKind.URL,
            grounding=Grounding.QQ,
            llm_target=False,
            query_provenance=QueryProvenance.TEMPLATE,
            scope=Scope.GENERAL,
            non_trivial=True,
            multilingual=False,
            multimodal=False,
            availability=Availability.OPEN,
            homepage="https://github.com/google-deepmind/limit",
        )

    def describe_source(self) -> str:
        return LIMIT_QUERIES_URL

    def _fetch_queries(self) -> Iterator[Query]:
        rows = load_dataset(
            "json", data_files=LIMIT_QUERIES_URL, split="train", streaming=True
        )
        for row in rows:
            yield Query(str(row["_id"]), row["text"])
