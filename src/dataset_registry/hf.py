from collections.abc import Iterator
from pathlib import Path

from datasets import load_dataset

from dataset_registry.core import (
    Availability,
    DatasetCard,
    DatasetName,
    Grounding,
    Query,
    RegistryDataset,
    Scope,
    SourceKind,
)

MIRACL_LANGUAGES = frozenset(
    {
        "ar", "bn", "de", "en", "es", "fa", "fi", "fr", "hi",
        "id", "ja", "ko", "ru", "sw", "te", "th", "yo", "zh",
    }
)


class MiraclDev(RegistryDataset):
    """MIRACL dev topics for one of its 18 languages (en: 799 questions).

    Streams the raw TSV: the miracl/miracl repo carries a legacy loading
    script that datasets >= 4 refuses to run.

    The card name is built from the language, so instantiating a language
    that has no DatasetName member fails validation at `card` access —
    registering a new language means adding the member plus an instance
    in registry.DATASETS.
    """

    def __init__(self, language: str = "en", cache_dir: Path | None = None) -> None:
        super().__init__(cache_dir)
        if language not in MIRACL_LANGUAGES:
            raise ValueError(
                f"unknown MIRACL language {language!r}; "
                f"expected one of {sorted(MIRACL_LANGUAGES)}"
            )
        self.language = language

    @property
    def data_files(self) -> str:
        return (
            f"hf://datasets/miracl/miracl/miracl-v1.0-{self.language}/topics/"
            f"topics.miracl-v1.0-{self.language}-dev.tsv"
        )

    @property
    def card(self) -> DatasetCard:
        return DatasetCard(
            name=DatasetName(f"miracl-{self.language}-dev"),
            source=SourceKind.HUGGINGFACE,
            grounding=Grounding.QQ,
            llm_target=False,
            scope=Scope.GENERAL,
            non_trivial=False,
            multilingual=True,
            multimodal=False,
            availability=Availability.OPEN,
            homepage="https://huggingface.co/datasets/miracl/miracl",
        )

    def describe_source(self) -> str:
        return f"hf://datasets/miracl/miracl ({self.language}-dev topics TSV)"

    def _fetch_queries(self) -> Iterator[Query]:
        topics = load_dataset(
            "csv",
            data_files=self.data_files,
            delimiter="\t",
            column_names=["query_id", "query"],
            split="train",
            streaming=True,
        )
        for row in topics:
            yield Query(str(row["query_id"]), row["query"])
