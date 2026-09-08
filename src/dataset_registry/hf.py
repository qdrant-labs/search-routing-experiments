from collections.abc import Iterator
from pathlib import Path

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
            query_provenance=QueryProvenance.HUMAN,
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


BRIGHT_SPLITS = frozenset(
    {
        "biology", "earth_science", "economics", "psychology", "robotics",
        "stackoverflow", "sustainable_living", "leetcode", "pony", "aops",
        "theoremqa_questions", "theoremqa_theorems",
    }
)


class BrightSplit(RegistryDataset):
    """One BRIGHT domain split: reasoning-intensive queries (SoA drops from
    59.0 nDCG on BEIR to 18.3 here). Registered v0: leetcode / aops /
    theoremqa_questions — the CODE_FRAGMENT / MATH_EXPRESSION harvest
    sources (SPEC d20-21). Other splits follow the MiraclDev rule: valid
    split names pass __init__, but card access fails until a DatasetName
    member plus a registry instance exist."""

    def __init__(self, split: str, cache_dir: Path | None = None) -> None:
        super().__init__(cache_dir)
        if split not in BRIGHT_SPLITS:
            raise ValueError(
                f"unknown BRIGHT split {split!r}; "
                f"expected one of {sorted(BRIGHT_SPLITS)}"
            )
        self.split = split

    @property
    def card(self) -> DatasetCard:
        return DatasetCard(
            name=DatasetName(f"bright-{self.split.replace('_', '-')}"),
            source=SourceKind.HUGGINGFACE,
            grounding=Grounding.QQ,
            llm_target=False,
            query_provenance=QueryProvenance.HUMAN,
            scope=Scope.GENERAL,
            non_trivial=True,
            multilingual=False,
            multimodal=False,
            availability=Availability.OPEN,
            homepage="https://huggingface.co/datasets/xlangai/BRIGHT",
        )

    def describe_source(self) -> str:
        return f"hf://datasets/xlangai/BRIGHT (examples/{self.split})"

    def _fetch_queries(self) -> Iterator[Query]:
        rows = load_dataset(
            "xlangai/BRIGHT", "examples", split=self.split, streaming=True
        )
        for row in rows:
            yield Query(str(row["id"]), row["query"])


class Quest(RegistryDataset):
    """QUEST: natural queries with implicit set operations ("shorebirds
    that are not sandpipers", ACL 2023). Validation + test splits only —
    the train split's auto-composed augmented queries are excluded
    (ratified 2026-07-21). Rows carry no id; ids are synthesized as
    `{split}-{index}`."""

    _SPLITS = ("validation", "test")

    @property
    def card(self) -> DatasetCard:
        return DatasetCard(
            name=DatasetName.QUEST,
            source=SourceKind.HUGGINGFACE,
            grounding=Grounding.QQ,
            llm_target=False,
            query_provenance=QueryProvenance.HUMAN,
            scope=Scope.GENERAL,
            non_trivial=True,
            multilingual=False,
            multimodal=False,
            availability=Availability.OPEN,
            homepage="https://huggingface.co/datasets/cmalaviya/quest",
        )

    def describe_source(self) -> str:
        return "hf://datasets/cmalaviya/quest (validation+test)"

    def _fetch_queries(self) -> Iterator[Query]:
        for split in self._SPLITS:
            rows = load_dataset("cmalaviya/quest", "main", split=split, streaming=True)
            for index, row in enumerate(rows):
                yield Query(f"{split}-{index}", row["query"])


CRUMB_TASKS = frozenset(
    {
        "clinical_trial", "code_retrieval", "legal_qa", "paper_retrieval",
        "set_operation_entity_retrieval", "stack_exchange",
        "theorem_retrieval", "tip_of_the_tongue",
    }
)


class CrumbTask(RegistryDataset):
    """One CRUMB task: complex multi-constraint queries (arXiv 2509.07253).
    All eight tasks register (SPEC d21)."""

    def __init__(self, task: str, cache_dir: Path | None = None) -> None:
        super().__init__(cache_dir)
        if task not in CRUMB_TASKS:
            raise ValueError(
                f"unknown CRUMB task {task!r}; expected one of {sorted(CRUMB_TASKS)}"
            )
        self.task = task

    @property
    def card(self) -> DatasetCard:
        return DatasetCard(
            name=DatasetName(f"crumb-{self.task.replace('_', '-')}"),
            source=SourceKind.HUGGINGFACE,
            grounding=Grounding.QQ,
            llm_target=True,
            query_provenance=QueryProvenance.UNKNOWN,
            scope=Scope.GENERAL,
            non_trivial=True,
            multilingual=False,
            multimodal=False,
            availability=Availability.OPEN,
            homepage="https://huggingface.co/datasets/jfkback/crumb",
        )

    def describe_source(self) -> str:
        return f"hf://datasets/jfkback/crumb (evaluation_queries/{self.task})"

    def _fetch_queries(self) -> Iterator[Query]:
        rows = load_dataset(
            "jfkback/crumb", "evaluation_queries", split=self.task, streaming=True
        )
        for row in rows:
            yield Query(str(row["query_id"]), row["query_content"])


RARB_POOLS = {"math": "math-pooled", "code": "humanevalpack-mbpp-pooled"}


class RarbPool(RegistryDataset):
    """RAR-b pooled reasoning-as-retrieval queries: math = MATH + GSM8K
    (6,319), code = HumanEvalPack + MBPP (1,484) — the densest
    MATH_EXPRESSION / CODE_FRAGMENT harvest source (SPEC d20-21).

    Streams the raw BEIR-style queries.jsonl: the RAR-b repos carry a
    legacy loading script that datasets >= 4 refuses to run (same
    workaround as MiraclDev)."""

    def __init__(self, pool: str, cache_dir: Path | None = None) -> None:
        super().__init__(cache_dir)
        if pool not in RARB_POOLS:
            raise ValueError(
                f"unknown RAR-b pool {pool!r}; expected one of {sorted(RARB_POOLS)}"
            )
        self.pool = pool

    @property
    def data_files(self) -> str:
        return f"hf://datasets/RAR-b/{RARB_POOLS[self.pool]}/queries.jsonl"

    @property
    def card(self) -> DatasetCard:
        return DatasetCard(
            name=DatasetName(f"rarb-{self.pool}"),
            source=SourceKind.HUGGINGFACE,
            grounding=Grounding.QQ,
            llm_target=False,
            query_provenance=QueryProvenance.HUMAN,
            scope=Scope.SPECIFIC,
            non_trivial=True,
            multilingual=False,
            multimodal=False,
            availability=Availability.OPEN,
            homepage=f"https://huggingface.co/datasets/RAR-b/{RARB_POOLS[self.pool]}",
        )

    def describe_source(self) -> str:
        return self.data_files

    def _fetch_queries(self) -> Iterator[Query]:
        rows = load_dataset(
            "json", data_files=self.data_files, split="train", streaming=True
        )
        for row in rows:
            yield Query(str(row["_id"]), row["text"])
