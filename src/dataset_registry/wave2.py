"""Wave-2 query sources (docs/datasets.md): FreshStack, ANTIQUE, LoTTE,
WebFAQ, ScIRGen-Geo, CLERC and GooAQ. FreshStack, WebFAQ and ScIRGen-Geo
carry repo/config/field guesses that only a first fetch can settle.
"""

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
from dataset_registry.irds import IRDatasetsBacked

FRESHSTACK_TOPICS = frozenset(
    {"angular", "godot", "langchain", "laravel", "yolo"}
)
FRESHSTACK_QUERIES_REPO = "freshstack/queries-oct-2024"

LOTTE_DOMAINS = frozenset(
    {"lifestyle", "recreation", "science", "technology", "writing"}
)
LOTTE_QUERY_SETS = frozenset({"search", "forum"})

WEBFAQ_REPO = "PaDaS-Lab/webfaq-retrieval"
WEBFAQ_LANGUAGE = "eng"
WEBFAQ_QUERIES_CONFIG = f"{WEBFAQ_LANGUAGE}-queries"

SCIRGEN_REPO = "usail-hkust/ScIRGen-Geo"

CLERC_QUERIES = "hf://datasets/jhu-clsp/CLERC/teva_train_dir/train_data.jsonl.gz"

GOOAQ_JSONL = Path(__file__).resolve().parent.parent / "data" / "gooaq" / "gooaq.jsonl"

_NON_WORD = str.maketrans(
    dict.fromkeys("0123456789 \t\n!\"#$%&()*+,-./:;<=>?@[\\]^_`{|}~", " ")
)

# GooAQ's non-English rows are ASCII ("0800 da natura do brasil?"), so the
# language test is a marker vote, not a charset test.
ENGLISH_MARKERS = frozenset(
    """an the this that these those what which who whom whose how why when
    where is are am was were be been being do does did doing done have has had
    can could shall should will would may might must not no of to in on at for
    with without from by about into between after before during than because
    while and or but if then there here it its they them their you your my mine
    his her our we i he she get make take use need know mean many much more
    most best good long""".split()
)
FOREIGN_MARKERS = frozenset(
    """da dos das de del dele el la las los os ao aos pelo pela um un una uno
    que qual quais cual cuales como onde donde cuando quando porque quien quem
    para por com con nao mais menos muito sem sobre tudo todos mesmo ser fazer
    tem sao esta este esse essa isso ele ela eles meu minha seu sua es eso esa
    ese pero tambien quel quelle quels quelles comment pourquoi est les des une
    dans avec pour sur sans sont wie wo warum wann welche ist sind und der die
    das den dem ein eine nicht van het hoe waarom wat bagaimana mengapa kenapa
    dimana kapan yang untuk dengan tidak adalah bisa cara harga""".split()
)


def is_english(text: str) -> bool:
    """More English function words than Portuguese/Spanish/French/German/
    Dutch/Indonesian ones — measured on GooAQ: keeps 99.2% of all rows and
    99.94% of answer-bearing ones, drops clean pt/es/nl/fr questions."""
    words = set(text.lower().translate(_NON_WORD).split())
    return len(words & ENGLISH_MARKERS) > len(words & FOREIGN_MARKERS)


def has_cjk(text: str) -> bool:
    """A CJK ideograph (U+4E00..U+9FFF) anywhere — ScIRGen-Geo's EN/ZH rows
    carry no language column, but the Chinese ones give themselves away."""
    return any("一" <= char <= "鿿" for char in text)


def expect_keys(row: dict, keys: tuple[str, ...], source: str) -> None:
    """Fail naming the real schema, so a wrong guess is legible at first fetch."""
    missing = [key for key in keys if key not in row]
    if missing:
        raise ValueError(f"{source}: missing {missing}; row has {sorted(row)}")


def scirgen_rows(row_limit: int | None = None) -> Iterator[dict]:
    """Every record once. The card documents four directories but the repo
    exposes a single split whose rows carry their own `split` column, so the
    name is resolved rather than assumed and records are deduped by id."""
    splits = load_dataset(SCIRGEN_REPO, streaming=True)
    if len(splits) != 1:
        raise ValueError(
            f"{SCIRGEN_REPO}: expected one split, got {sorted(splits)} — "
            "pick one deliberately rather than concatenating them."
        )
    seen: set[str] = set()
    for row in next(iter(splits.values())):
        expect_keys(row, ("id", "query"), SCIRGEN_REPO)
        record = str(row["id"])
        if record in seen:
            continue
        seen.add(record)
        yield row
        if row_limit is not None and len(seen) >= row_limit:
            return


def scirgen_questions(row: dict) -> Iterator[tuple[str, str]]:
    """Every English question on one record, id'd by its category and position."""
    record = str(row["id"])
    for category, entries in sorted((row["query"] or {}).items()):
        for index, entry in enumerate(entries or ()):
            text = (entry.get("QuestionEn") or "").strip()
            if text and not has_cjk(text):
                yield f"{record}-{category}-{index}", text


class FreshStackTopic(RegistryDataset):
    """One FreshStack topic: StackOverflow questions over that framework's
    own doc corpus. A question is its title plus its body, joined, because
    both carry retrievable signal and the body is where code fragments live."""

    def __init__(self, topic: str, cache_dir: Path | None = None) -> None:
        super().__init__(cache_dir)
        if topic not in FRESHSTACK_TOPICS:
            raise ValueError(
                f"unknown FreshStack topic {topic!r}; "
                f"expected one of {sorted(FRESHSTACK_TOPICS)}"
            )
        self.topic = topic

    @property
    def card(self) -> DatasetCard:
        return DatasetCard(
            name=DatasetName(f"freshstack-{self.topic}"),
            source=SourceKind.HUGGINGFACE,
            grounding=Grounding.QQ,
            llm_target=False,
            query_provenance=QueryProvenance.HUMAN,
            scope=Scope.SPECIFIC,
            non_trivial=True,
            multilingual=False,
            multimodal=False,
            availability=Availability.OPEN,
            homepage=f"https://huggingface.co/datasets/{FRESHSTACK_QUERIES_REPO}",
        )

    def describe_source(self) -> str:
        return f"hf://datasets/{FRESHSTACK_QUERIES_REPO} ({self.topic})"

    def _fetch_queries(self) -> Iterator[Query]:
        rows = load_dataset(
            FRESHSTACK_QUERIES_REPO, self.topic, split="test", streaming=True
        )
        for row in rows:
            expect_keys(row, ("query_id", "query_title", "query_text"), self.topic)
            title = (row["query_title"] or "").strip()
            body = (row["query_text"] or "").strip()
            yield Query(str(row["query_id"]), f"{title}\n{body}".strip())


class Antique(IRDatasetsBacked):
    """ANTIQUE test: 200 non-factoid Yahoo Answers questions over 403,666
    answer passages, graded 1-4 (levels 3-4 answer the question). The first
    download logs the authors' data-usage-agreement notice."""

    irds_id = "antique/test"

    @property
    def card(self) -> DatasetCard:
        return DatasetCard(
            name=DatasetName.ANTIQUE,
            source=SourceKind.IR_DATASETS,
            grounding=Grounding.QQ,
            llm_target=False,
            query_provenance=QueryProvenance.HUMAN,
            scope=Scope.GENERAL,
            non_trivial=True,
            multilingual=False,
            multimodal=False,
            availability=Availability.OPEN,
            homepage="https://ir-datasets.com/antique.html#antique/test",
        )


class LotteSubset(IRDatasetsBacked):
    """One LoTTE domain's test queries in one register: `search`
    (Google-style) or `forum` (StackExchange-style) over the same corpus.
    Registered v0: technology; other domains follow the MiraclDev rule."""

    def __init__(
        self, domain: str, query_set: str, cache_dir: Path | None = None
    ) -> None:
        super().__init__(cache_dir)
        if domain not in LOTTE_DOMAINS:
            raise ValueError(
                f"unknown LoTTE domain {domain!r}; "
                f"expected one of {sorted(LOTTE_DOMAINS)}"
            )
        if query_set not in LOTTE_QUERY_SETS:
            raise ValueError(
                f"unknown LoTTE query set {query_set!r}; "
                f"expected one of {sorted(LOTTE_QUERY_SETS)}"
            )
        self.domain = domain
        self.query_set = query_set
        self.irds_id = f"lotte/{domain}/test/{query_set}"

    @property
    def card(self) -> DatasetCard:
        return DatasetCard(
            name=DatasetName(f"lotte-{self.domain}-{self.query_set}"),
            source=SourceKind.IR_DATASETS,
            grounding=Grounding.QQ,
            llm_target=False,
            query_provenance=QueryProvenance.HUMAN,
            scope=Scope.SPECIFIC,
            non_trivial=True,
            multilingual=False,
            multimodal=False,
            availability=Availability.OPEN,
            homepage=f"https://ir-datasets.com/lotte.html#{self.irds_id}",
        )


class WebFaq(RegistryDataset):
    """WebFAQ English retrieval slice (arXiv 2502.20936): FAQ questions
    mined from schema.org markup, shipped with source qrels. The
    `<lang>-queries` config naming and BEIR field names are unverified."""

    @property
    def card(self) -> DatasetCard:
        return DatasetCard(
            name=DatasetName.WEBFAQ_ENG,
            source=SourceKind.HUGGINGFACE,
            grounding=Grounding.QQ,
            llm_target=False,
            query_provenance=QueryProvenance.HUMAN,
            scope=Scope.GENERAL,
            non_trivial=True,
            multilingual=True,
            multimodal=False,
            availability=Availability.OPEN,
            homepage=f"https://huggingface.co/datasets/{WEBFAQ_REPO}",
            recommended_sample=50_000,
        )

    def describe_source(self) -> str:
        return f"hf://datasets/{WEBFAQ_REPO} ({WEBFAQ_QUERIES_CONFIG})"

    def _fetch_queries(self) -> Iterator[Query]:
        splits = load_dataset(WEBFAQ_REPO, WEBFAQ_QUERIES_CONFIG, streaming=True)
        for rows in splits.values():
            for row in rows:
                expect_keys(row, ("_id", "text"), WEBFAQ_QUERIES_CONFIG)
                yield Query(str(row["_id"]), row["text"])


class ScirgenGeo(RegistryDataset):
    """ScIRGen-Geo English questions: LLM-written dataset-seeking queries over
    geoscience metadata records. One record carries several question categories,
    each a list, and every question ships parallel English and Chinese text."""

    @property
    def card(self) -> DatasetCard:
        return DatasetCard(
            name=DatasetName.SCIRGEN_GEO_EN,
            source=SourceKind.HUGGINGFACE,
            grounding=Grounding.QQ,
            llm_target=True,
            query_provenance=QueryProvenance.LLM,
            scope=Scope.SPECIFIC,
            non_trivial=True,
            multilingual=True,
            multimodal=False,
            availability=Availability.OPEN,
            homepage=f"https://huggingface.co/datasets/{SCIRGEN_REPO}",
        )

    def describe_source(self) -> str:
        return f"hf://datasets/{SCIRGEN_REPO} (English questions)"

    def _fetch_queries(self) -> Iterator[Query]:
        for row in scirgen_rows():
            for query_id, text in scirgen_questions(row):
                yield Query(query_id, text)


class Clerc(RegistryDataset):
    """CLERC (arXiv 2406.17186): 327,414 legal case-text queries whose gold
    documents are the cited cases' passages. Streams the Tevatron-format
    training file — the one CLERC layout confirmed against a local cache."""

    @property
    def card(self) -> DatasetCard:
        return DatasetCard(
            name=DatasetName.CLERC,
            source=SourceKind.HUGGINGFACE,
            grounding=Grounding.QQ,
            llm_target=False,
            query_provenance=QueryProvenance.HUMAN,
            scope=Scope.SPECIFIC,
            non_trivial=True,
            multilingual=False,
            multimodal=False,
            availability=Availability.OPEN,
            homepage="https://huggingface.co/datasets/jhu-clsp/CLERC",
            recommended_sample=50_000,
        )

    def describe_source(self) -> str:
        return CLERC_QUERIES

    def _fetch_queries(self) -> Iterator[Query]:
        # the passages ride on every row, so this pass reads ~15GB of JSON
        # to keep 327K queries; the parquet cache pays it once
        rows = load_dataset(
            "json", data_files=CLERC_QUERIES, split="train", streaming=True
        )
        for row in rows:
            expect_keys(row, ("query_id", "query"), CLERC_QUERIES)
            yield Query(str(row["query_id"]), row["query"])


class Gooaq(RegistryDataset):
    """GooAQ questions mined from Google autocomplete, kept only where a
    snippet `answer` grounds them (3,032,114 of 5,030,530 rows — 60.3%).
    Reads the DVC-tracked local copy, not the HF repo."""

    @property
    def card(self) -> DatasetCard:
        return DatasetCard(
            name=DatasetName.GOOAQ,
            source=SourceKind.HUGGINGFACE,
            grounding=Grounding.QC,
            llm_target=False,
            query_provenance=QueryProvenance.HUMAN,
            scope=Scope.GENERAL,
            non_trivial=False,
            multilingual=False,
            multimodal=False,
            availability=Availability.OPEN,
            homepage="https://huggingface.co/datasets/allenai/gooaq",
            recommended_sample=50_000,
        )

    def describe_source(self) -> str:
        return str(GOOAQ_JSONL)

    def _fetch_queries(self) -> Iterator[Query]:
        rows = load_dataset(
            "json", data_files=str(GOOAQ_JSONL), split="train", streaming=True
        )
        for row in rows:
            expect_keys(row, ("id", "question", "answer"), str(GOOAQ_JSONL))
            if row["answer"] and is_english(row["question"]):
                yield Query(str(row["id"]), row["question"])
