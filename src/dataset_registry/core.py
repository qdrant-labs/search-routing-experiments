import random
from abc import ABC, abstractmethod
from collections import deque
from collections.abc import Iterator, Sequence
from enum import StrEnum
from pathlib import Path
from typing import NamedTuple

import pyarrow as pa
import pyarrow.parquet as pq
from pydantic import BaseModel, ConfigDict
from tqdm.auto import tqdm

DEFAULT_CACHE_DIR = Path(__file__).resolve().parent.parent / "data" / "registry_cache"

_QUERY_SCHEMA = pa.schema([("query_id", pa.string()), ("text", pa.string())])
_WRITE_BATCH_SIZE = 65_536


class DatasetName(StrEnum):
    """Closed set of registered datasets — the registry's counterpart to
    StructuralIdentifier. Registering a dataset (or a new language of a
    parameterized one) means adding a member here plus an instance in
    registry.DATASETS; a card naming an unlisted dataset fails validation.
    """

    MSMARCO_PASSAGE_DEV = "msmarco-passage-dev"
    TREC_DL_2022 = "trec-dl-2022"
    BEIR_NFCORPUS = "beir-nfcorpus"
    MIRACL_EN_DEV = "miracl-en-dev"


class Grounding(StrEnum):
    """What retrieval assets the dataset ships with."""

    QO = "query_only"
    """Queries only — cheapest to find, hardest to validate, hard to label."""

    QC = "query_corpus"
    """Queries + corpus — cheap to find, easy to label, hard to validate."""

    QQ = "query_qrels"
    """Queries + corpus + qrels — hard to find, easy to validate."""


class Scope(StrEnum):
    GENERAL = "general"
    """Generic queries spanning many topics (web search, Wikipedia QA)."""

    SPECIFIC = "specific"
    """Queries bound to one domain (medical, finance, geoscience)."""


class Availability(StrEnum):
    OPEN = "open"
    """Loads with no auth: ir_datasets download or public HF repo."""

    GATED = "gated"
    """Needs an account or accepted terms (Kaggle auth, HF gated repo)."""

    RESTRICTED = "restricted"
    """Manual approval: email the authors, institutional agreement."""


class SourceKind(StrEnum):
    IR_DATASETS = "ir_datasets"
    HUGGINGFACE = "huggingface"
    URL = "url"


class Query(NamedTuple):
    query_id: str
    text: str


class DatasetCard(BaseModel):
    """Frozen metadata describing one dataset along the six selection
    dimensions, plus acquisition facts."""

    model_config = ConfigDict(frozen=True)

    name: DatasetName
    source: SourceKind
    grounding: Grounding
    llm_target: bool
    scope: Scope
    non_trivial: bool
    multilingual: bool
    multimodal: bool
    availability: Availability
    homepage: str
    recommended_sample: int | None = None
    """Cap for taxonomy profiling on huge query sets (e.g. ORCAS at 18.8M).
    None means the full query set is small enough to always load whole."""


def _to_record_batch(queries: Sequence[Query]) -> pa.RecordBatch:
    return pa.RecordBatch.from_arrays(
        [
            pa.array([query.query_id for query in queries], type=pa.string()),
            pa.array([query.text for query in queries], type=pa.string()),
        ],
        schema=_QUERY_SCHEMA,
    )


def _batch_to_queries(batch: pa.RecordBatch) -> Iterator[Query]:
    return map(
        Query,
        batch.column("query_id").to_pylist(),
        batch.column("text").to_pylist(),
    )


def _take_rows(parquet: pq.ParquetFile, indexes: Sequence[int]) -> list[Query]:
    """Rows at ascending global `indexes`, walking batches so at most one
    batch is decoded in memory at a time."""
    wanted = deque(indexes)
    rows: list[Query] = []
    offset = 0
    for batch in parquet.iter_batches():
        local: list[int] = []
        while wanted and wanted[0] < offset + batch.num_rows:
            local.append(wanted.popleft() - offset)
        if local:
            rows.extend(_batch_to_queries(batch.take(pa.array(local, type=pa.int64()))))
        if not wanted:
            break
        offset += batch.num_rows
    return rows


class RegistryDataset(ABC):
    """
    One entry per dataset. Metadata lives in `card`. The raw source is
    fetched over the network exactly once: `_fetch_queries` streams into a
    local parquet cache, and every read or sample runs against that file.
    """

    def __init__(self, cache_dir: Path | None = None) -> None:
        self._cache_dir = cache_dir if cache_dir is not None else DEFAULT_CACHE_DIR
        self._cache_announced = False

    @property
    @abstractmethod
    def card(self) -> DatasetCard:
        """Selection dimensions and acquisition facts for this dataset."""

    @abstractmethod
    def _fetch_queries(self) -> Iterator[Query]:
        """Stream queries from the raw source; runs once per cache fill and
        must never materialize the corpus."""

    def _fetch_total(self) -> int | None:
        """Expected query count when the source knows it cheaply; sizes the
        progress bar. None -> count-only bar."""
        return None

    def describe_source(self) -> str:
        """Human-readable locator of the raw source, shown in progress
        output so a fetch is never anonymous."""
        return self.card.source.value

    @property
    def cache_path(self) -> Path:
        return self._cache_dir / f"{self.card.name.value}.parquet"

    def _fill_cache(self, path: Path) -> None:
        """One full pass over the raw source, batch-written so huge query
        sets never materialize in RAM."""
        queries = tqdm(
            self._fetch_queries(),
            total=self._fetch_total(),
            desc=f"fetch {self.card.name.value}",
            unit="query",
        )
        row_groups = 0
        with pq.ParquetWriter(path, _QUERY_SCHEMA) as writer:
            pending: list[Query] = []
            for query in queries:
                pending.append(query)
                if len(pending) >= _WRITE_BATCH_SIZE:
                    writer.write_batch(_to_record_batch(pending))
                    pending.clear()
                    row_groups += 1
                    queries.set_postfix(row_groups=row_groups, refresh=False)
            if pending:
                writer.write_batch(_to_record_batch(pending))

    def _ensure_cached(self) -> pq.ParquetFile:
        path = self.cache_path
        name = self.card.name.value
        if not path.exists():
            tqdm.write(
                f"[{name}] cache miss -> fetching {self.describe_source()} "
                f"(one-time network pass)"
            )
            self._cache_dir.mkdir(parents=True, exist_ok=True)
            # write to a tmp file and rename: a fetch that dies mid-stream
            # must not leave a truncated cache that later reads as complete
            tmp = path.with_name(path.name + ".tmp")
            try:
                self._fill_cache(tmp)
                tmp.rename(path)
            finally:
                tmp.unlink(missing_ok=True)
            parquet = pq.ParquetFile(path)
            size_mb = path.stat().st_size / 1e6
            tqdm.write(
                f"[{name}] cached {parquet.metadata.num_rows:,} queries "
                f"-> {path} ({size_mb:.1f} MB)"
            )
            self._cache_announced = True
            return parquet
        parquet = pq.ParquetFile(path)
        if not self._cache_announced:
            tqdm.write(
                f"[{name}] cache hit -> {path} "
                f"({parquet.metadata.num_rows:,} queries, no network)"
            )
            self._cache_announced = True
        return parquet

    def load_queries(self) -> Iterator[Query]:
        """Stream all queries from the local cache, filling it on first use."""
        for batch in self._ensure_cached().iter_batches():
            yield from _batch_to_queries(batch)

    def sample_queries(self, n: int | None = None, *, seed: int = 0) -> list[Query]:
        """Uniform random sample of `n` queries, in source order.

        Draws row indexes against the cached parquet's row count (metadata
        only), then takes just those rows batch by batch — after the one-time
        cache fill, no call ever re-iterates the raw source. `n` defaults to
        `card.recommended_sample`; None means the full query set.
        """
        cap = n if n is not None else self.card.recommended_sample
        parquet = self._ensure_cached()
        n_rows = parquet.metadata.num_rows
        if cap is None or cap >= n_rows:
            return list(self.load_queries())
        indexes = sorted(random.Random(seed).sample(range(n_rows), cap))
        return _take_rows(parquet, indexes)
