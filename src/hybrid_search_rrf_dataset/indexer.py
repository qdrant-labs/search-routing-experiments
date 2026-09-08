import logging
import pickle
import threading
import time
import uuid
from abc import ABC, abstractmethod
from collections.abc import Iterable, Iterator, Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, ClassVar, Generic, Literal, TypeVar

import numpy as np
from fastembed import SparseTextEmbedding, TextEmbedding
from pydantic import BaseModel, ConfigDict, Field
from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import ResponseHandlingException, UnexpectedResponse
from qdrant_client.models import (
    Distance,
    Document,
    Modifier,
    PointStruct,
    SparseVector,
    SparseVectorParams,
    VectorParams,
)
from tqdm.auto import tqdm

T = TypeVar("T", bound=BaseModel)

logger = logging.getLogger(__name__)

_EMBED_CHECKPOINT_SIZE = 2000
"""Missing texts are embedded and flushed to the on-disk cache in chunks of this
size rather than all at once — a hard kill (memguard, OOM, a crashed kernel)
loses at most one chunk instead of the whole call. A `finally` cannot save what
a SIGKILL never let it reach; only a save that already ran on disk survives."""
_UPSERT_MAX_ATTEMPTS = 4
_UPSERT_BACKOFF_S = 5.0
"""A cloud-inference vector slot embeds INSIDE the upsert call, so a transient
30s Qdrant Cloud Inference timeout (or any other 4xx/5xx from the server)
surfaces as `UnexpectedResponse` here, not as a network-layer error. Retries
with doubling backoff (5s, 10s, 20s) rather than failing the whole batch."""
_RETRYABLE_STATUS = frozenset({408, 429, 500, 502, 503, 504})
"""Only transient statuses are worth retrying. A 400/401/403 (bad request, auth,
or an exhausted provider key — 'Key limit exceeded') will never recover in 20s, so
fail fast and surface it instead of burning every attempt on a lost cause."""


def _chunked(seq: list[Any], size: int) -> Iterator[list[Any]]:
    for i in range(0, len(seq), size):
        yield seq[i : i + size]


class EmbeddingCache:
    """Disk-backed cache of vectors keyed by (namespace, model_id, item_id).

    Survives kernel restarts so a stopped/crashed upload doesn't lose work.
    One pickle file per (namespace, model_id, kind). Atomic write via
    tmp-then-rename. `namespace` is the lane: `item_id` hashes a bare `doc_id`
    and doc_ids collide across lanes, so an unnamespaced cache hands one lane's
    vectors to another lane's document under the same id.
    """

    def __init__(
        self, cache_dir: str | Path = "./.embedding_cache", namespace: str = ""
    ) -> None:
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.namespace = namespace
        self._memory: dict[tuple[str, str], dict[str, Any]] = {}

    def _path(self, model_id: str, kind: str) -> Path:
        safe = model_id.replace("/", "__")
        parts = [safe, self.namespace, kind] if self.namespace else [safe, kind]
        return self.cache_dir / f"{'.'.join(parts)}.pkl"

    def load(self, model_id: str, kind: str) -> dict[str, Any]:
        key = (model_id, kind)
        if key in self._memory:
            return self._memory[key]
        p = self._path(model_id, kind)
        data: dict[str, Any] = {}
        if p.exists():
            with p.open("rb") as f:
                data = pickle.load(f)
        self._memory[key] = data
        return data

    def save(self, model_id: str, kind: str) -> None:
        key = (model_id, kind)
        if key not in self._memory:
            return
        p = self._path(model_id, kind)
        tmp = p.with_suffix(p.suffix + ".tmp")
        with tmp.open("wb") as f:
            pickle.dump(self._memory[key], f, protocol=pickle.HIGHEST_PROTOCOL)
        tmp.replace(p)


class EmbeddingConfig(BaseModel):
    """One named vector slot in a Qdrant collection.

    Extend a collection by appending another EmbeddingConfig to the indexer's
    `embeddings` list — the slot's `name` becomes the named-vector key on every
    PointStruct.
    """

    model_config = ConfigDict(protected_namespaces=())

    name: str
    model_id: str
    kind: Literal["dense", "sparse"]
    size: int | None = None
    distance: Distance = Distance.COSINE
    providers: list[str] | None = None
    parallel: int | None = None
    """fastembed multi-process workers for CPU inference. None = single process,
    onnxruntime's own intra-op threading. Each worker loads its OWN full model
    copy — N workers means N times the model's on-disk size in RAM before any
    inference buffers, e.g. bge-small (0.067GB) at 4 is ~270MB, but a 2GB+
    model at 4 is 8GB+. Scale to CPU cores - 1 only for small models; leave
    None for anything bge-large-sized or up, and always when using GPU
    providers."""
    modifier: Modifier | None = Modifier.IDF
    """Qdrant query-time scoring modifier for sparse slots. Qdrant/bm25 vectors
    carry only the TF/length half of BM25 and expect `Modifier.IDF` on the
    collection — without it, matches are scored TF-only and rare tokens get no
    rarity weight. Collection schema: changing it requires recreate + re-upload.
    Defaulted rather than passed per call site, because a missed one creates a
    TF-only collection that every later `ensure_collection` skips. Pass None
    explicitly for a learned-sparse model (SPLADE) that bakes in its own
    rarity weights; ignored for dense slots."""
    query_prompt: str = ""
    doc_prompt: str = ""
    """Retrieval prefixes prepended before embedding — the role marker asymmetric
    dense models need and won't infer. e5 wants `query: ` / `passage: `; Qwen3
    wants an `Instruct: ...\\nQuery: ` on queries and nothing on docs; bge-small
    needs neither, so both default empty and leg-1 embeds exactly as before.
    Applied to text, so cached vectors already carry the prefix — one prompt per
    (model, run); change it and re-index under a fresh namespace."""
    cloud: bool = False
    """Embed via Qdrant Cloud Inference (`Document`) instead of local fastembed
    — for a model with no local ONNX build (e.g. an OpenRouter-hosted one).
    Per-slot, not per-client: the client still needs `cloud_inference=True` to
    construct, but that flag only activates handling for the slots marked
    here, so a dense OpenRouter model and a local sparse BM25 can share one
    indexer without BM25 losing its embedding cache."""
    provider_options: dict[str, Any] | None = None
    """Passed verbatim as `Document.options` — provider auth/routing, e.g.
    `{"openrouter-api-key": ...}`. Qdrant forwards it to the inference service
    as-is (Document.options docstring). Never log this config: it carries the
    API key in plain text."""
    model_options: dict[str, Any] | None = None
    """Constructor kwargs for the LOCAL fastembed model — corpus-dependent knobs the
    model cannot infer. miniCOIL is built on BM25 and takes `avg_len` (its default 150
    assumes short docs; crumb-legal-qa averages 272 words once capped), so leaving it
    unset mis-weights length normalization on every document."""
    engine: Literal["fastembed", "sentence_transformers"] = "fastembed"
    """Which local runtime produces this slot's vectors (ignored when `cloud`).
    `fastembed` is the default. `sentence_transformers` routes a SPARSE slot to a
    `SparseEncoder` — for learned-sparse models fastembed does not ship (e.g.
    `opensearch-project/opensearch-neural-sparse-encoding-doc-v3-gte`), whose
    doc/query asymmetry the encoder handles itself, so leave `doc_prompt` empty."""
    sentence_transformers_batch_size: int = Field(default=16, ge=1)
    """Documents per learned-sparse ``SparseEncoder`` call. This is deliberately
    separate from upload batching: an encoder call retains its sparse tensors until it
    returns, so a conservative default bounds MPS/unified-memory use even when Qdrant
    checkpoints thousands of documents at a time. Ignored by fastembed and cloud slots."""
    max_input_chars: int | None = None
    """Truncate each text to this many chars before embedding — hosted models with a hard
    context limit (openai/text-embedding-3-large caps at 8192 tokens, and a 66k-char legal
    doc is ~16k tokens -> 400), and local models whose cost is quadratic in sequence length:
    miniCOIL truncates at 8192 tokens (SPLADE at 512) and pads a batch to its longest member,
    so one 66k-char doc measured 11.45GB alone and OOM-kills a batch of 64. None = no cap,
    e.g. Qwen3's 32k context swallows the whole corpus. A conservative char cap avoids a
    tokenizer dependency."""


_ST_SPARSE_MODELS: dict[str, Any] = {}
_ST_SPARSE_LOCK = threading.Lock()
"""Serializes st_sparse_vectors end to end. Callers reach it from thread pools
(labelling at max_workers>1), and neither half of the function survives that:
the check-then-set cache races into N full transformer loads on MPS, and one
thread's `torch.mps.empty_cache()` fires while another is mid-forward — a
native abort that kills the process without a traceback. Cheap to hold: the
query-side encode is tokenizer-bound, and the win from workers is the cloud
dense round-trips, which stay outside this function."""


def st_sparse_vectors(model_id: str, texts: list[str], *, is_query: bool,
                      max_seq_length: int = 512, batch_size: int = 16,
                      device: str | None = None) -> list[SparseVector]:
    """sentence-transformers `SparseEncoder` -> Qdrant `SparseVector`s, honouring the
    doc/query asymmetry (`encode_query` vs `encode_document`) that learned-sparse models
    fastembed does not ship carry. The model is a full transformer: the first call
    downloads and loads it (real RAM/disk) — a heavy step, so run it under memguard / on
    a machine that can hold it. Cached per `model_id`.

    `max_seq_length` caps the tokens per text: a GTE backbone defaults to ~8k, and the
    sparse head's `[batch, seq, vocab]` projection over long docs (legal corpora hit tens
    of thousands of chars) blows to tens of GiB. 512 also matches fastembed SPLADE's
    truncation, keeping the bake-off fair; raise it if you have the RAM.

    The encoder call must also be bounded, not merely its internal forward batch. Its
    implementation retains every batch's sparse tensor in ``all_embeddings`` until the
    call returns; passing an indexing slice of 2,000 documents left those tensors on MPS
    and its caching allocator eventually consumed tens of GB of unified memory. Encode
    one small batch per call and move it to CPU before converting it to Qdrant objects.
    ``batch_size=16`` is measured-optimal on MPS for doc-v3-distill: 32 is slightly
    slower and 64 collapses ~6x. Do not increase it without re-measuring."""
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")
    from sentence_transformers import SparseEncoder  # heavy; imported only when used

    with _ST_SPARSE_LOCK:
        # `device` joins the key: the query side runs on "cpu" (an IDF lookup —
        # and a wedged MPS op inside this lock froze a whole 24-worker labelling
        # run silently), while doc-side indexing keeps the default (MPS) device.
        key = (model_id, device)
        model = _ST_SPARSE_MODELS.get(key)
        if model is None:
            # trust_remote_code: GTE-backboned models (opensearch-...-gte -> Alibaba-NLP/new-impl)
            # ship custom modeling code. This engine is opt-in per config, so `model_id` is a
            # deliberate, trusted choice, not arbitrary input.
            model = _ST_SPARSE_MODELS[key] = SparseEncoder(
                model_id, trust_remote_code=True, device=device
            )
            model.max_seq_length = max_seq_length
        encode = model.encode_query if is_query else model.encode_document
        vectors: list[SparseVector] = []
        try:
            for texts_batch in _chunked(texts, batch_size):
                rows = encode(
                    texts_batch,
                    batch_size=len(texts_batch),
                    convert_to_tensor=False,
                    convert_to_sparse_tensor=True,
                    save_to_cpu=True,
                    show_progress_bar=False,
                )
                vectors.extend(
                    SparseVector(
                        indices=t.coalesce().indices()[0].tolist(),
                        values=t.coalesce().values().tolist(),
                    )
                    for t in rows
                )
        finally:
            # The batches above leave no live MPS tensors, so this releases only allocator
            # cache. It prevents long indexing runs from keeping prior batches' unified
            # memory and forcing macOS to compress/swap it. A CPU-pinned model never
            # touched MPS — skip entirely so a wedged GPU can't hang this call.
            if device != "cpu":
                import torch

                if torch.backends.mps.is_available():
                    torch.mps.empty_cache()
        return vectors


class BaseIndexer(ABC, Generic[T]):
    """Manage a Qdrant collection of items of type T.

    Subclasses bind T via `item_type` and implement item_id/text/payload. The
    base class owns collection creation, batched embedding across N vector
    slots, and upload.
    """

    item_type: ClassVar[type[BaseModel]]

    def __init__(
        self,
        client: QdrantClient,
        collection_name: str,
        embeddings: list[EmbeddingConfig],
        cache: EmbeddingCache | None = None,
        reuse_cloud_from: str | None = None,
    ) -> None:
        if not embeddings:
            raise ValueError("Indexer requires at least one EmbeddingConfig.")
        self.client = client
        self.collection_name = collection_name
        self.embeddings = embeddings
        self.cache = cache
        self.reuse_cloud_from = reuse_cloud_from
        """Collection to copy cloud-slot vectors from by point id instead of
        re-paying server-side inference — same item => same uuid5 id, so a
        sibling collection on the same corpus already holds identical floats.
        Ids absent there fall back to the paid path."""
        self._dense_models: dict[str, TextEmbedding] = {}
        self._sparse_models: dict[str, SparseTextEmbedding] = {}

    @abstractmethod
    def item_id(self, item: T) -> int | str: ...

    @abstractmethod
    def item_text(self, item: T) -> str: ...

    @abstractmethod
    def item_payload(self, item: T) -> dict[str, Any]: ...

    def ensure_collection(self, recreate: bool = False) -> None:
        """Create the collection with the configured vector slots.

        If the collection already exists and `recreate` is False, no-op. If
        True, drop and recreate so schema changes (new vector slot) take effect.
        """
        exists = self.client.collection_exists(self.collection_name)
        if exists and not recreate:
            return
        if exists:
            self.client.delete_collection(self.collection_name)

        dense_config = {
            cfg.name: VectorParams(size=self._dense_size(cfg), distance=cfg.distance)
            for cfg in self.embeddings
            if cfg.kind == "dense"
        }
        sparse_config = {
            cfg.name: SparseVectorParams(modifier=cfg.modifier)
            for cfg in self.embeddings
            if cfg.kind == "sparse"
        }
        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=dense_config or None,
            sparse_vectors_config=sparse_config or None,
        )

    def missing(self, items: Sequence[T], batch_size: int = 1024) -> list[T]:
        """Items whose point id is not in the collection yet — the upload set
        for a corpus that grew, so adding one document costs one embedding
        instead of re-embedding the lane.

        Identity is the point id alone: edited text under an unchanged id is
        deliberately NOT re-embedded, matching the behaviour of the count check
        this replaces.
        """
        ids = [self.item_id(item) for item in items]
        found: set[str] = set()
        for chunk in _chunked(ids, batch_size):
            found.update(
                str(point.id)
                for point in self.client.retrieve(
                    collection_name=self.collection_name,
                    ids=chunk,
                    with_payload=False,
                    with_vectors=False,
                )
            )
        return [
            item
            for item, point_id in zip(items, ids, strict=True)
            if str(point_id) not in found
        ]

    def upload(
        self,
        items: Sequence[T],
        batch_size: int = 64,
        parallel: int = 1,
    ) -> None:
        """Embed `items` for every configured slot and upload as PointStructs.

        Raises TypeError if items aren't `item_type` — guards against using the
        wrong indexer subclass for the data. `parallel` > 1 sends the per-batch
        upserts through a thread pool: for a CLOUD slot the embedding runs
        server-side inside each upsert, so N concurrent upserts give ~Nx
        throughput on that I/O-bound step (sequential was ~8s/batch on gemini).
        """
        if not items:
            return
        if not isinstance(items[0], self.item_type):
            raise TypeError(
                f"{type(self).__name__} expects {self.item_type.__name__}, "
                f"got {type(items[0]).__name__}"
            )

        # An empty embed text is unembeddable and cloud encoders reject it with a
        # 400 (Qdrant Cloud Inference -> gemini), so drop such items rather than
        # fail the whole batch. Logged, not silent (some corpora carry blank docs).
        kept = [i for i in items if self.item_text(i).strip()]
        if len(kept) < len(items):
            logger.warning(
                "%s: skipped %d item(s) with empty embed text (unembeddable).",
                self.collection_name, len(items) - len(kept),
            )
        items = kept
        if not items:
            return

        texts = [self.item_text(i) for i in items]
        ids = [self.item_id(i) for i in items]
        payloads = [self.item_payload(i) for i in items]

        vectors: list[dict[str, Any]] = [{} for _ in items]
        for cfg in self.embeddings:
            for i, vec in enumerate(self._vectors_for(cfg, ids, texts, batch_size)):
                vectors[i][cfg.name] = vec

        points = [
            PointStruct(id=ids[i], vector=vectors[i], payload=payloads[i])
            for i in range(len(items))
        ]
        # for a cloud slot, embedding happens INSIDE this upsert (Qdrant calls
        # the provider per point), so this loop — not `_embed` — is where a
        # remote-encoder upload actually spends its time; local runs get the
        # same bar for a big corpus's upsert phase, cheaply.
        chunks = list(_chunked(points, batch_size))
        bar = tqdm(total=len(chunks), desc=f"upload:{self.collection_name}")
        if parallel > 1:
            # map yields in submission order; the pool keeps `parallel` upserts
            # in flight, so a cloud slot embeds `parallel` batches concurrently.
            with ThreadPoolExecutor(max_workers=parallel) as pool:
                for _ in pool.map(self._upsert_with_retry, chunks):
                    bar.update(1)
        else:
            for chunk in chunks:
                self._upsert_with_retry(chunk)
                bar.update(1)
        bar.close()

        # Duplicate ids collapse onto one point under upsert, so the
        # completeness check must count unique ids, not rows.
        expected = len({str(p.id) for p in points})
        if expected < len(points):
            logger.warning(
                "%s: %d duplicate point ids in this upload were collapsed by upsert.",
                self.collection_name,
                len(points) - expected,
            )
        actual = self.client.count(self.collection_name, exact=True).count
        if actual < expected:
            raise RuntimeError(
                f"Upload incomplete for {self.collection_name!r}: "
                f"expected >={expected} points, got {actual}."
            )

    def _upsert_with_retry(self, points: list[PointStruct]) -> None:
        for attempt in range(_UPSERT_MAX_ATTEMPTS):
            try:
                self.client.upsert(
                    collection_name=self.collection_name,
                    points=points,
                    wait=True,
                )
                return
            except (UnexpectedResponse, ResponseHandlingException) as exc:
                # ResponseHandlingException wraps a transport failure — a write
                # timeout or dropped connection, which has no status_code and is
                # always worth another try (a heavy upsert stalling the socket is
                # the one that actually kills a long unattended run). An
                # UnexpectedResponse carries an HTTP status: retry the transient
                # ones, fail fast on the rest (a 403 exhausted key no backoff can
                # fix, retried 4x is 4x the wasted calls).
                status = getattr(exc, "status_code", None)
                transient = isinstance(exc, ResponseHandlingException) or status in _RETRYABLE_STATUS
                if not transient or attempt == _UPSERT_MAX_ATTEMPTS - 1:
                    raise
                delay = _UPSERT_BACKOFF_S * (2**attempt)
                tqdm.write(
                    f"{self.collection_name}: upsert {status or type(exc).__name__} "
                    f"(attempt {attempt + 1}/{_UPSERT_MAX_ATTEMPTS}), "
                    f"retrying in {delay:.0f}s"
                )
                time.sleep(delay)
                remaining = self._uncommitted(points)
                if points and not remaining:
                    return
                points = remaining

    def _uncommitted(self, points: list[PointStruct]) -> list[PointStruct]:
        """The points Qdrant does not already hold. A cloud slot embeds INSIDE
        the upsert, so the server may bill a point, store it, and still fail
        the response — a blind retry then buys the same embedding again, up to
        four times over a rate-limited run.

        If the read itself fails (the same network trouble that triggered the
        retry can stall this retrieve), fall back to resending everything: for a
        local slot it costs only bandwidth, and re-billing one slice's cloud
        embeddings beats aborting the whole run on a transient read error."""
        if not points:
            return []
        try:
            stored = {
                str(point.id)
                for point in self.client.retrieve(
                    collection_name=self.collection_name,
                    ids=[p.id for p in points],
                    with_payload=False,
                    with_vectors=False,
                )
            }
        except (UnexpectedResponse, ResponseHandlingException):
            return points
        return [p for p in points if str(p.id) not in stored]

    def upload_points_iter(
        self,
        items: Iterable[T | dict[str, Any]],
        batch_size: int = 64,
        parallel: int = 1,
        max_retries: int = 3,
    ) -> None:
        """Stream items into Qdrant via `client.upload_points`.

        Uses qdrant-client's built-in bulk uploader: parallel worker pool
        for network I/O, built-in retries, and its own tqdm bar via
        `show_progress=True`. Best for very large corpora where upsert
        latency dominates.

        `batch_size` controls both embedding chunk size and upsert chunk size.
        `parallel` sets the number of worker threads inside qdrant-client.

        Note: bypasses `EmbeddingCache` — the qdrant-client streaming path
        doesn't compose with the (id -> vector) cache. For repeated runs on
        the same items with local models, use `upload`.
        """
        # No show_progress: qdrant-client 1.18's upload_points asserts no unknown
        # kwargs. Progress isn't observable here anyway — for a cloud slot the work
        # is server-side inside the parallel upserts, not in the point generator.
        self.client.upload_points(
            collection_name=self.collection_name,
            points=self._points_iter(items, batch_size),
            batch_size=batch_size,
            parallel=parallel,
            max_retries=max_retries,
        )

    def _points_iter(
        self,
        items: Iterable[T | dict[str, Any]],
        batch_size: int,
    ) -> Iterator[PointStruct]:
        """Yield PointStructs lazily, embedding one batch at a time."""
        batch: list[T] = []
        for item in items:
            if isinstance(item, dict):
                item = self.item_type.model_validate(item)
            batch.append(item)  # type: ignore[arg-type]
            if len(batch) >= batch_size:
                yield from self._batch_to_points(batch, batch_size)
                batch = []
        if batch:
            yield from self._batch_to_points(batch, batch_size)

    def _batch_to_points(
        self,
        batch: list[T],
        batch_size: int,
    ) -> Iterator[PointStruct]:
        # Same empty-embed-text guard as `upload` — the streaming/parallel path
        # must drop unembeddable items too, or a cloud slot 400s on Document("").
        pairs = [(i, t) for i in batch if (t := self.item_text(i)).strip()]
        if len(pairs) < len(batch):
            logger.warning(
                "%s: skipped %d item(s) with empty embed text (unembeddable).",
                self.collection_name, len(batch) - len(pairs),
            )
        if not pairs:
            return
        batch = [i for i, _ in pairs]
        texts = [t for _, t in pairs]
        ids = [self.item_id(i) for i in batch]
        payloads = [self.item_payload(i) for i in batch]
        vectors: list[dict[str, Any]] = [{} for _ in batch]
        for cfg in self.embeddings:
            for i, vec in enumerate(self._embed(cfg, texts, batch_size)):
                vectors[i][cfg.name] = vec
        for i in range(len(batch)):
            yield PointStruct(id=ids[i], vector=vectors[i], payload=payloads[i])

    def _vectors_for(
        self,
        cfg: EmbeddingConfig,
        ids: list[int | str],
        texts: list[str],
        batch_size: int,
    ) -> list[Any]:
        """Return vectors aligned with `ids`/`texts`, hitting cache where possible."""
        if cfg.cloud and self.reuse_cloud_from:
            return self._reused_vectors(cfg, ids, texts, batch_size)
        if self.cache is None or cfg.cloud:
            return self._embed(cfg, texts, batch_size)

        store = self.cache
        cache = store.load(cfg.model_id, cfg.kind)

        missing_idx = [i for i, id_ in enumerate(ids) if str(id_) not in cache]
        for chunk in _chunked(missing_idx, _EMBED_CHECKPOINT_SIZE):
            chunk_texts = [texts[i] for i in chunk]
            new_vecs = self._embed(cfg, chunk_texts, batch_size)
            for original_i, vec in zip(chunk, new_vecs, strict=True):
                cache[str(ids[original_i])] = vec
            store.save(cfg.model_id, cfg.kind)

        return [cache[str(id_)] for id_ in ids]

    def _reused_vectors(
        self,
        cfg: EmbeddingConfig,
        ids: list[int | str],
        texts: list[str],
        batch_size: int,
    ) -> list[Any]:
        """Copy this slot's vectors from `reuse_cloud_from` by point id; only ids
        the source collection doesn't hold go through the paid `_embed` path."""
        found: dict[str, Any] = {}
        for chunk in _chunked(list(ids), 256):
            for point in self.client.retrieve(
                self.reuse_cloud_from, ids=list(chunk),
                with_vectors=[cfg.name], with_payload=False,
            ):
                vec = point.vector
                if isinstance(vec, dict):
                    vec = vec.get(cfg.name)
                if vec is not None:
                    found[str(point.id)] = vec
        missing = [i for i, id_ in enumerate(ids) if str(id_) not in found]
        if missing:
            logger.warning(
                "%s: %d/%d ids absent from %s — embedding those via the paid path.",
                self.collection_name, len(missing), len(ids), self.reuse_cloud_from,
            )
            embedded = self._embed(cfg, [texts[i] for i in missing], batch_size)
            for i, vec in zip(missing, embedded, strict=True):
                found[str(ids[i])] = vec
        return [found[str(id_)] for id_ in ids]

    def _embed(
        self,
        cfg: EmbeddingConfig,
        texts: list[str],
        batch_size: int,
    ) -> list[Any]:
        if cfg.doc_prompt:  # role marker for asymmetric dense models; "" for bm25/bge
            texts = [cfg.doc_prompt + t for t in texts]
        if cfg.max_input_chars:  # cloud context limits AND local quadratic-cost blowups
            texts = [t[: cfg.max_input_chars] for t in texts]
        if cfg.cloud:  # this slot only; a sibling local slot embeds unaffected
            return [
                Document(text=text, model=cfg.model_id, options=cfg.provider_options)
                for text in texts
            ]

        if cfg.kind == "dense":
            model = self._dense(cfg)
            stream = tqdm(
                model.embed(texts, batch_size=batch_size, parallel=cfg.parallel),
                total=len(texts),
                desc=f"embed:{cfg.name}",
            )
            return [np.asarray(v, dtype=np.float32) for v in stream]

        if cfg.engine == "sentence_transformers":  # learned-sparse via SparseEncoder (docs)
            return st_sparse_vectors(
                cfg.model_id,
                texts,
                is_query=False,
                batch_size=cfg.sentence_transformers_batch_size,
            )
        model = self._sparse(cfg)
        stream = tqdm(
            model.embed(texts, batch_size=batch_size, parallel=cfg.parallel),
            total=len(texts),
            desc=f"embed:{cfg.name}",
        )
        return [
            SparseVector(
                indices=np.asarray(s.indices, dtype=np.int64).tolist(),
                values=np.asarray(s.values, dtype=np.float32).tolist(),
            )
            for s in stream
        ]

    def _dense(self, cfg: EmbeddingConfig) -> TextEmbedding:
        if cfg.model_id not in self._dense_models:
            self._dense_models[cfg.model_id] = TextEmbedding(
                cfg.model_id, providers=cfg.providers
            )
        return self._dense_models[cfg.model_id]

    def _sparse(self, cfg: EmbeddingConfig) -> SparseTextEmbedding:
        options = cfg.model_options or {}
        key = f"{cfg.model_id}:{sorted(options.items())}"  # options change the vectors
        if key not in self._sparse_models:
            self._sparse_models[key] = SparseTextEmbedding(
                cfg.model_id, providers=cfg.providers, **options
            )
        return self._sparse_models[key]

    @staticmethod
    def _dense_size(cfg: EmbeddingConfig) -> int:
        if cfg.size is None:
            raise ValueError(f"Dense embedding {cfg.name!r} requires `size`.")
        return cfg.size


class CorpusDocument(BaseModel):
    doc_id: str
    title: str = ""  # auxiliary — prepended to text for embedding; some corpora omit it
    text: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    """Extra payload preserved alongside the canonical fields — not embedded."""


class CorpusIndexer(BaseIndexer[CorpusDocument]):
    """Concrete indexer for IR corpus documents.

    item_id uses UUID5 of doc_id so it works for both numeric and
    string-keyed corpora (Qdrant requires int or UUID point IDs).
    The original doc_id is preserved in the payload for filtering/lookup.
    """

    item_type = CorpusDocument

    def item_id(self, item: CorpusDocument) -> str:
        return str(uuid.uuid5(uuid.NAMESPACE_DNS, item.doc_id))

    def item_text(self, item: CorpusDocument) -> str:
        return f"{item.title} {item.text}".strip()

    def item_payload(self, item: CorpusDocument) -> dict[str, Any]:
        # Canonical fields win over metadata on key collision.
        return {
            **item.metadata,
            "doc_id": item.doc_id,
            "title": item.title,
            "text": item.text,
        }
