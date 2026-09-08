"""The sentence-transformers SparseEncoder engine: conversion to Qdrant SparseVector
and doc/query routing, exercised without downloading a model (a fake encoder is injected
into the module cache)."""

import torch

from hybrid_search_rrf_dataset import indexer


class _FakeSparseEncoder:
    """Returns one 1-D sparse COO tensor per input: nonzero at [2, 5] = [0.5, 0.25]."""

    def encode_document(self, texts, **_):
        return [torch.sparse_coo_tensor(torch.tensor([[2, 5]]), torch.tensor([0.5, 0.25]), (10,))
                for _ in texts]

    def encode_query(self, texts, **_):
        return [torch.sparse_coo_tensor(torch.tensor([[1]]), torch.tensor([0.75]), (10,))
                for _ in texts]


class _RecordingSparseEncoder(_FakeSparseEncoder):
    def __init__(self):
        self.document_calls = []

    def encode_document(self, texts, **kwargs):
        self.document_calls.append((list(texts), kwargs))
        return super().encode_document(texts, **kwargs)


def test_st_sparse_vectors_converts_and_routes():
    indexer._ST_SPARSE_MODELS[("fake", None)] = _FakeSparseEncoder()
    try:
        docs = indexer.st_sparse_vectors("fake", ["a", "b"], is_query=False)
        qry = indexer.st_sparse_vectors("fake", ["q"], is_query=True)
    finally:
        indexer._ST_SPARSE_MODELS.pop(("fake", None), None)

    assert len(docs) == 2
    assert docs[0].indices == [2, 5] and docs[0].values == [0.5, 0.25]   # encode_document path
    assert len(qry) == 1
    assert qry[0].indices == [1] and qry[0].values == [0.75]             # encode_query path (asymmetric)


def test_st_sparse_vectors_bounds_encoder_calls_and_offloads_results():
    """A 2,000-doc index slice must not make SparseEncoder retain 2,000 MPS tensors."""
    encoder = _RecordingSparseEncoder()
    indexer._ST_SPARSE_MODELS[("recording", None)] = encoder
    try:
        vectors = indexer.st_sparse_vectors(
            "recording", [str(i) for i in range(33)], is_query=False, batch_size=16
        )
    finally:
        indexer._ST_SPARSE_MODELS.pop(("recording", None), None)

    assert len(vectors) == 33
    assert [len(texts) for texts, _ in encoder.document_calls] == [16, 16, 1]
    for texts, kwargs in encoder.document_calls:
        assert kwargs["batch_size"] == len(texts)
        assert kwargs["save_to_cpu"] is True


def test_st_sparse_vectors_loads_model_once_under_threads(monkeypatch):
    """Concurrent first calls (labelling at max_workers>1) must not each load a
    full transformer: the unlocked check-then-set raced into N MPS model loads
    and a native crash."""
    import time
    from concurrent.futures import ThreadPoolExecutor

    import sentence_transformers

    loads: list[str] = []

    class _SlowLoadEncoder(_FakeSparseEncoder):
        def __init__(self, model_id, **_):
            loads.append(model_id)
            time.sleep(0.05)  # wide race window: unlocked, all 8 threads load

    monkeypatch.setattr(sentence_transformers, "SparseEncoder", _SlowLoadEncoder)
    try:
        with ThreadPoolExecutor(max_workers=8) as pool:
            results = list(pool.map(
                lambda _: indexer.st_sparse_vectors("race-model", ["q"], is_query=True),
                range(8),
            ))
    finally:
        indexer._ST_SPARSE_MODELS.pop(("race-model", None), None)

    assert loads == ["race-model"]
    assert all(len(r) == 1 and r[0].indices == [1] for r in results)
