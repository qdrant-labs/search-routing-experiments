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


def test_st_sparse_vectors_converts_and_routes():
    indexer._ST_SPARSE_MODELS["fake"] = _FakeSparseEncoder()
    try:
        docs = indexer.st_sparse_vectors("fake", ["a", "b"], is_query=False)
        qry = indexer.st_sparse_vectors("fake", ["q"], is_query=True)
    finally:
        indexer._ST_SPARSE_MODELS.pop("fake", None)

    assert len(docs) == 2
    assert docs[0].indices == [2, 5] and docs[0].values == [0.5, 0.25]   # encode_document path
    assert len(qry) == 1
    assert qry[0].indices == [1] and qry[0].values == [0.75]             # encode_query path (asymmetric)
