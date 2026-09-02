"""The /classify/pure endpoint returns only route strings — no model load."""

from __future__ import annotations

import pytest
from fastapi import HTTPException

from router_service import api
from router_service.api import ClassifyRequest, classify, classify_pure


class _FakeRouter:
    meta = {"arm": "fake"}

    def classify(self, queries, delta):
        return [
            {"query": q, "route": "dense_only" if "d" in q else "sparse_only"}
            for q in queries
        ]


def test_classify_pure_strips_wrapper_to_route_strings():
    api._state["router"] = _FakeRouter()
    try:
        pure = classify_pure(ClassifyRequest(queries=["dog", "cat"]))
        assert pure == ["dense_only", "sparse_only"]
        assert all(isinstance(route, str) for route in pure)
        # same routing as /classify, just without the arm/query wrapper
        full = classify(ClassifyRequest(queries=["dog"]))
        assert full["results"][0]["route"] == pure[0]
    finally:
        api._state.clear()


def test_classify_pure_503_when_router_not_loaded():
    api._state.clear()
    with pytest.raises(HTTPException) as exc:
        classify_pure(ClassifyRequest(queries=["x"]))
    assert exc.value.status_code == 503
