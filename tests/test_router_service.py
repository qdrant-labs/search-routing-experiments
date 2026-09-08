"""Endpoint unit tests, no model load: /classify/pure route strings, and the
guarded /reload arm-swap (gate, path guard, swap, serve-safe rejection)."""

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


def test_reload_disabled_by_default_returns_403(monkeypatch):
    monkeypatch.setattr(api, "ALLOW_RELOAD", False)
    with pytest.raises(HTTPException) as exc:
        api.reload_arm(api.ReloadRequest(arm="zipf_shape_nocorpus"))
    assert exc.value.status_code == 403


def test_reload_404_for_unknown_or_traversal_arm(tmp_path, monkeypatch):
    monkeypatch.setattr(api, "ALLOW_RELOAD", True)
    monkeypatch.setattr(api, "ARM_LIBRARY", tmp_path)   # empty library
    for bad in ("nonexistent", "../escape"):            # missing, and outside the library
        with pytest.raises(HTTPException) as exc:
            api.reload_arm(api.ReloadRequest(arm=bad))
        assert exc.value.status_code == 404


def test_reload_swaps_router_for_valid_arm(tmp_path, monkeypatch):
    (tmp_path / "good_arm").mkdir()
    (tmp_path / "good_arm" / "meta.json").write_text("{}")
    monkeypatch.setattr(api, "ALLOW_RELOAD", True)
    monkeypatch.setattr(api, "ARM_LIBRARY", tmp_path)
    monkeypatch.setattr(api, "ROUTER_DIR", "orig")      # restored on teardown despite the global reassign
    monkeypatch.setattr(api, "SavedRouter", type("_Stub", (), {"load": staticmethod(lambda d: _FakeRouter())}))
    api._state.clear()
    try:
        out = api.reload_arm(api.ReloadRequest(arm="good_arm"))
        assert out["status"] == "reloaded" and out["arm"] == "fake"
        assert isinstance(api._state["router"], _FakeRouter)
        assert api.ROUTER_DIR.endswith("good_arm")      # live version reflects the swap
    finally:
        api._state.clear()


def test_reload_400_when_arm_not_serve_safe(tmp_path, monkeypatch):
    (tmp_path / "feat_arm").mkdir()
    (tmp_path / "feat_arm" / "meta.json").write_text("{}")
    monkeypatch.setattr(api, "ALLOW_RELOAD", True)
    monkeypatch.setattr(api, "ARM_LIBRARY", tmp_path)

    def _boom(_):
        raise ValueError("needs the taxonomy extractor at inference")

    monkeypatch.setattr(api, "SavedRouter", type("_Boom", (), {"load": staticmethod(_boom)}))
    with pytest.raises(HTTPException) as exc:
        api.reload_arm(api.ReloadRequest(arm="feat_arm"))
    assert exc.value.status_code == 400
