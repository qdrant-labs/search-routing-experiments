"""spec verification (10): a corpus or retrieval fingerprint change
invalidates reuse. Identity-only match is a miss."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from rungs.cache import CacheKey, LabelCache
from rungs.cost_planner import CostPlanner


def _label_row(**overrides) -> dict[str, object]:
    base = {
        "dataset": "A",
        "query_id": "q1",
        "query_fp": "q-fp",
        "corpus_fp": "c-fp",
        "qrels_fp": "qr-fp",
        "retrieval_stack_fp": "rs-fp",
        "route_class": "dense",
    }
    base.update(overrides)
    return base


def test_full_key_match_hits(tmp_path: Path):
    label_path = tmp_path / "labels.parquet"
    pd.DataFrame([_label_row()]).to_parquet(label_path, index=False)
    cache = LabelCache([label_path])
    key = CacheKey("A", "q1", "q-fp", "c-fp", "qr-fp", "rs-fp")
    assert cache.has(key)


def test_corpus_fp_change_is_a_miss(tmp_path: Path):
    label_path = tmp_path / "labels.parquet"
    pd.DataFrame([_label_row()]).to_parquet(label_path, index=False)
    cache = LabelCache([label_path])
    key = CacheKey("A", "q1", "q-fp", "different-c-fp", "qr-fp", "rs-fp")
    assert not cache.has(key), "corpus fp change must invalidate reuse"


def test_retrieval_stack_fp_change_is_a_miss(tmp_path: Path):
    label_path = tmp_path / "labels.parquet"
    pd.DataFrame([_label_row()]).to_parquet(label_path, index=False)
    cache = LabelCache([label_path])
    key = CacheKey("A", "q1", "q-fp", "c-fp", "qr-fp", "different-rs-fp")
    assert not cache.has(key), "retrieval-stack fp change must invalidate reuse"


def test_labels_without_fingerprints_are_ignored(tmp_path: Path):
    """A legacy label file without the 4 fp columns must be treated as if
    it does not exist — spec:151-153."""
    legacy_path = tmp_path / "legacy_labels.parquet"
    pd.DataFrame([{
        "dataset": "A", "query_id": "q1", "route_class": "dense",
    }]).to_parquet(legacy_path, index=False)
    cache = LabelCache([legacy_path])
    assert cache.size() == 0, "labels missing fp columns must not appear as reusable"


def test_cost_planner_never_reorders_plan(tmp_path: Path):
    label_path = tmp_path / "labels.parquet"
    pd.DataFrame([_label_row()]).to_parquet(label_path, index=False)
    cache = LabelCache([label_path])
    plan = pd.DataFrame({"dataset": ["A", "A", "A"], "query_id": ["q3", "q1", "q2"]})
    fps = {
        ("A", "q3"): ("", "", "", ""),
        ("A", "q1"): ("q-fp", "c-fp", "qr-fp", "rs-fp"),
        ("A", "q2"): ("", "", "", ""),
    }
    part = CostPlanner(cache).partition(plan, fingerprints=fps, budget=10)
    assert len(part.hits) == 1
    assert part.hits.iloc[0]["query_id"] == "q1"
    # misses preserve original plan order
    assert part.misses["query_id"].tolist() == ["q3", "q2"]
