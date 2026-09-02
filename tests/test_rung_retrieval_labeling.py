"""Retrieval execution preserves the immutable Rung A query surface."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from rungs import retrieval_labeling


def test_plan_query_text_overrides_snapshot_text(tmp_path: Path, monkeypatch):
    lane_dir = tmp_path / "lane-a"
    lane_dir.mkdir()
    pd.DataFrame([
        {"query_id": "1", "text": "stale snapshot text"},
        {"query_id": "2", "text": "unselected"},
    ]).to_parquet(lane_dir / "queries.parquet", index=False)
    pd.DataFrame([
        {"query_id": "1", "doc_id": "d1", "relevance": 1},
        {"query_id": "2", "doc_id": "d2", "relevance": 1},
    ]).to_parquet(lane_dir / "qrels.parquet", index=False)
    pd.DataFrame([
        {"doc_id": "d1", "title": "", "text": "one"},
        {"doc_id": "d2", "title": "", "text": "two"},
    ]).to_parquet(lane_dir / "corpus.parquet", index=False)
    monkeypatch.setattr(retrieval_labeling, "DATA_DIR", tmp_path)

    source = retrieval_labeling._plan_source(
        "lane-a",
        pd.DataFrame([
            {
                "dataset": "lane-a",
                "query_id": "1",
                "query": "frozen plan text",
                "provenance": "natural",
            }
        ]),
    )

    assert source.queries().to_dict("records") == [
        {"query_id": "1", "text": "frozen plan text"}
    ]
    assert source.qrels().to_dict("records") == [
        {"query_id": "1", "doc_id": "d1", "relevance": 1}
    ]
