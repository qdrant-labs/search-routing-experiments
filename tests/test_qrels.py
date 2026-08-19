"""`AugmentationQrels.backfill()` — a pure retry of the inherit-path lookup
`mint()` already does, for pool rows whose lane wasn't fully materialized at
mint time (2026-08 follow-up). No new judgment source: only recovers what
was already on disk somewhere and just wasn't visible yet.
"""

import pandas as pd

from augmentation.config import AugmentationPaths
from augmentation.core import AnswerKeyPath, AugmentedCandidate
from augmentation.qrels import AugmentationQrels

LANE = "beir-nfcorpus"


def test_mint_writes_parent_grades_against_every_grounding_doc(tmp_path):
    """d43d/#6 fix: Inject mints against ALL judged docs the surface reaches,
    at the parent's REAL grades — not one synthetic relevance=1 that a graded
    lane thresholds straight to all_zero."""
    paths = AugmentationPaths(data_dir=tmp_path)
    lane_dir = paths.data_dir / LANE
    lane_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([
        {"query_id": "p1", "doc_id": "docA", "relevance": 2},
        {"query_id": "p1", "doc_id": "docB", "relevance": 1},  # no surface
        {"query_id": "p1", "doc_id": "docC", "relevance": 3},
    ]).to_parquet(lane_dir / "qrels.parquet", index=False)
    candidate = AugmentedCandidate(
        query_id="aug-id-tech-p1", query="laptops SKU-1", floor="id:tech",
        operator="inject", provenance="doc_grounded", generated_from="p1",
        parent_dataset=LANE, home_lane=LANE,
        grounding_doc_id="docA", grounding_doc_ids=("docA", "docC"),
        meaning_preserved=False, answer_key=AnswerKeyPath.MINTED, attempts=1,
    )

    written = AugmentationQrels(paths).mint(candidate)

    assert written == 2
    saved = AugmentationQrels(paths).load()
    assert dict(zip(saved["doc_id"], saved["relevance"])) == {"docA": 2, "docC": 3}
    assert set(saved["source"]) == {"constructed"}


def _pool_row(query_id: str, generated_from: str, answer_key: str = "inherit") -> dict:
    return {
        "query_id": query_id, "generated_from": generated_from,
        "home_lane": LANE, "answer_key": answer_key,
    }


def test_backfill_recovers_nothing_when_the_lane_still_has_no_judgments(tmp_path):
    paths = AugmentationPaths(data_dir=tmp_path)
    pool = pd.DataFrame([_pool_row("aug-q1", "p1")])

    still_unlabelable = AugmentationQrels(paths).backfill(pool)

    assert list(still_unlabelable["query_id"]) == ["aug-q1"]
    assert not paths.qrels.exists()   # nothing to write — no spurious file


def test_backfill_recovers_once_the_lane_catches_up(tmp_path):
    """The exact scenario found in production: a parent's judgment didn't
    exist when the row was first minted, but does now."""
    paths = AugmentationPaths(data_dir=tmp_path)
    pool = pd.DataFrame([
        _pool_row("aug-q1", "p1"),
        _pool_row("aug-q2", "p2"),
    ])
    qrels = AugmentationQrels(paths)

    first_pass = qrels.backfill(pool)
    assert set(first_pass["query_id"]) == {"aug-q1", "aug-q2"}

    lane_dir = paths.data_dir / LANE
    lane_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([
        {"query_id": "p1", "doc_id": "docA", "relevance": 1},
        {"query_id": "p2", "doc_id": "docB", "relevance": 1},
    ]).to_parquet(lane_dir / "qrels.parquet", index=False)

    second_pass = qrels.backfill(pool)

    assert second_pass.empty
    saved = qrels.load()
    assert set(saved["query_id"]) == {"aug-q1", "aug-q2"}
    assert set(saved["source"]) == {"human"}
    assert list(saved.loc[saved["query_id"] == "aug-q1", "doc_id"]) == ["docA"]


def test_backfill_never_retries_a_row_that_already_has_a_qrels_entry(tmp_path):
    paths = AugmentationPaths(data_dir=tmp_path)
    pool = pd.DataFrame([_pool_row("aug-q1", "p1")])
    qrels = AugmentationQrels(paths)
    paths.qrels.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{
        "query_id": "aug-q1", "doc_id": "already-there", "relevance": 1,
        "source": "human", "inherited_from": "p1",
    }]).to_parquet(paths.qrels, index=False)

    still_unlabelable = qrels.backfill(pool)

    assert still_unlabelable.empty
    assert list(qrels.load()["doc_id"]) == ["already-there"]   # untouched


def test_backfill_reports_an_orphaned_minted_row_without_retrying_it(tmp_path, capsys):
    """A minted row's answer key has no external dependency at mint time —
    if one is somehow orphaned anyway, that's a different bug, and backfill
    must not paper over it by treating it as an inherit lookup."""
    paths = AugmentationPaths(data_dir=tmp_path)
    pool = pd.DataFrame([_pool_row("aug-q1", "p1", answer_key="minted")])

    still_unlabelable = AugmentationQrels(paths).backfill(pool)

    assert still_unlabelable.empty   # not even reported as "unlabelable" — reported separately
    assert "unexpected" in capsys.readouterr().out
