"""spec verification (3): unmaterialized registry candidates can participate
when their answer manifest guarantees corpus inclusion; and (4): raw,
unadmitted augmentation rows cannot enter the catalog."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from rungs.catalog import CandidateCatalog, _admitted_augmentation


def _make_pool_row(query_id: str, gate: str) -> dict[str, object]:
    return {
        "query_id": query_id,
        "query": f"query {query_id}",
        "floor": "cell_x",
        "operator": "corrupt",
        "provenance": "augmented",
        "generated_from": "parent-1",
        "parent_dataset": "A",
        "home_lane": "A",
        "grounding_doc_id": None,
        "meaning_preserved": True,
        "answer_key": "inherit",
        "attempts": 1,
        "credit_gate": gate,
        "hops": None,
        "tokens": None,
        "elapsed_s": None,
        "grounding_doc_ids": None,
    }


def test_ungated_rows_are_admitted(tmp_path: Path):
    pool = pd.DataFrame([_make_pool_row("aug-1", "none"), _make_pool_row("aug-2", "none")])
    admitted = _admitted_augmentation(pool, tmp_path)
    assert set(admitted["query_id"]) == {"aug-1", "aug-2"}


def test_coherence_gated_rows_need_verdict_true(tmp_path: Path):
    pool = pd.DataFrame([
        _make_pool_row("aug-good", "coherence_gate"),
        _make_pool_row("aug-bad", "coherence_gate"),
        _make_pool_row("aug-unaudited", "coherence_gate"),
    ])
    audit = pd.DataFrame([
        {"query_id": "aug-good", "verdict": True, "floor": "cell_x",
         "reason": "", "model": "", "judged_at": ""},
        {"query_id": "aug-bad", "verdict": False, "floor": "cell_x",
         "reason": "", "model": "", "judged_at": ""},
    ])
    audit.to_parquet(tmp_path / "coherence_audit.parquet")
    admitted = _admitted_augmentation(pool, tmp_path)
    assert set(admitted["query_id"]) == {"aug-good"}, (
        "raw pool rows must be admitted only via a passing coherence verdict; "
        "unaudited and failed rows must not enter the candidate catalog"
    )


def test_declaration_audit_needs_query_id_in_passlist(tmp_path: Path):
    pool = pd.DataFrame([
        _make_pool_row("aug-cleared", "declaration_audit"),
        _make_pool_row("aug-uncleared", "declaration_audit"),
    ])
    (tmp_path / "declaration_audit_passed.txt").write_text(
        "# reviewed\naug-cleared\n"
    )
    admitted = _admitted_augmentation(pool, tmp_path)
    assert set(admitted["query_id"]) == {"aug-cleared"}


def test_unknown_gate_is_rejected(tmp_path: Path):
    pool = pd.DataFrame([_make_pool_row("aug-x", "nonexistent_gate")])
    admitted = _admitted_augmentation(pool, tmp_path)
    assert admitted.empty, "unknown gate types must default to reject, not admit"


def test_coverage_is_content_level_not_id_level(tmp_path: Path):
    """A gold doc that exists by id but has empty content does NOT cover its
    answer — counting it covered mints false all-zero rows (the doc-grounded
    extraction-failure bug). Coverage must read text, not just doc_id."""
    lane = "unknown-lane"  # absent from LANES -> min_relevance defaults to 1
    lane_dir = tmp_path / lane
    lane_dir.mkdir()
    pd.DataFrame({
        "doc_id": ["real", "empty"],
        "title": ["", ""],
        "text": ["actual content here", ""],   # 'empty' has a valid id but no body
    }).to_parquet(lane_dir / "corpus.parquet")
    pd.DataFrame({
        "query_id": ["q_ok", "q_empty_gold"],
        "doc_id": ["real", "empty"],
        "relevance": [1, 1],
    }).to_parquet(lane_dir / "qrels.parquet")

    covered, _ = CandidateCatalog(data_dir=tmp_path, datasets=[])._coverage_index(lane)

    assert set(covered) == {"q_ok"}, (
        "the empty-content gold doc must not cover q_empty_gold; an id-level "
        "check would wrongly admit it"
    )
