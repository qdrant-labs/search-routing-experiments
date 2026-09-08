"""Stress tests for the parent pool (d51g) and `CellFill.admit` (d51j): the
structural guarantee that a spent query can never come back as a parent, and
that admission only ever credits a row that actually measures into its cell.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import augmentation.parents as parents
from augmentation.config import AugmentationPaths
from augmentation.parents import ParentPool
from augmentation.pool import GeneratedPool
from composition.cellfill import CellFill
from composition.floors import identifier_floor_key
from query_taxonomy.features import FeatureExtractor

CELL = "bare_concept_token"
"""A real, simple two-band cell (length_words<3, number<1) from cells.yaml —
reused rather than invented, so admission is proven against the actual
registry `CellFill.admit` dispatches through."""


def _pool(paths: AugmentationPaths, catalog=None, selection=None) -> ParentPool:
    return ParentPool(
        catalog if catalog is not None else pd.DataFrame(),
        selection if selection is not None else pd.DataFrame(),
        {},
        pool=GeneratedPool(paths),
        paths=paths,
    )


# --------------------------------------------------------------------------
# available()
# --------------------------------------------------------------------------


def test_available_excludes_a_row_at_every_stage(tmp_path):
    """A row the selection holds must be excluded whether it is a candidate,
    a reused row or a control row — available() must not special-case any."""
    catalog = pd.DataFrame({
        "dataset": ["A", "A", "A", "A"],
        "query_id": ["cand", "reused", "ctrl", "free"],
        "checkable": [True, True, True, True],
    })
    selection = pd.DataFrame({
        "dataset": ["A", "A", "A"],
        "query_id": ["cand", "reused", "ctrl"],
        "stage": ["candidate", "reused", "control"],
    })
    available = _pool(AugmentationPaths(data_dir=tmp_path), catalog, selection).available()
    assert list(available["query_id"]) == ["free"]


def test_available_excludes_an_unselected_non_checkable_row(tmp_path):
    """A row nobody selected but that is not checkable must still be
    excluded — checkable is a hard gate, not a fallback."""
    catalog = pd.DataFrame({
        "dataset": ["A"], "query_id": ["q1"], "checkable": [False],
    })
    selection = pd.DataFrame(columns=["dataset", "query_id"])
    available = _pool(AugmentationPaths(data_dir=tmp_path), catalog, selection).available()
    assert available.empty


def test_available_key_is_the_dataset_and_query_id_pair(tmp_path):
    """Two lanes may legitimately share a query_id, so selecting one lane's
    row must not exclude the other lane's row of the same id."""
    catalog = pd.DataFrame({
        "dataset": ["A", "B"], "query_id": ["q1", "q1"],
        "checkable": [True, True],
    })
    selection = pd.DataFrame({"dataset": ["A"], "query_id": ["q1"]})
    available = _pool(AugmentationPaths(data_dir=tmp_path), catalog, selection).available()
    assert list(zip(available["dataset"], available["query_id"])) == [("B", "q1")]


def test_available_carries_the_derived_span_totals(tmp_path):
    """Cells band on `derived.*` totals no bank emits, so the parent pool
    must serve a catalog that already carries them — the missing-column
    KeyError that killed a campaign plan the first time every cell was
    hungry."""
    catalog = pd.DataFrame({
        "dataset": ["A"], "query_id": ["q1"], "checkable": [True],
        "structured_identifiers.code_identifier": [2],
        "structured_identifiers.number": [1],
    })
    selection = pd.DataFrame(columns=["dataset", "query_id"])
    available = _pool(AugmentationPaths(data_dir=tmp_path), catalog, selection).available()
    assert available["derived.identifier_spans"].tolist() == [3]
    assert available["derived.corruption_spans"].tolist() == [0]


# --------------------------------------------------------------------------
# reserved()
# --------------------------------------------------------------------------


def test_reserved_is_empty_with_no_pool_on_disk(tmp_path):
    """No pool file yet means nothing has ever been spent as a parent."""
    assert _pool(AugmentationPaths(data_dir=tmp_path)).reserved() == set()


def test_reserved_ignores_nan_generated_from_rows(tmp_path):
    """A pool row with no `generated_from` (a first-generation/natural row)
    must not turn into a bogus `nan`-suffixed key that reserves nothing real."""
    paths = AugmentationPaths(data_dir=tmp_path)
    paths.augmentation_dir.mkdir(parents=True)
    pd.DataFrame({
        "generated_from": ["p1", np.nan],
        "parent_dataset": ["A", "B"],
    }).to_parquet(paths.pool, index=False)
    assert _pool(paths).reserved() == {"A\x00p1"}


def test_reserved_parent_survives_no_later_fill(tmp_path):
    """The round trip the whole rule exists for: a query used once as a
    parent must be excluded from `available()` even though it was never in
    the selection — reused() alone, not just the selection, must catch it."""
    paths = AugmentationPaths(data_dir=tmp_path)
    paths.augmentation_dir.mkdir(parents=True)
    pd.DataFrame({
        "generated_from": ["p1"], "parent_dataset": ["A"],
    }).to_parquet(paths.pool, index=False)
    catalog = pd.DataFrame({
        "dataset": ["A", "A"], "query_id": ["p1", "p2"],
        "checkable": [True, True],
    })
    selection = pd.DataFrame(columns=["dataset", "query_id"])
    available = _pool(paths, catalog, selection).available()
    assert list(available["query_id"]) == ["p2"]


# --------------------------------------------------------------------------
# hydrate() — join_text is stubbed via monkeypatch rather than a ParentPool
# subclass: hydrate() calls the module-level `join_text` name directly, so
# patching `parents.join_text` exercises the real blank/whitespace/NaN drop
# logic without needing a registry cache parquet on disk.
# --------------------------------------------------------------------------


def test_hydrate_drops_blank_whitespace_and_nan_query_text(tmp_path, monkeypatch):
    """Some source caches carry blank query rows; a blank parent would spend
    an LLM call rewriting nothing, so hydrate() must drop all three shapes
    of "no text" and keep only the real one."""
    monkeypatch.setattr(
        parents, "join_text",
        lambda rows, datasets: pd.Series(["ok", "", "   ", np.nan], index=rows.index),
    )
    frame = pd.DataFrame({
        "dataset": ["A"] * 4, "query_id": ["q1", "q2", "q3", "q4"],
    })
    hydrated = _pool(AugmentationPaths(data_dir=tmp_path)).hydrate(frame)
    assert list(hydrated["query_id"]) == ["q1"]
    assert list(hydrated["query"]) == ["ok"]


def test_hydrate_derives_the_identifier_floor_for_a_present_bank(tmp_path, monkeypatch):
    """A row with an email span must carry that bank's floor key so Inject's
    structural check can see it — derived from the real `identifier_floor_key`,
    never a hand-copied string."""
    monkeypatch.setattr(
        parents, "join_text",
        lambda rows, datasets: pd.Series(["has email", "no ids"], index=rows.index),
    )
    frame = pd.DataFrame({
        "dataset": ["A", "A"], "query_id": ["q1", "q2"],
        "structured_identifiers.email": [1.0, 0.0],
    })
    hydrated = _pool(AugmentationPaths(data_dir=tmp_path)).hydrate(frame)
    assert hydrated.loc[hydrated["query_id"] == "q1", "floors"].iloc[0] == (
        [identifier_floor_key("email")]
    )
    assert hydrated.loc[hydrated["query_id"] == "q2", "floors"].iloc[0] == []


# --------------------------------------------------------------------------
# gold_text() — lane_dirs() is monkeypatched the same way, so no real
# `hybrid_search_rrf_dataset.lanes.LANES` import is needed for the doc
# resolution paths below.
# --------------------------------------------------------------------------


def _lane(tmp_path, monkeypatch, key: str = "beir-nfcorpus", name: str = "nfcorpus"):
    monkeypatch.setattr(parents, "lane_dirs", lambda: {key: name})
    return AugmentationPaths(data_dir=tmp_path)


def _write_lane(paths: AugmentationPaths, lane: str, corpus=None, qrels=None) -> None:
    lane_dir = paths.data_dir / lane
    lane_dir.mkdir(parents=True, exist_ok=True)
    if corpus is not None:
        pd.DataFrame(corpus).to_parquet(paths.lane_corpus(lane), index=False)
    if qrels is not None:
        pd.DataFrame(qrels).to_parquet(paths.lane_qrels(lane), index=False)


def test_gold_text_prefers_grounding_doc_id_over_the_qrel_fallback(tmp_path, monkeypatch):
    """Inject's own join must win: pinning this stops a future edit from
    quietly preferring the cheaper qrel lookup over the doc Inject named."""
    paths = _lane(tmp_path, monkeypatch)
    _write_lane(
        paths, "nfcorpus",
        corpus=[
            {"doc_id": "G1", "title": "", "text": "grounded text"},
            {"doc_id": "Q1", "title": "", "text": "qrel top text"},
        ],
        qrels=[{"query_id": "p1", "doc_id": "Q1", "relevance": 2}],
    )
    parent = pd.Series({
        "dataset": "beir-nfcorpus", "query_id": "p1", "grounding_doc_id": "G1",
    })
    assert _pool(paths).gold_text(parent) == "grounded text"


def test_gold_text_falls_back_to_the_top_graded_qrel(tmp_path, monkeypatch):
    """No grounding doc named means Inject never touched this parent, so the
    cut must fall back to its own highest-relevance judged document."""
    paths = _lane(tmp_path, monkeypatch)
    _write_lane(
        paths, "nfcorpus",
        corpus=[
            {"doc_id": "Q0", "title": "", "text": "low grade"},
            {"doc_id": "Q1", "title": "", "text": "top graded text"},
        ],
        qrels=[
            {"query_id": "p1", "doc_id": "Q0", "relevance": 1},
            {"query_id": "p1", "doc_id": "Q1", "relevance": 2},
        ],
    )
    parent = pd.Series({"dataset": "beir-nfcorpus", "query_id": "p1"})
    assert _pool(paths).gold_text(parent) == "top graded text"


def test_gold_text_is_empty_for_an_unknown_lane(tmp_path, monkeypatch):
    """A dataset key `lane_dirs()` does not know must degrade to corpus-blind,
    never raise a KeyError up through the augmentation loop."""
    monkeypatch.setattr(parents, "lane_dirs", lambda: {})
    parent = pd.Series({
        "dataset": "no-such-lane", "query_id": "p1", "grounding_doc_id": "G1",
    })
    assert _pool(AugmentationPaths(data_dir=tmp_path)).gold_text(parent) == ""


def test_gold_text_is_empty_when_the_corpus_file_is_missing(tmp_path, monkeypatch):
    """A lane that resolves but has no materialized corpus.parquet yet must
    not crash the loop — the pushdown read degrades to ''."""
    paths = _lane(tmp_path, monkeypatch)
    parent = pd.Series({
        "dataset": "beir-nfcorpus", "query_id": "p1", "grounding_doc_id": "G1",
    })
    assert _pool(paths).gold_text(parent) == ""


def test_gold_text_is_empty_when_the_doc_id_is_absent_from_the_corpus(tmp_path, monkeypatch):
    """A grounding doc id that the corpus does not (or no longer) carry must
    degrade to '' rather than raise on an empty pushdown-filtered read."""
    paths = _lane(tmp_path, monkeypatch)
    _write_lane(
        paths, "nfcorpus", corpus=[{"doc_id": "Q1", "title": "", "text": "t"}],
    )
    parent = pd.Series({
        "dataset": "beir-nfcorpus", "query_id": "p1", "grounding_doc_id": "GX",
    })
    assert _pool(paths).gold_text(parent) == ""


def test_gold_text_truncates_to_the_requested_length(tmp_path, monkeypatch):
    """`chars` bounds what the operator's instruction embeds, so the cut
    prompt never grows with the source document."""
    paths = _lane(tmp_path, monkeypatch)
    _write_lane(
        paths, "nfcorpus", corpus=[{"doc_id": "G1", "title": "", "text": "grounded text"}],
    )
    parent = pd.Series({
        "dataset": "beir-nfcorpus", "query_id": "p1", "grounding_doc_id": "G1",
    })
    assert _pool(paths).gold_text(parent, chars=4) == "grou"


def test_gold_text_nan_grounding_doc_id_still_uses_the_qrel_fallback(
    tmp_path, monkeypatch,
):
    """NaN is truthy, so a plain `or` took it, stringified it to 'nan', matched
    nothing, and aimed the cut at no evidence."""
    paths = _lane(tmp_path, monkeypatch)
    _write_lane(
        paths, "nfcorpus",
        corpus=[{"doc_id": "Q1", "title": "", "text": "top graded text"}],
        qrels=[{"query_id": "p1", "doc_id": "Q1", "relevance": 2}],
    )
    parent = pd.Series({
        "dataset": "beir-nfcorpus", "query_id": "p1",
        "grounding_doc_id": float("nan"),
    })
    assert _pool(paths).gold_text(parent) == "top graded text"


# --------------------------------------------------------------------------
# CellFill.admit() — a lightweight regex-only extractor is enough: the probe
# cell's bands (length_words, structured_identifiers.number) are both
# regex-engine features, so no spaCy model download is needed.
# --------------------------------------------------------------------------


def _natural_selection(n: int, cell: str = CELL) -> pd.DataFrame:
    return pd.DataFrame({
        "dataset": ["lane-x"] * n,
        "query_id": [f"nat-{i}" for i in range(n)],
        "checkable": True,
        "cell": cell,
        "stage": "reused",
        "route": "dense_only",
        "provenance": "natural",
        "floors": [[cell]] * n,
        "query": [f"word{i}" for i in range(n)],
    })


def _sheet(missing: float, cell: str = CELL) -> pd.DataFrame:
    return pd.DataFrame([{
        "slice": "cell", "floor": cell, "amount": 10.0,
        "credit": 0.0, "missing": missing, "reason": "exhausted",
    }])


def _fill(tmp_path, selection: pd.DataFrame, sheet: pd.DataFrame) -> CellFill:
    selection.to_parquet(tmp_path / "cell_selection.parquet", index=False)
    sheet.to_parquet(tmp_path / "cell_order_sheet.parquet", index=False)
    return CellFill(out_dir=tmp_path)


def _admit(cf: CellFill, pool: pd.DataFrame) -> pd.DataFrame:
    return cf.admit(pool, extractor=FeatureExtractor())


def test_admit_takes_a_row_that_measures_into_its_cell(tmp_path):
    """The one-word "cat" satisfies `bare_concept_token`; a real cell
    predicate, not a declared operator, is the admission test (d51j)."""
    cf = _fill(tmp_path, _natural_selection(20), _sheet(2.0))
    pool = pd.DataFrame({
        "floor": [CELL], "query_id": ["aug-1"], "home_lane": ["lane-a"],
        "query": ["cat"], "provenance": ["augmented"], "credit_gate": ["none"],
    })
    admitted = _admit(cf, pool)
    assert list(admitted["query_id"]) == ["aug-1"]
    assert admitted.iloc[0]["cell"] == CELL


def test_admit_refuses_a_row_that_does_not_measure_into_its_cell(tmp_path):
    """A nine-word pool row misses `length_words<3`, so it must be refused
    even though it targets a hungry cell — admission re-measures, never
    trusts the floor the row was minted for."""
    cf = _fill(tmp_path, _natural_selection(20), _sheet(2.0))
    pool = pd.DataFrame({
        "floor": [CELL], "query_id": ["aug-1"], "home_lane": ["lane-a"],
        "query": ["a very long phrase with many words in it"],
        "provenance": ["augmented"], "credit_gate": ["none"],
    })
    admitted = _admit(cf, pool)
    assert admitted.empty
    kept = pd.read_parquet(cf.selection_path)
    assert list(kept["query_id"]) == [f"nat-{i}" for i in range(20)]


def test_admit_skips_a_gated_row_as_feature_stock(tmp_path):
    """A row behind an open declaration/coherence gate is feature-stock
    (d42h) — it must never earn floor credit even though its text would
    otherwise satisfy the cell."""
    cf = _fill(tmp_path, _natural_selection(20), _sheet(2.0))
    pool = pd.DataFrame({
        "floor": [CELL], "query_id": ["aug-1"], "home_lane": ["lane-a"],
        "query": ["cat"], "provenance": ["doc_grounded"],
        "credit_gate": ["declaration_audit"],
    })
    admitted = _admit(cf, pool)
    assert admitted.empty


def test_admit_does_not_readmit_a_row_already_in_the_selection(tmp_path):
    """A pool row whose query_id the selection already holds must be
    excluded, whatever its text — otherwise a rerun could double-credit
    a floor from the same row."""
    selection = _natural_selection(19)
    already = pd.DataFrame({
        "dataset": ["lane-c"], "query_id": ["dup-1"], "checkable": True,
        "cell": CELL, "stage": "reused", "route": "dense_only",
        "provenance": "natural", "floors": [[CELL]], "query": ["already here"],
    })
    selection = pd.concat([selection, already], ignore_index=True)
    cf = _fill(tmp_path, selection, _sheet(2.0))
    pool = pd.DataFrame({
        "floor": [CELL], "query_id": ["dup-1"], "home_lane": ["lane-a"],
        "query": ["cat"], "provenance": ["augmented"], "credit_gate": ["none"],
    })
    assert _admit(cf, pool).empty


def test_admit_credits_the_sheet_and_drops_a_fully_satisfied_line(tmp_path):
    """Two valid rows against a shortfall of two must zero the cell's
    `missing` and drop its order-sheet line entirely."""
    cf = _fill(tmp_path, _natural_selection(20), _sheet(2.0))
    pool = pd.DataFrame({
        "floor": [CELL, CELL], "query_id": ["aug-1", "aug-2"],
        "home_lane": ["lane-a", "lane-b"], "query": ["cat", "dog"],
        "provenance": ["augmented", "augmented"], "credit_gate": ["none", "none"],
    })
    admitted = _admit(cf, pool)
    assert len(admitted) == 2
    new_sheet = pd.read_parquet(cf.order_sheet_path)
    assert new_sheet.empty


def test_admit_partial_credit_leaves_the_remaining_shortfall(tmp_path):
    """A shortfall of five served by two valid rows must leave the cell on
    the sheet with `missing` reduced by exactly two, not zeroed or dropped."""
    cf = _fill(tmp_path, _natural_selection(20), _sheet(5.0))
    pool = pd.DataFrame({
        "floor": [CELL, CELL], "query_id": ["aug-1", "aug-2"],
        "home_lane": ["lane-a", "lane-b"], "query": ["cat", "dog"],
        "provenance": ["augmented", "augmented"], "credit_gate": ["none", "none"],
    })
    _admit(cf, pool)
    new_sheet = pd.read_parquet(cf.order_sheet_path)
    assert new_sheet.set_index("floor").loc[CELL, "missing"] == 3.0


def test_admit_preserves_the_pool_operators_provenance(tmp_path):
    """d51k: an admitted row's provenance comes from the operator that made
    it, never relabelled to 'natural' just because the fill has no
    `generated_from` column to test instead."""
    cf = _fill(tmp_path, _natural_selection(20), _sheet(2.0))
    pool = pd.DataFrame({
        "floor": [CELL], "query_id": ["aug-1"], "home_lane": ["lane-a"],
        "query": ["cat"], "provenance": ["doc_grounded"], "credit_gate": ["none"],
    })
    admitted = _admit(cf, pool)
    assert admitted.iloc[0]["provenance"] == "doc_grounded"


def test_admit_raises_when_natural_share_would_fall_below_the_floor(tmp_path):
    """Admitting past the recipe's natural-share ceiling must raise and
    leave both artifacts untouched on disk — a partial write would corrupt
    the next fill's invariants."""
    selection = _natural_selection(1)
    sheet = _sheet(2.0)
    cf = _fill(tmp_path, selection, sheet)
    pool = pd.DataFrame({
        "floor": [CELL, CELL], "query_id": ["aug-1", "aug-2"],
        "home_lane": ["lane-a", "lane-b"], "query": ["cat", "dog"],
        "provenance": ["augmented", "augmented"], "credit_gate": ["none", "none"],
    })
    with pytest.raises(AssertionError, match="natural share"):
        _admit(cf, pool)
    assert list(pd.read_parquet(cf.selection_path)["query_id"]) == ["nat-0"]
    assert pd.read_parquet(cf.order_sheet_path).equals(sheet)


def test_admit_returns_empty_when_no_pool_row_targets_a_hungry_cell(tmp_path):
    """A pool row floored for a cell that is not on the order sheet at all
    must be ignored rather than admitted against the wrong shortfall."""
    cf = _fill(tmp_path, _natural_selection(20), _sheet(2.0))
    pool = pd.DataFrame({
        "floor": ["not_a_hungry_cell"], "query_id": ["aug-1"],
        "home_lane": ["lane-a"], "query": ["cat"],
        "provenance": ["augmented"], "credit_gate": ["none"],
    })
    assert _admit(cf, pool).empty
