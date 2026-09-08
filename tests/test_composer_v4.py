"""v4 composition invariants — the pre-label coverage contract.

Every test drives ComposerV4 with a plain catalog DataFrame: no pool, no disk,
no labels. The suite enforces that composition depends ONLY on the all-query
candidate catalog and its pre-label features, never on route outcomes.
"""

from __future__ import annotations

import pandas as pd
import pytest

from composition.composer_v4 import DEFAULT_STRATUM_FLOOR, ComposerV4


def _row(qid, lane, *, cells=(), provenance="natural", family=None, operator=None, **bands):
    return {
        "query_id": str(qid),
        "dataset": lane,
        "query": f"q-{lane}-{qid}",
        "cells": frozenset(cells),
        "corruption_degree": bands.get("corruption_degree", "clean"),
        "corpus_idf": bands.get("corpus_idf", "mid_idf"),
        "corpus_oov": bands.get("corpus_oov", "in_vocab"),
        "corpus_pmi": bands.get("corpus_pmi", "co_occurring"),
        "provenance": provenance,
        "operator": operator,
        "family": family,
    }


def _catalog(rows):
    if not rows:
        df = pd.DataFrame([_row(0, "a")]).iloc[0:0]
    else:
        df = pd.DataFrame(rows)
    df = df.reset_index(drop=True)
    df["cells"] = df["cells"].map(frozenset)
    return df


def _compose(catalog, target, *, kappa=1.0, stratum_floor=DEFAULT_STRATUM_FLOOR, rho=0.5):
    return ComposerV4().compose(
        catalog, target, kappa=kappa, stratum_floor=stratum_floor, rho=rho
    )


def _ids(frame):
    return list(zip(frame["dataset"], frame["query_id"]))


def _broad(n=120, lanes=("laneA", "laneB", "laneC")):
    idf = ("low_idf", "mid_idf", "high_idf")
    oov = ("in_vocab", "has_oov")
    pmi = ("co_occurring", "never_co_occurs", "unmeasured")
    corr = ("clean", "light", "heavy")
    cellsets = (("c1",), ("c2",), ("c1", "c3"), ())
    rows = [
        _row(
            i, lanes[i % len(lanes)],
            cells=cellsets[i % len(cellsets)],
            corpus_idf=idf[i % 3], corpus_oov=oov[i % 2],
            corpus_pmi=pmi[i % 3], corruption_degree=corr[i % 3],
        )
        for i in range(n)
    ]
    return _catalog(rows)


def _by_req(report):
    return {(r.axis, r.name): r for r in report.requirements}


# --- A. No-label invariance -------------------------------------------------
def test_label_columns_never_change_selection():
    """Post-label columns present on the catalog are ignored by composition."""
    cat = _broad()
    d1, _ = _compose(cat, 40)
    labelled = cat.assign(
        route_class="dense", route_class_any="sparse", winner="pure_rrf",
        margin=0.7, depth=3, certified=True, is_waste=True,
    )
    d2, _ = _compose(labelled, 40)
    assert _ids(d1) == _ids(d2)


# --- B. New-lane test -------------------------------------------------------
def test_new_lane_with_zero_labels_participates():
    """A freshly materialized lane with no labels is eligible and selected."""
    cat = _broad()
    new = _catalog([
        _row(1000 + i, "newlane", cells=("c1",), corpus_idf="high_idf")
        for i in range(30)
    ])
    combined = _catalog(pd.concat([cat, new], ignore_index=True).to_dict("records"))
    d, report = _compose(combined, 60)
    assert "newlane" in set(d["dataset"])
    assert ("lane", "newlane") in _by_req(report)


# --- C. Label-addition invariance -------------------------------------------
def test_labelling_more_candidates_does_not_move_selection():
    """Marking half the catalog as labelled leaves the chosen ids identical."""
    cat = _broad(120)
    d1, _ = _compose(cat, 40)
    marked = cat.assign(labelled=[i % 2 == 0 for i in range(len(cat))])
    d2, _ = _compose(marked, 40)
    assert _ids(d1) == _ids(d2)


# --- D. Pre-label feature sensitivity ---------------------------------------
def test_new_candidates_for_undercovered_cell_change_selection():
    """Adding genuine supply for a new cell is ALLOWED to move selection."""
    base = _broad(60)
    d1, _ = _compose(base, 30)
    extra = [_row(2000 + i, "laneA", cells=("rare",)) for i in range(20)]
    combined = _catalog(base.to_dict("records") + extra)
    d2, _ = _compose(combined, 30)
    assert _ids(d1) != _ids(d2)
    assert any("rare" in cells for cells in d2["cells"])


# --- E. Augmentation / corruption parity ------------------------------------
def test_augmented_supply_selectable_like_natural():
    """An augmented row that alone satisfies an urgent deficit is selected."""
    rows = [_row(i, "laneA", corpus_pmi="co_occurring") for i in range(30)]
    rows += [
        _row(100 + i, "laneB", corpus_pmi="never_co_occurs",
             provenance="augmented", operator="corrupt", family=f"fam{i % 3}")
        for i in range(10)
    ]
    d, _ = _compose(_catalog(rows), 40)
    aug = d[d["provenance"] == "augmented"]
    assert len(aug) > 0
    assert set(aug["corpus_pmi"]) == {"never_co_occurs"}


# --- F. Genuine shortage ----------------------------------------------------
def test_genuine_shortage_reports_missing_without_substitution():
    """17 candidates for a floor of 25 -> selected 17, missing 8, no filler."""
    rows = [_row(i, "laneA", corpus_pmi="co_occurring") for i in range(40)]
    rows += [_row(100 + i, "laneA", corpus_pmi="never_co_occurs") for i in range(17)]
    _, report = _compose(_catalog(rows), 200, kappa=1.0, stratum_floor=25)
    never = _by_req(report)[("corpus_pmi", "never_co_occurs")]
    assert never.available == 17
    assert never.selected == 17
    assert never.missing == 8


# --- G. Determinism ---------------------------------------------------------
def test_determinism_row_for_row():
    """Same catalog + config -> byte-identical selection and report."""
    cat = _broad(150)
    d1, r1 = _compose(cat, 50)
    d2, r2 = _compose(cat, 50)
    pd.testing.assert_frame_equal(d1, d2)
    assert r1 == r2


# --- termination on adversarial catalogs ------------------------------------
@pytest.mark.parametrize(
    "builder",
    [
        lambda: _catalog([]),
        lambda: _catalog([_row(i, "solo", cells=("c1",)) for i in range(60)]),
        lambda: _catalog([
            _row(i, ("a", "b")[i % 2], corpus_idf="mid_idf") for i in range(30)
        ]),
    ],
)
def test_termination(builder):
    """Adversarial catalogs terminate; size matches the reported size."""
    catalog = builder()
    frame, report = _compose(catalog, 50)
    assert len(frame) == report.size
