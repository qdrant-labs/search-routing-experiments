"""GATE 3 for the wave-2 draft cells: positives authored from `looks_like`
alone, negatives blatant non-members. Loads `cells_v3_wave2.yaml` directly —
the draft is deliberately registered nowhere until review."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
import yaml

from composition.cells import CELLS_BY_NAME, ArchetypeCell
from composition.mini_catalog import mini_catalog
from query_taxonomy.features import FeatureExtractor

WAVE2_PATH = (
    Path(__file__).resolve().parents[1]
    / "src"
    / "composition"
    / "cells_v3_wave2.yaml"
)
CELLS_WAVE2: tuple[ArchetypeCell, ...] = tuple(
    ArchetypeCell.model_validate(entry)
    for entry in yaml.safe_load(WAVE2_PATH.read_text())["cells"]
)

PLAIN_CONCEPT = "coffee grinder"
PLAIN_QUESTION = "why do cats purr"
TELEGRAM = "postgres index bloat vacuum"
SATURATED = "i am trying to find out what the cause of the noise in my car is"
BARE_NUMBER = "1984"
PASTED = (
    "The migration ran for about forty minutes on the primary replica before "
    "the coordinator reported a lock timeout, and at that point the queue "
    "backed up behind the write path while the read replicas kept serving "
    "stale rows to the checkout service, which retried each request three "
    "times and eventually surfaced a generic error page to customers in the "
    "European region for roughly a quarter of an hour."
)

CASES: dict[str, tuple[tuple[str, ...], tuple[str, ...]]] = {
    "thin_grammar_short_lookup": (
        ("who invented velcro", "side effects of melatonin in adults"),
        (PLAIN_QUESTION, TELEGRAM),
    ),
    "short_number_lookup": (
        ("chapter 7 bankruptcy filing timeline", "route 66 attractions map"),
        (BARE_NUMBER, PLAIN_CONCEPT),
    ),
    "bare_technical_lookup": (
        ("ERR_CONNECTION_RESET", "useEffect cleanup"),
        (PLAIN_CONCEPT, PLAIN_QUESTION),
    ),
    "content_heavy_midlength": (
        (
            "treatment options for chronic lower back pain during pregnancy",
            "energy efficient window replacement options for older homes in "
            "cold climates",
        ),
        (SATURATED, PLAIN_CONCEPT),
    ),
    "spec_bullet_paste": (
        (
            "stainless steel kitchen sink undermount double bowl 16 gauge "
            "brushed finish 32 x 19 inch drain included limited lifetime "
            "warranty",
            "product name wireless ergonomic keyboard model number K380 "
            "connectivity bluetooth battery type AAA compatible windows macos "
            "android warranty two years color graphite",
        ),
        (PLAIN_QUESTION, PASTED),
    ),
}


@pytest.fixture(scope="module")
def measured() -> pd.DataFrame:
    """Every case text as a catalog row, extracted ONCE with the full engine
    set (the parser stats the cells band on need spaCy)."""
    texts = sorted({t for pos, neg in CASES.values() for t in (*pos, *neg)})
    pool = pd.DataFrame({
        "home_lane": "gate",
        "query_id": [str(i) for i in range(len(texts))],
        "query": texts,
    })
    columns = tuple({band.column for cell in CELLS_WAVE2 for band in cell.bands})
    mini = mini_catalog(pool, FeatureExtractor(engines=None), columns=columns)
    return mini.set_axis(pd.Index(texts, name="query"))


def test_every_cell_has_two_positives_and_two_negatives():
    assert set(CASES) == {cell.name for cell in CELLS_WAVE2}
    assert all(len(side) == 2 for sides in CASES.values() for side in sides)


def test_wave2_names_collide_with_no_registered_cell():
    taken = {cell.name for cell in CELLS_WAVE2} & set(CELLS_BY_NAME)
    assert not taken, taken


@pytest.mark.parametrize("cell", CELLS_WAVE2, ids=lambda c: c.name)
def test_prose_positives_satisfy_their_own_predicate(cell, measured):
    positives = list(CASES[cell.name][0])
    admitted = cell.select(measured.loc[positives])
    rejected = [q for q, ok in zip(positives, admitted) if not ok]
    assert not rejected, (
        f"{cell.name}: its looks_like describes queries its predicate "
        f"rejects: {rejected}"
    )


@pytest.mark.parametrize("cell", CELLS_WAVE2, ids=lambda c: c.name)
def test_blatant_non_members_are_rejected(cell, measured):
    negatives = list(CASES[cell.name][1])
    admitted = cell.select(measured.loc[negatives])
    claimed = [q for q, ok in zip(negatives, admitted) if ok]
    assert not claimed, f"{cell.name}: over-admits {claimed}"
