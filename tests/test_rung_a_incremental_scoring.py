"""The fast selector's private state must equal the declared scoring formula."""

from __future__ import annotations

from rungs.rung_a import (
    _SelectionState,
    _novelty,
    _prepare,
    _structural_gain,
)
from tests.rungs_fixtures import default_config, make_catalog, make_row


def test_incremental_state_matches_reference_gain_and_lane_novelty():
    """Every update is equivalent to rescanning the selected pandas rows."""
    catalog = make_catalog([
        make_row("A:1", "A", "alpha beta gamma", ["cell_x"]),
        make_row("A:2", "A", "alpha beta delta", ["cell_x", "cell_y"]),
        make_row("A:3", "A", "unrelated words", ["cell_y"]),
        make_row("B:1", "B", "alpha beta gamma", ["cell_z"]),
        make_row("B:2", "B", "alpha epsilon", ["cell_z"]),
    ])
    config = default_config(catalog, ceiling=5, floor_by_cell=2, floor_by_lane=2)
    candidates = _prepare(catalog)
    state = _SelectionState.from_candidates(candidates)
    chosen: list[int] = []

    for picked in (0, 3, 1):
        chosen_set = set(chosen)
        selected = candidates.iloc[chosen] if chosen else None
        for i, row in candidates.iterrows():
            assert state.gain(i, config) == _structural_gain(row, selected, config)
            assert state.novelty(i) == _novelty(row, candidates, chosen_set)
        state.record_pick(picked)
        chosen.append(picked)
