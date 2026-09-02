"""spec verification (12): identical catalog and configuration fingerprints
produce byte-identical plans. And a check that the config fp changes when
any dial moves."""

from __future__ import annotations

from dataclasses import replace

from rungs.rung_a import RungA, _config_fp
from tests.rungs_fixtures import default_config, make_catalog, make_row


def _catalog():
    return make_catalog([
        make_row("A:1", "A", "first query", ["cell_x"]),
        make_row("A:2", "A", "second query", ["cell_y"]),
        make_row("B:1", "B", "third query", ["cell_x"]),
        make_row("B:2", "B", "fourth query", ["cell_z"]),
    ])


def test_two_runs_produce_the_same_planned_set_and_trace():
    catalog = _catalog()
    config = default_config(catalog, ceiling=4)
    first = RungA().compose(catalog, config, catalog_fp="fp-A")
    second = RungA().compose(catalog, config, catalog_fp="fp-A")
    assert first.planned_set.equals(second.planned_set)
    assert first.selection_trace.equals(second.selection_trace)
    assert first.provenance["config_fp"] == second.provenance["config_fp"]


def test_config_fp_changes_when_theta_changes():
    catalog = _catalog()
    config = default_config(catalog, ceiling=4, theta0=0.7)
    changed = replace(config, theta0=0.5)
    assert _config_fp(config) != _config_fp(changed)


def test_config_fp_changes_when_floors_change():
    catalog = _catalog()
    config = default_config(catalog, ceiling=4)
    floors = dict(config.floors)
    key = next(iter(floors))
    floors[key] = floors[key] + 1
    changed = replace(config, floors=floors)
    assert _config_fp(config) != _config_fp(changed)
