"""Structural invariants over the archetype cells: their bands name real
catalog columns, their route predictions are real routes, and every cell that
declares an operator reaches it AND resolves to supply that operator can use.
"""

import pytest

from augmentation.config import AugmentationConfig
from augmentation.operators import default_operators, operator_for
from composition.cell_targets import generation_branches
from composition.cells import CELLS, GENERATION_CELLS, ArchetypeCell, AxisBand
from composition.compose import DEFAULT_CATALOG

IDENTIFIERS = "structured_identifiers."


def test_cells_have_unique_names():
    assert len({cell.name for cell in CELLS}) == len(CELLS)


def test_predicts_are_real_routes():
    from hybrid_search_rrf_dataset.fusion import StrategyName

    routes = {route.value for route in StrategyName}
    assert {r for cell in CELLS for r in cell.predicts} <= routes


def test_every_band_becomes_one_target():
    for cell in CELLS:
        for branch in generation_branches(cell):
            spans_and_stats = len(branch.targets.spans) + len(branch.targets.stats)
            expected = len(cell.predicate) + (branch.alternative is not None)
            assert spans_and_stats == expected, cell.name


def test_any_of_yields_one_branch_per_alternative():
    for cell in CELLS:
        assert len(generation_branches(cell)) == max(len(cell.any_of), 1), cell.name


def test_required_banks_count_only_bands_demanding_presence():
    cell = ArchetypeCell(
        name="probe",
        predicate=(
            AxisBand(column=f"{IDENTIFIERS}uuid", at_least=1),
            AxisBand(column=f"{IDENTIFIERS}number", below=1),
            AxisBand(column="length.length_words", at_least=3),
        ),
        predicts=("sparse_only",),
        rationale="probe",
        source="test",
    )
    assert cell.required_banks == {"uuid"}


def test_declared_operator_is_the_one_that_serves_the_cell():
    operators = default_operators()
    for cell in CELLS:
        if cell.operator is None:
            continue
        served = operator_for(cell.name, operators)
        assert served is not None, cell.name
        assert served.declaration.operator == cell.operator, cell.name


def test_generation_cells_resolve_to_supply_their_operator_can_use():
    """A cell name is not a supply key: inject reads a per-bank surface
    index, decorate a phrase list. A cell reaching an operator that cannot
    resolve it produces nothing, silently."""
    decorations = set(AugmentationConfig().decorations)
    for cell in CELLS:
        if cell.operator == "inject":
            identifiers = {
                band.member
                for band in cell.bands
                if band.column.startswith(IDENTIFIERS)
            }
            assert cell.required_banks & identifiers, cell.name
        elif cell.operator == "decorate":
            assert cell.required_banks & decorations, cell.name


def test_every_operator_named_by_a_cell_claims_it():
    served = {op.declaration.operator for op in default_operators()}
    assert set(GENERATION_CELLS) <= served, set(GENERATION_CELLS) - served


@pytest.mark.skipif(not DEFAULT_CATALOG.exists(), reason="feature table not built")
def test_band_columns_exist_in_the_catalog():
    import pandas as pd

    columns = set(pd.read_parquet(DEFAULT_CATALOG).columns)
    absent = sorted(
        band.column
        for cell in CELLS
        for band in cell.bands
        if band.column not in columns
    )
    assert not absent, absent
