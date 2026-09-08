"""The lane-share cap arithmetic, the mass cap that throttles a cell to its
natural prevalence, and the per-cell readout when the artifact has been built.
"""

import pandas as pd
import pytest

from composition.cellfill import CellFill, LaneCap, MassCap
from composition.cells import CELLS, ArchetypeCell
from composition.recipe import Recipe


def _counts(*values: int) -> pd.Series:
    return pd.Series(values, index=[f"lane{i}" for i, _ in enumerate(values)])


def test_the_target_holds_when_the_lanes_can_satisfy_it():
    cap = LaneCap(_counts(100, 100, 100, 100, 100, 100), draw=50, target=0.2)
    assert cap.share == 0.2
    assert not cap.infeasible
    assert cap.capacity == 50


def test_few_lanes_record_the_achievable_floor():
    cap = LaneCap(_counts(100, 100, 100), draw=50, target=0.2)
    assert cap.floor == pytest.approx(0.34)
    assert cap.share == pytest.approx(0.34)
    assert cap.infeasible
    assert cap.capacity == 50


def test_a_single_lane_cell_fills_at_share_one():
    cap = LaneCap(_counts(80), draw=50, target=0.2)
    assert cap.share == 1.0
    assert cap.infeasible
    assert cap.capacity == 50


def test_thin_supply_caps_capacity_below_the_draw():
    cap = LaneCap(_counts(3, 2), draw=50, target=0.2)
    assert cap.floor == 1.0
    assert cap.capacity == 5


def test_an_empty_cell_has_no_capacity():
    cap = LaneCap(_counts(), draw=50, target=0.2)
    assert cap.capacity == 0


def test_without_a_cap_every_cell_keeps_the_flat_quota():
    mass = MassCap(natural=10, pool=440_000, recipe=Recipe())
    assert mass.cap == 200
    assert mass.quota == 200
    assert not mass.generation_only


def test_v3_recipe_turns_on_the_cap_and_nothing_else():
    v2, v3 = Recipe(), Recipe.v3()
    assert v2.k_cap is None and v3.k_cap == 5.0
    assert v3.model_dump(exclude={"k_cap"}) == v2.model_dump(exclude={"k_cap"})
    assert Recipe.v3(seed=7).seed == 7


def test_a_fat_cell_is_still_bounded_by_the_flat_quota():
    mass = MassCap(5_000, 10_000, Recipe(k_cap=5.0, certified_total=4_000))
    assert mass.cap == 10_000
    assert mass.quota == 200


def test_a_thin_cell_is_throttled_to_five_times_its_mass():
    # 0.1% of the pool, 5x it over a 4,000-row total = 20 rows per route
    mass = MassCap(10, 10_000, Recipe(k_cap=5.0, certified_total=4_000))
    assert mass.cap == 20
    assert mass.quota == 20
    assert not mass.generation_only


def test_a_cell_the_cap_keeps_under_the_floor_is_generation_only():
    mass = MassCap(1, 10_000, Recipe(k_cap=5.0, certified_total=4_000))
    assert mass.quota == 2
    assert mass.generation_only


def test_an_empty_catalog_has_no_prevalence():
    assert MassCap(0, 0, Recipe(k_cap=5.0)).p_natural == 0.0


# --- the whole fill over a synthetic catalog -------------------------------
# five lanes x 2,000 rows; three disjoint cells banded on one column, so
# every prevalence and the uncovered share are hand-computable.
LANES = [f"lane{i}" for i in range(5)]
SPANS = {"fat": (10, 1_000), "thin": (30, 60), "tiny": (50, 2)}


def _cell(name: str) -> ArchetypeCell:
    low = SPANS[name][0]
    return ArchetypeCell(
        name=name,
        predicate=(
            {"column": "length.length_words", "at_least": low, "below": low + 10},
        ),
        predicts=("dense_only",),
        rationale="synthetic",
        source="test",
    )


def _catalog() -> pd.DataFrame:
    rows = []
    for lane in LANES:
        words = [0] * 2_000
        at = 0
        for name, (low, per_lane) in SPANS.items():
            del name
            words[at:at + per_lane] = [low + 5] * per_lane
            at += per_lane
        rows.append(pd.DataFrame({
            "dataset": lane,
            "query_id": [f"{lane}-{i}" for i in range(2_000)],
            "checkable": True,
            "length.length_words": words,
        }))
    return pd.concat(rows, ignore_index=True)


def _labels(catalog: pd.DataFrame) -> pd.DataFrame:
    """Every `thin` row a dense winner, plus 250 dense / 250 sparse in `fat`."""
    thin = catalog[catalog["length.length_words"] == 35]
    fat = catalog[catalog["length.length_words"] == 15]
    dense = pd.concat([thin, fat.groupby("dataset").head(50)])
    sparse = fat.groupby("dataset").tail(50)
    return pd.concat([
        dense.assign(
            score_dense_only=1.0, score_sparse_only=0.0, score_pure_rrf=0.0
        ),
        sparse.assign(
            score_dense_only=0.0, score_sparse_only=1.0, score_pure_rrf=0.0
        ),
    ], ignore_index=True)[
        ["dataset", "query_id", "checkable",
         "score_dense_only", "score_sparse_only", "score_pure_rrf"]
    ]


@pytest.fixture
def synthetic(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "composition.cellfill.join_text",
        lambda frame, datasets: pd.Series("q", index=frame.index),
    )
    catalog = _catalog()
    catalog.to_parquet(tmp_path / "catalog.parquet", index=False)
    _labels(catalog).to_parquet(tmp_path / "labels.parquet", index=False)

    def fill(recipe: Recipe, tag: str = "capped") -> CellFill:
        return CellFill(
            recipe=recipe,
            catalog_path=tmp_path / "catalog.parquet",
            labels_path=tmp_path / "labels.parquet",
            out_dir=tmp_path / tag,
            cells=tuple(_cell(name) for name in SPANS),
        )

    return fill


def test_the_cap_throttles_thin_cells_and_leaves_fat_ones_alone(synthetic):
    fill = synthetic(Recipe(k_cap=5.0, certified_total=1_000))
    fill.build(force=True)
    report = pd.read_parquet(fill.report_path).set_index("cell")
    # fat 5,000/10,000 -> cap 2,500; thin 300 -> 150; tiny 10 -> 5
    assert list(report["p_natural"]) == [0.5, 0.03, 0.001]
    assert list(report["mass_cap"]) == [2_500, 150, 5]
    assert list(report["quota"]) == [200, 150, 5]
    # the throttle reaches the draw, not just the readout
    assert report.at["thin", "reused_dense"] == 150
    assert report.at["fat", "reused_dense"] == 200


def test_the_cap_flags_a_cell_it_denies_its_floor(synthetic):
    fill = synthetic(Recipe(k_cap=5.0, certified_total=1_000))
    fill.build(force=True)
    report = pd.read_parquet(fill.report_path).set_index("cell")
    assert list(report["generation_only"]) == [False, False, True]
    summary = fill.summary_path.read_text()
    assert "generation-only 1 cells" in summary
    assert "tiny" in summary.partition("generation-only")[2].splitlines()[0]


def test_the_summary_reports_the_share_no_cell_claims(synthetic):
    fill = synthetic(Recipe(k_cap=5.0, certified_total=1_000))
    fill.build(force=True)
    # 10,000 rows, 5,310 claimed by the three cells
    assert "uncovered 46.9% of the catalog" in fill.summary_path.read_text()


def test_the_flat_quota_is_what_the_default_recipe_still_draws(synthetic):
    capped = synthetic(Recipe(k_cap=5.0, certified_total=1_000))
    flat = synthetic(Recipe(), tag="flat")
    capped.build(force=True)
    flat.build(force=True)
    report = pd.read_parquet(flat.report_path).set_index("cell")
    assert list(report["quota"]) == [200, 200, 200]
    assert not report["generation_only"].any()
    assert report.at["thin", "reused_dense"] == 200
    assert len(pd.read_parquet(flat.selection_path)) > len(
        pd.read_parquet(capped.selection_path)
    )


@pytest.mark.skipif(
    not CellFill().report_path.exists(), reason="cell fill not built"
)
def test_the_readout_covers_every_cell_and_respects_its_cap():
    report = pd.read_parquet(CellFill().report_path)
    assert list(report["cell"]) == [cell.name for cell in CELLS]
    assert (report["lane_share_cap"] >= report["lane_share_floor"]).all()
    assert (report["natural_capacity"] <= report["natural_rows"]).all()
