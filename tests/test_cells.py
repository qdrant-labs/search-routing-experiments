"""Structural invariants over the archetype cells: their bands name real
catalog columns, their route predictions are real routes, and every cell that
declares an operator reaches it AND resolves to supply that operator can use.
"""

import pandas as pd
import pytest

from augmentation.config import AugmentationConfig
from augmentation.dispatch import Stage, Step, plan, requirements, stage_of
from augmentation.operators import _span_names, default_operators, operator_for
from composition.cell_targets import generation_branches
from composition.cells import CELLS, ArchetypeCell, AxisBand
from composition.compose import DEFAULT_CATALOG
from composition.floors import IDENTIFIER_SPANS, with_derived
from taxonomy_generators.verify import SpanTarget, Targets

IDENTIFIERS = "structured_identifiers."
LENGTH_WORDS = "length.length_words"
NL_SHARE = "natural_language_signal.natural_language_share"
PARSER_SCALARS = (
    "syntactic_depth.nesting_depth",
    "syntactic_depth.statement_count",
    "coordination.widest_list_size",
)
SHORT_CEILING = 10
"""Word ceiling under which one opaque token dominates the whole query."""


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


def test_serves_never_claims_a_cell_name():
    """The two dispatch mechanisms stay apart: `serves` answers for floor
    keys, cells go through `dispatch`, which needs a parent to decide."""
    operators = default_operators()
    claimed = [
        cell.name for cell in CELLS if operator_for(cell.name, operators)
    ]
    assert not claimed, claimed


def _probe(name: str, *bands: AxisBand) -> ArchetypeCell:
    return ArchetypeCell(
        name=name,
        predicate=bands,
        predicts=("sparse_only",),
        rationale="probe",
        source="test",
    )


def test_a_band_demanding_absence_is_minted_by_nobody():
    """`number below 1` means "carry no number". Minting into it pushes the
    row further out, so no operator may claim it — five real cells band a
    feature this way."""
    absent = AxisBand(column=f"{IDENTIFIERS}number", below=1)
    assert not [
        op.declaration.operator for op in default_operators() if op.mints(absent)
    ]


def test_stage_of_classifies_a_requirement_by_cost_to_satisfy():
    """The staging axis is what a requirement costs, not which taxonomy group
    it belongs to: found-only, corpus-copied, woven freely, or refused."""
    operators = default_operators()
    cases = [
        # nothing mints an upper bound or an acronym: both must already hold
        (Stage.SELECT, f"{IDENTIFIERS}number", {"below": 1}),
        (Stage.SELECT, "sentence_markers.acronym", {"at_least": 1}),
        # nor a relevance-changing feature — a parent that already carries
        # "not" was judged carrying it, so finding one is safe
        (Stage.SELECT, "sentence_markers.negation", {"at_least": 1}),
        (Stage.SELECT, "logical_structures.temporal", {"at_least": 1}),
        (Stage.CORPUS, f"{IDENTIFIERS}uuid", {"at_least": 1}),
        (Stage.QUERY_ONLY, "sentence_markers.politeness", {"at_least": 1}),
        (Stage.QUERY_ONLY, "logical_structures.operator_syntax", {"at_least": 1}),
    ]
    for expected, column, bound in cases:
        stage, _ = stage_of((AxisBand(column=column, **bound),), operators)
        assert stage is expected, (column, stage)


def _with_negation_weavable() -> tuple:
    """Operators built as if someone had registered `negation` as a decoration
    — the tempting move REBUILD exists to refuse."""
    base = AugmentationConfig()
    return default_operators(
        base.model_copy(
            update={"decorations": {**base.decorations, "negation": "a negation"}}
        )
    )


def test_minting_a_relevance_changing_feature_is_refused():
    """Registering `negation` as weavable makes it mintable, and the stage
    classifier must then refuse it: adding "not" to a judged query makes its
    gold document the wrong answer."""
    band = AxisBand(column="sentence_markers.negation", at_least=1)
    operators = _with_negation_weavable()
    assert any(op.mints(band) for op in operators)
    assert stage_of((band,), operators)[0] is Stage.REBUILD


def test_a_safe_sibling_keeps_an_any_of_family_servable():
    """A family is ONE requirement, so striking out its relevance-changing
    alternative leaves it servable through the safe sibling."""
    family = (
        AxisBand(column="sentence_markers.negation", at_least=1),
        AxisBand(column="sentence_markers.politeness", at_least=1),
    )
    stage, operator = stage_of(family, _with_negation_weavable())
    assert stage is Stage.QUERY_ONLY
    assert operator is not None


def test_stat_rewrite_allows_exactly_the_spans_the_request_asked_for():
    """A composed cell mints a span AND expands the query in one call, so
    "new span" must mean "one nobody asked for" — otherwise Inject and
    StatRewrite veto each other and no corpus+stat cell can ever pass (d52d).

    The child text is measured, not asserted: which bank claims a surface is
    a bank decision that moves under repair (`0.01` was version_string until
    the pattern tightened, then became number), so the authorisation is built
    from what the text actually exhibits.
    """
    operator = next(
        op for op in default_operators()
        if op.declaration.operator == "stat_rewrite"
    )
    parent = pd.Series({"query": "prenatal vitamins"})
    child = "prenatal vitamins v1.2.3 dosage information"
    gained = _span_names(child) - _span_names(str(parent["query"]))
    assert gained, "the fixture must gain at least one span to be meaningful"

    asked = Targets(spans=tuple(SpanTarget(feature=n) for n in sorted(gained)))
    assert operator.structural(parent, child, asked) == []
    assert operator.structural(parent, child, Targets()) != []
    marker = Targets(spans=(SpanTarget(feature="greeting"),))
    assert operator.structural(
        parent, "hi there, prenatal vitamins dosage information", marker
    ) == []


def test_an_upper_bound_constrains_last_and_filters_nobody():
    """A `length_words < 6` band is a cut the model performs, not an entry
    requirement, so a 40-word parent still qualifies (d53). Filtering on it
    first is search logic and starves the cell."""
    cell = _probe(
        "cut_last",
        AxisBand(column=f"{IDENTIFIERS}uuid", at_least=1),
        AxisBand(column="length.length_words", below=6),
    )
    stages = {
        step.stage
        for step in (Step(r, *stage_of(r, default_operators()))
                     for r in requirements(cell))
    }
    assert stages == {Stage.CORPUS, Stage.CONSTRAIN}

    long_parent = pd.DataFrame({
        f"{IDENTIFIERS}uuid": [0.0],
        "length.length_words": [40.0],
        "dataset": ["d"], "query_id": ["a"], "checkable": [True],
    })
    result = plan(cell, long_parent, default_operators())
    # the corpus stage drops it (no surface supply in a bare frame), but the
    # length band must never be the reason
    assert Stage.CONSTRAIN not in {s.stage for s in result.unsatisfied} or (
        Stage.CORPUS in {s.stage for s in result.unsatisfied}
    )


def test_a_cut_reads_the_gold_document_when_one_is_known():
    """The cut has to keep the judged document answering, so that document is
    the brief — and its absence degrades to corpus-blind rather than crashing.
    Uses a real cell: StatRewrite resolves a cell's band through CELLS_BY_NAME."""
    operator = next(
        op for op in default_operators()
        if op.declaration.operator == "stat_rewrite"
    )
    cell = "registry_structured_identifier"
    blind = operator.instruction(cell, pd.Series({"query": "q"}))
    grounded = operator.instruction(
        cell, pd.Series({"query": "q", "gold_text": "ASPIRIN LOWERS RISK"})
    )
    assert "Shorten" in blind and "Shorten" in grounded
    assert "ASPIRIN LOWERS RISK" in grounded
    assert "must still be found" not in blind


def _length_ceiling(cell: ArchetypeCell) -> float | None:
    return next(
        (b.below for b in cell.bands if b.column == LENGTH_WORDS and b.below),
        None,
    )


def _demands_content(cell: ArchetypeCell) -> bool:
    """Whether the cell asks for anything positive beyond being short — a span
    it is built around, or the character mass that marks a blob. A cell that
    asks for neither is claiming "short and ordinary" and must say the second
    half out loud."""
    return bool(cell.required_banks) or any(
        b.column == "length.length_chars" and b.at_least for b in cell.bands
    )


def _forbids_identifiers(cell: ArchetypeCell) -> bool:
    return any(
        b.column == IDENTIFIER_SPANS and b.below is not None and b.below <= 1
        for b in cell.bands
    )


def test_a_short_concept_cell_admits_no_identifier():
    """A short cell demanding nothing of its content must forbid identifiers as
    a CLASS, not one bank at a time — guarding a single column passes this shape
    of test while every other opaque token still falls in (SPEC d62f)."""
    sinks = [
        cell.name
        for cell in CELLS
        if (ceiling := _length_ceiling(cell)) is not None
        and ceiling <= SHORT_CEILING
        and not _demands_content(cell)
        and not _forbids_identifiers(cell)
    ]
    assert not sinks, sinks


def test_parser_scalars_require_a_natural_language_floor():
    """The parser invents structure on non-sentences — a bare UUID reads
    nesting_depth 3.0, above a real question — so the scalar only means
    anything above a natural-language floor."""
    unguarded = [
        cell.name
        for cell in CELLS
        if any(b.column in PARSER_SCALARS for b in cell.bands)
        and not any(
            b.column == NL_SHARE and b.at_least is not None for b in cell.bands
        )
    ]
    assert not unguarded, unguarded


def test_derived_identifier_spans_totals_the_banks():
    frame = pd.DataFrame({
        f"{IDENTIFIERS}uuid": [1.0, 0.0],
        f"{IDENTIFIERS}cve": [2.0, 0.0],
        "sentence_markers.acronym": [5.0, 5.0],
        LENGTH_WORDS: [1.0, 2.0],
    })
    assert list(with_derived(frame)[IDENTIFIER_SPANS]) == [3.0, 0.0]


def test_derived_identifier_spans_stays_out_of_the_span_prefixes():
    """`SpanCountAxis` and `span_mask` sum everything under a group prefix; a
    total living there would be counted twice."""
    assert not IDENTIFIER_SPANS.startswith(IDENTIFIERS)


@pytest.mark.skipif(not DEFAULT_CATALOG.exists(), reason="feature table not built")
def test_band_columns_exist_in_the_catalog():
    import pandas as pd

    columns = set(with_derived(pd.read_parquet(DEFAULT_CATALOG)).columns)
    absent = sorted(
        band.column
        for cell in CELLS
        for band in cell.bands
        if band.column not in columns
    )
    assert not absent, absent
