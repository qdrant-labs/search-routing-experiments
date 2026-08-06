"""Stress tests for the four augmentation operator families (d42d/d52d/d53f/
d54) and the `Operator` contract (d42c). No LLM call anywhere: families are
exercised directly on hand-built frames, never through `Augmenter.run`.
"""

from __future__ import annotations

import pandas as pd
import pytest

from augmentation.config import AugmentationConfig
from augmentation.core import AnswerKeyPath, Operator, SurfaceOrigin
from augmentation.operators import (
    DecorateOperator,
    InjectOperator,
    OperatorSyntaxRewrite,
    StatRewrite,
    default_operators,
)
from composition.cells import ArchetypeCell, AxisBand, CELLS_BY_NAME
from taxonomy_generators.verify import SpanTarget, Targets

IDENT = "structured_identifiers."


# --- Operator contract ------------------------------------------------------


def test_operator_is_abstract_without_every_hook():
    """d42c default-deny is enforced by the ABC itself, not convention: a
    family missing `structural` cannot even be constructed."""

    class Incomplete(Operator):
        def mints(self, band):
            return False

        def serves(self, floor):
            return False

        def eligible(self, selection, floor):
            return selection

        def instruction(self, floor, parent):
            return ""

        def targets(self, floor, parent):
            return Targets()

    with pytest.raises(TypeError):
        Incomplete(AugmentationConfig())


def test_minted_answer_key_implies_a_grounded_origin():
    """A MINTED key means a document was consulted; an INHERIT key means
    none was — mixing them would mint a key nobody can point at."""
    for op in default_operators():
        d = op.declaration
        if d.answer_key is AnswerKeyPath.MINTED:
            assert d.surface_origin in (SurfaceOrigin.DOC_COPIED, SurfaceOrigin.SYNTHETIC)
        else:
            assert d.surface_origin is SurfaceOrigin.NONE


def test_surface_origin_provenance_covers_the_selection_values():
    assert SurfaceOrigin.NONE.provenance == "augmented"
    assert SurfaceOrigin.DOC_COPIED.provenance == "doc_grounded"
    assert SurfaceOrigin.SYNTHETIC.provenance == "synthetic"


def test_unsatisfied_passes_a_catalog_pool_through_untouched():
    """A cell pool carries no `floors` column — dispatch already excluded
    satisfying rows — so the filter must be a no-op, not an empty result."""
    pool = pd.DataFrame({"query_id": ["a", "b"]})
    assert DecorateOperator().unsatisfied(pool, "marker:greeting") is pool


def test_unsatisfied_filters_a_selection_pool_by_floor_membership():
    pool = pd.DataFrame({
        "query_id": ["a", "b"], "floors": [["marker:greeting"], []],
    })
    kept = DecorateOperator().unsatisfied(pool, "marker:greeting")
    assert list(kept["query_id"]) == ["b"]


@pytest.mark.parametrize("operator_cls,column,minted", [
    (DecorateOperator, "sentence_markers.politeness", True),
    (DecorateOperator, "sentence_markers.acronym", False),
    (OperatorSyntaxRewrite, "logical_structures.operator_syntax", True),
    (OperatorSyntaxRewrite, "sentence_markers.acronym", False),
    (InjectOperator, f"{IDENT}uuid", True),
    (InjectOperator, "sentence_markers.greeting", False),
])
def test_family_mints_only_the_band_it_owns(operator_cls, column, minted):
    """A presence band (`at_least >= 1`): mints() must say yes for the
    feature the family actually weaves and no for everything else."""
    assert operator_cls().mints(AxisBand(column=column, at_least=1)) is minted


@pytest.mark.parametrize("operator_cls,column", [
    (DecorateOperator, "sentence_markers.politeness"),
    (OperatorSyntaxRewrite, "logical_structures.operator_syntax"),
    (InjectOperator, f"{IDENT}uuid"),
])
def test_family_refuses_a_band_demanding_absence(operator_cls, column):
    """`below=1` means "carry none" — minting pushes the row further out,
    so `demands_presence` must gate every additive family, even on its
    own feature."""
    assert operator_cls().mints(AxisBand(column=column, below=1)) is False


# --- DecorateOperator --------------------------------------------------------


def test_decorate_resolves_a_cell_to_one_required_marker():
    """`conversational_courtesy_wrapper` any-of's greeting/politeness/
    interjection; any single one must satisfy it, chosen deterministically."""
    op = DecorateOperator()
    assert op.marker("conversational_courtesy_wrapper") == "greeting"
    assert op.serves("conversational_courtesy_wrapper") is False, (
        "cells dispatch through the loop, never through serves() (d51c)"
    )
    assert op.serves("marker:greeting") is True


def test_decorate_structural_is_a_declared_permanent_no_op():
    """d42c: a family with no parent-relative check DECLARES that with an
    explicit `return []` — never silence. Pin it for arbitrary input."""
    op = DecorateOperator()
    parent = pd.Series({"query": "x", "floors": []})
    assert op.structural(parent, "anything at all, even garbage <>{}", Targets()) == []


# --- OperatorSyntaxRewrite ---------------------------------------------------


def test_operator_syntax_instruction_forbids_not():
    instruction = OperatorSyntaxRewrite().instruction("logical:operator_syntax", pd.Series())
    assert "NOT" in instruction and "Never use NOT" in instruction


def test_operator_syntax_structural_allows_and_or_but_rejects_new_words():
    op = OperatorSyntaxRewrite()
    parent = pd.Series({"query": "cats dogs"})
    assert op.structural(parent, "cats AND dogs", Targets()) == []
    problems = op.structural(parent, "cats AND NOT dogs", Targets())
    assert any("new content tokens" in p for p in problems)
    assert any("emitted NOT" in p for p in problems)


def test_operator_syntax_structural_rejects_not_even_when_not_a_new_word():
    """d52h's guard is a dedicated check, not a side effect of the
    new-token scan: uppercasing a NOT the parent already had (lowercase)
    introduces no new token, yet must still be refused."""
    op = OperatorSyntaxRewrite()
    parent = pd.Series({"query": "why not aspirin"})
    problems = op.structural(parent, "why NOT aspirin", Targets())
    assert problems == ["emitted NOT, which changes the relevant doc set"]


# --- StatRewrite --------------------------------------------------------------


def test_stat_rewrite_mints_both_declared_directions():
    """d53f: `(length_words, DOWN)` is now a declared move alongside UP."""
    op = StatRewrite()
    assert op.mints(AxisBand(column="length.length_words", at_least=60))
    assert op.mints(AxisBand(column="length.length_words", below=6))


def test_stat_rewrite_refuses_an_undeclared_axis():
    op = StatRewrite()
    assert op.mints(AxisBand(column="depth.nesting_depth", at_least=4)) is False
    assert op.mints(AxisBand(column="depth.nesting_depth", below=2)) is False


def test_stat_rewrite_serves_only_floor_keys_never_cell_names():
    """serves() answers FLOOR keys only (d51c) — a cell reaches this family
    through dispatch, which needs a parent to derive the failing band."""
    op = StatRewrite()
    assert op.serves("registry_structured_identifier") is False
    assert op.serves("length_words:60+") is True


def test_stat_rewrite_serves_refuses_a_band_on_an_undeclared_axis():
    """The band resolves fine (nesting_depth is a real axis) but the
    direction is undeclared, so serves() must still say no."""
    assert StatRewrite().serves("nesting_depth:0-1") is False


def _pinned_op(tmp_path, catalog):
    from augmentation.config import AugmentationPaths

    paths = AugmentationPaths(data_dir=tmp_path)
    paths.catalog.parent.mkdir(parents=True, exist_ok=True)
    catalog.to_parquet(paths.catalog, index=False)
    return StatRewrite(AugmentationConfig(paths=paths))


def test_stat_rewrite_eligible_includes_a_row_that_already_holds(tmp_path):
    """2026-08 follow-up: `movable` alone only catches rows OUTSIDE a
    two-sided band. A row already sitting inside "3-6" (bare, no edit
    needed) was being dropped entirely — the exact bug this fixes."""
    catalog = pd.DataFrame({
        "dataset": ["d", "d", "d"], "query_id": ["below", "in_band", "above"],
        "checkable": [True, True, True],
        "length.length_words": [1.0, 5.0, 8.0],
    })
    op = _pinned_op(tmp_path, catalog)
    selection = catalog[["dataset", "query_id", "checkable"]]
    out = op.eligible(selection, "length_words:3-6")
    assert set(out["query_id"]) == {"below", "in_band", "above"}, (
        "'in_band' (bare 5, already inside [3,7)) must not be dropped for "
        "needing no move"
    )


def test_stat_rewrite_eligible_sorts_a_move_surfaces_already_covered_first(
    tmp_path,
):
    """No edit beats any edit: a row a prior CORPUS mint's committed words
    already push into the open "60+" band must sort ahead of one that still
    needs genuine padding, even though both are `checkable`."""
    catalog = pd.DataFrame({
        "dataset": ["d", "d"], "query_id": ["far", "almost"],
        "checkable": [True, True],
        "length.length_words": [10.0, 55.0],
    })
    op = _pinned_op(tmp_path, catalog)
    selection = pd.DataFrame({
        "dataset": ["d", "d"], "query_id": ["far", "almost"],
        "checkable": [True, True],
        "surfaces": [(), ("one two three four five six",)],
    })
    out = op.eligible(selection, "length_words:60+")
    assert list(out["query_id"])[0] == "almost", (
        "55 bare words + 6 mandatory surface words = 61, already in band; "
        "'far' (10 words) still needs a genuine, costly move"
    )


def test_stat_rewrite_structural_new_span_rule():
    """d52d: 'new' means 'nobody asked for it'. A span the parent already
    carried is free, an authorised span is free, an unrequested one fails."""
    op = StatRewrite()
    parent = pd.Series({"query": "v1.0 dosage information"})
    kept_old = "v1.0 dosage information for patients"
    assert op.structural(parent, kept_old, Targets()) == []

    gained_unrequested = "v1.0 dosage information hi there"
    problems = op.structural(parent, gained_unrequested, Targets())
    assert problems == ["child gained spans: ['greeting']"]

    asked = Targets(spans=(SpanTarget(feature="greeting"),))
    assert op.structural(parent, gained_unrequested, asked) == []


def test_stat_rewrite_up_instruction_states_the_band_and_reads_current_value():
    op = StatRewrite()
    open_ended = op.instruction("length_words:60+", pd.Series({"query": "q"}))
    assert "at least 60" in open_ended and "currently measures" not in open_ended

    with_value = op.instruction(
        "length_words:3-6", pd.Series({"query": "q", "stat_value": 4.0})
    )
    assert "between 3 and 7 (exclusive)" in with_value
    assert "currently measures 4" in with_value


def test_stat_rewrite_down_instruction_embeds_the_gold_document():
    """d53d: the cut is corpus-aware, so the document is quoted verbatim
    into the brief when one is known — uses a real registered cell,
    because `_cell_band` resolves through the global CELLS_BY_NAME."""
    op = StatRewrite()
    cell = "registry_structured_identifier"
    grounded = op.instruction(cell, pd.Series({"query": "q", "gold_text": "ASPIRIN LOWERS RISK"}))
    assert "ASPIRIN LOWERS RISK" in grounded
    assert "must still be found" in grounded


def test_stat_rewrite_down_instruction_degrades_without_a_document():
    """d53d's escape hatch: no resolvable gold text falls back to a
    corpus-blind instruction rather than failing."""
    op = StatRewrite()
    cell = "registry_structured_identifier"
    blind = op.instruction(cell, pd.Series({"query": "q"}))
    assert "must still be found" not in blind
    assert "Shorten" in blind


def test_stat_rewrite_resolves_a_cell_the_registry_never_saw():
    """A generated cell is absent from CELLS_BY_NAME, so resolving its band
    through that global was a dead end. The caller now hands the requirement
    over, and a caller that hands over nothing gets a clear refusal instead of
    a TypeError from unpacking None."""
    op = StatRewrite()
    local_cell = ArchetypeCell(
        name="not_in_the_registry",
        predicate=(AxisBand(column="length.length_words", at_least=10),),
        predicts=("sparse_only",),
        rationale="probe",
        source="test",
    )
    assert local_cell.name not in CELLS_BY_NAME

    given = op.instruction(
        local_cell.name, pd.Series({"query": "q"}), local_cell.predicate
    )
    assert "length_words" in given

    with pytest.raises(ValueError, match="no band"):
        op.instruction(local_cell.name, pd.Series({"query": "q"}))


# --- InjectOperator -----------------------------------------------------------


def test_inject_wanted_reads_a_cells_count_band():
    """d54: a `code_identifier >= 2` cell asks for two surfaces from one
    doc/bank; a bare id: floor (no count) defaults to one."""
    assert InjectOperator.wanted("symbol_pile_no_grammar") == 2
    assert InjectOperator.wanted("id:tech") == 1


def _surfaces(*rows: tuple[str, str, str]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=["doc_id", "bank", "surface"]).assign(floor="id:tech")


def test_offers_a_repeated_surface_can_never_satisfy_a_count():
    """d54c: two copies of one token are ONE span to the banks, so three
    rows of the same surface still offer only one distinct surface."""
    surfaces = _surfaces(
        ("d1", "code_identifier", "cPGES"),
        ("d1", "code_identifier", "cPGES"),
        ("d1", "code_identifier", "cPGES"),
    )
    assert InjectOperator._offers(surfaces, 2).empty
    single = InjectOperator._offers(surfaces, 1)
    assert list(single["surfaces"]) == [("cPGES",)], "wanted=1 must still be a tuple"


def test_offers_groups_doc_and_bank_independently():
    surfaces = _surfaces(
        ("d1", "code_identifier", "cPGES"),
        ("d1", "code_identifier", "lipoxinA4"),
        ("d1", "uuid", "550e8400-e29b-41d4-a716-446655440000"),
    )
    offers = InjectOperator._offers(surfaces, 2)
    assert list(zip(offers["doc_id"], offers["bank"])) == [("d1", "code_identifier")]
    assert offers.iloc[0]["surfaces"] == ("cPGES", "lipoxinA4")


def test_offers_truncates_extra_surfaces_to_exactly_wanted():
    surfaces = _surfaces(
        ("d1", "code_identifier", "a"),
        ("d1", "code_identifier", "b"),
        ("d1", "code_identifier", "c"),
    )
    offers = InjectOperator._offers(surfaces, 2)
    assert offers.iloc[0]["surfaces"] == ("a", "b")


def test_offers_drops_a_doc_short_of_the_demand():
    surfaces = _surfaces(("d1", "code_identifier", "onlyOne"))
    assert InjectOperator._offers(surfaces, 2).empty


_PARENT = pd.Series({
    "surfaces": ("cPGES",), "bank": "code_identifier", "floors": [],
})


def test_inject_structural_catches_a_missing_or_case_altered_surface():
    op = InjectOperator()
    missing = op.structural(_PARENT, "totally different text", Targets())
    assert any("cPGES" in p for p in missing)
    cased = op.structural(_PARENT, "cpges alternative medicine", Targets())
    assert any("cPGES" in p for p in cased), "the containment check is case-sensitive"


def test_inject_structural_catches_a_cross_domain_identifier_gain():
    """A copied code_identifier surface authorises id:tech, not id:medical —
    an unrequested medical code riding along must still be caught."""
    op = InjectOperator()
    problems = op.structural(
        _PARENT, "cPGES J45.909 alternative medicine",
        Targets(spans=(SpanTarget(feature="code_identifier"),)),
    )
    assert problems == ["gained other identifier floors: ['id:medical']"]


def test_inject_structural_ignores_a_marker_target_without_crashing():
    """A marker target names no identifier bank, so it must never reach
    `identifier_floor_key` (which only knows identifier banks)."""
    op = InjectOperator()
    targets = Targets(spans=(
        SpanTarget(feature="greeting"),
        SpanTarget(feature="code_identifier"),
    ))
    assert op.structural(_PARENT, "hi there, cPGES alternative medicine", targets) == []


def test_inject_structural_does_not_split_floors_by_bank():
    """`code_identifier` and `uuid` both roll up to `id:tech` (d33a domain
    aggregation), so an extra same-domain bank is not a NEW floor — the
    check is floor-grained, not bank-grained; a surprise worth pinning."""
    op = InjectOperator()
    problems = op.structural(
        _PARENT, "cPGES 550e8400-e29b-41d4-a716-446655440000 alternative medicine",
        Targets(spans=(SpanTarget(feature="code_identifier"),)),
    )
    assert problems == []


def test_inject_targets_counts_every_offered_surface():
    parent = pd.Series({"bank": "code_identifier", "surfaces": ("cPGES", "lipoxinA4")})
    target = InjectOperator().targets("symbol_pile_no_grammar", parent).spans[0]
    assert (target.feature, target.min_count) == ("code_identifier", 2)


def test_inject_candidate_mints_a_key_against_the_grounding_doc():
    parent = pd.Series({
        "query_id": "q1", "dataset": "beir-nfcorpus", "grounding_doc_id": "MED-1",
    })
    row = InjectOperator().candidate(parent, "id:tech", "cPGES alternative medicine", 1)
    assert row.grounding_doc_id == "MED-1"
    assert row.answer_key is AnswerKeyPath.MINTED
    assert row.provenance == "doc_grounded"
