"""Stress tests for the cell dispatch/staging layer (SPEC d51-d54): the
harm-ordered stage classifier, the shrinking-frame planner, and the
plan-to-postcondition crossing. Real operators are used wherever their
`eligible()` touches no file (Decorate, StatRewrite's classification-only
paths); a config tweak makes one span mintable where the veto matters,
following `tests/test_cells.py`'s `_with_negation_weavable` pattern.
"""

import pandas as pd
import pytest

from augmentation.config import AugmentationConfig
from augmentation.dispatch import (
    Call,
    CellPlan,
    Stage,
    Step,
    calls_for,
    plan,
    plan_report,
    planned_targets,
    reduces,
    requirements,
    stage_of,
    unreachable,
)
from augmentation.operators import DecorateOperator, StatRewrite, default_operators
from composition.cells import ArchetypeCell, AxisBand

IDENT = "structured_identifiers."
MARK = "sentence_markers."


def _cell(name: str, *bands: AxisBand, any_of: tuple[AxisBand, ...] = ()) -> ArchetypeCell:
    return ArchetypeCell(
        name=name, predicate=bands, any_of=any_of,
        predicts=("sparse_only",), rationale="probe", source="test",
    )


def _weavable(feature: str) -> tuple:
    """Operators as if `feature` had been registered as a decoration — the
    move d52(g)'s veto exists to catch, not a supported config."""
    base = AugmentationConfig()
    config = base.model_copy(
        update={"decorations": {**base.decorations, feature: f"a {feature}"}}
    )
    return default_operators(config)


def _rows(**columns) -> pd.DataFrame:
    """A parent frame with the bookkeeping columns dispatch/operators read."""
    base = {"dataset": "d", "checkable": True}
    base.update(columns)
    n = max((len(v) for v in base.values() if isinstance(v, list)), default=1)
    base.setdefault("query_id", [f"q{i}" for i in range(n)])
    return pd.DataFrame({
        k: (v if isinstance(v, list) else [v] * n) for k, v in base.items()
    })


# --- reduces(): what "bounded only from above" actually covers -----------


def test_reduces_is_true_only_for_a_bare_upper_bound_scalar():
    """A two-sided or span-shaped band is never `reduces()`, even though a
    below-only span reads as "less of something" in plain language."""
    below_only_scalar = AxisBand(column="length.length_words", below=6)
    two_sided_scalar = AxisBand(column="length.length_words", at_least=3, below=10)
    span_presence = AxisBand(column=f"{IDENT}uuid", at_least=1)
    span_absence = AxisBand(column=f"{IDENT}number", below=1)
    assert reduces((below_only_scalar,))
    assert not reduces((two_sided_scalar,))
    assert not reduces((span_presence,))
    assert not reduces((span_absence,))


def test_two_sided_band_is_one_up_mintable_requirement_never_constrain():
    """d53(h): `at_least 3, below 10` is a range the model must land inside
    in one move, not a pre-filter plus a cut — `headroom()` was deleted."""
    band = AxisBand(column="length.length_words", at_least=3, below=10)
    stage, operator = stage_of((band,), default_operators())
    assert stage is Stage.QUERY_ONLY
    assert operator.declaration.operator == "stat_rewrite"


def test_a_bare_upper_bound_is_select_when_its_axis_has_no_direction():
    """`reduces()` alone does not make a band mintable: an axis nobody
    declared UP or DOWN for still classifies SELECT, not CONSTRAIN."""
    band = AxisBand(column="depth.nesting_depth", below=4)
    assert reduces((band,))
    stage, operator = stage_of((band,), default_operators())
    assert stage is Stage.SELECT
    assert operator is None


def test_mixed_any_of_family_stage_depends_on_which_band_is_listed_first():
    """Neither real cell mixes types in one `any_of`, but nothing stops one:
    reordering the SAME two bands flips (stage, operator) from
    (CORPUS, inject) to (QUERY_ONLY, stat_rewrite) while `reduces()` — which
    reads the whole family — stays False either way. The single `operator`
    a Step carries is decided by iteration order, not by any harm rule."""
    span_band = AxisBand(column=f"{IDENT}uuid", at_least=1)
    scalar_down = AxisBand(column="length.length_words", below=6)
    ops = default_operators()

    stage_a, op_a = stage_of((span_band, scalar_down), ops)
    stage_b, op_b = stage_of((scalar_down, span_band), ops)

    assert not reduces((span_band, scalar_down))
    assert not reduces((scalar_down, span_band))
    assert (stage_a, op_a.declaration.operator) == (Stage.CORPUS, "inject")
    assert (stage_b, op_b.declaration.operator) == (Stage.QUERY_ONLY, "stat_rewrite")


# --- stage order: only SELECT is an entry requirement (d53) ---------------


def test_select_drops_a_violator_but_constrain_keeps_one():
    """A parent lacking the SELECT-required uuid never becomes a candidate;
    a parent 40 words over the CONSTRAIN bound stays a candidate regardless
    — the cut is a rewrite the model performs, never a filter (d53b)."""
    cell = _cell(
        "select_vs_constrain",
        AxisBand(column=f"{IDENT}uuid", at_least=1),
        AxisBand(column="length.length_words", below=6),
    )
    pool = _rows(
        **{f"{IDENT}uuid": [1.0, 0.0], "length.length_words": [40.0, 3.0]}
    )
    # only StatRewrite is wired: no operator mints uuid, so it stays SELECT
    result = plan(cell, pool, (StatRewrite(AugmentationConfig()),))

    assert list(result.parents["query_id"]) == ["q0"]
    assert result.parents.iloc[0]["length.length_words"] == 40.0
    assert [s.stage for s in result.mints] == [Stage.CONSTRAIN]
    assert not result.unsatisfied


# --- CellPlan's four states -----------------------------------------------


def test_needs_selection_when_nothing_is_left_to_mint():
    """Parents already satisfy a SELECT-only cell: mints is empty, so the
    shortfall is the fill's selection, not a generation target (d52i)."""
    cell = _cell("already_true", AxisBand(column=f"{MARK}acronym", at_least=1))
    pool = _rows(**{f"{MARK}acronym": 2.0})
    result = plan(cell, pool, default_operators())

    assert result.servable and result.needs_selection
    assert not result.mints and not result.unsatisfied and not result.operators


def test_servable_with_mints_and_no_partial():
    """A single QUERY_ONLY band fully served leaves nothing unsatisfied and
    something to mint, so `needs_selection` must be False."""
    cell = _cell("politeness_only", AxisBand(column=f"{MARK}politeness", at_least=1))
    pool = _rows(checkable=True)
    result = plan(cell, pool, (DecorateOperator(AugmentationConfig()),))

    assert result.servable and not result.needs_selection and not result.partial
    assert [s.stage for s in result.mints] == [Stage.QUERY_ONLY]
    assert result.operators[0].declaration.operator == "decorate"


def test_partial_when_a_mint_succeeds_beside_an_unservable_rebuild():
    """A served QUERY_ONLY band and a REBUILD-vetoed band in the same cell:
    the row is worth producing, but only part of the predicate is checked."""
    cell = _cell(
        "mixed",
        AxisBand(column=f"{MARK}politeness", at_least=1),
        AxisBand(column=f"{MARK}negation", at_least=1),
    )
    pool = _rows(checkable=True)
    result = plan(cell, pool, _weavable("negation"))

    assert result.servable and result.partial and not result.needs_selection
    assert [s.stage for s in result.mints] == [Stage.QUERY_ONLY]
    assert [s.stage for s in result.unsatisfied] == [Stage.REBUILD]


def test_pure_select_failure_leaves_no_unsatisfied_trail():
    """A SELECT band a parent fails is never a Step at all — it just narrows
    `candidates` to nothing — so `unsatisfied` cannot be read as "why this
    cell has no parents": it can be empty even when nothing survived."""
    cell = _cell("unreachable", AxisBand(column=f"{MARK}acronym", at_least=1))
    pool = _rows(**{f"{MARK}acronym": 0.0})
    result = plan(cell, pool, default_operators())

    assert not result.servable
    assert not result.needs_selection
    assert not result.unsatisfied and not result.partial


def test_select_failure_beside_a_rebuild_reports_only_the_rebuild():
    """`partial` and `servable` are independent: a cell can be unservable
    (SELECT starves it) while still carrying a reported REBUILD veto."""
    cell = _cell(
        "unreachable_and_vetoed",
        AxisBand(column=f"{MARK}acronym", at_least=1),
        AxisBand(column=f"{MARK}negation", at_least=1),
    )
    pool = _rows(**{f"{MARK}acronym": 0.0})
    result = plan(cell, pool, _weavable("negation"))

    assert not result.servable and result.partial
    assert [s.stage for s in result.unsatisfied] == [Stage.REBUILD]


def test_any_of_family_stays_servable_through_its_safe_sibling():
    """The REBUILD veto strikes one alternative, not the family: politeness
    still reaches QUERY_ONLY and the plan mints it (d52g)."""
    cell = _cell(
        "family",
        any_of=(
            AxisBand(column=f"{MARK}negation", at_least=1),
            AxisBand(column=f"{MARK}politeness", at_least=1),
        ),
    )
    pool = _rows(checkable=True)
    result = plan(cell, pool, _weavable("negation"))

    assert result.servable and not result.partial
    assert [s.stage for s in result.mints] == [Stage.QUERY_ONLY]


# --- planned_targets: the postcondition crossing --------------------------


def test_planned_targets_drops_an_unsatisfied_requirement_by_value():
    """The identity trap: `plan()` and `planned_targets` each build their own
    `requirements(cell)`, so comparing tuples by `id()` never matched and a
    REBUILD band leaked into the postcondition. Frozen bands compare by value,
    so the unserved requirement must now be absent."""
    cell = _cell(
        "leaky",
        AxisBand(column=f"{MARK}acronym", at_least=1),
        AxisBand(column=f"{MARK}negation", at_least=1),
    )
    pool = _rows(**{f"{MARK}acronym": 2.0, f"{MARK}negation": 0.0})
    result = plan(cell, pool, _weavable("negation"))
    assert result.partial, "fixture must actually carry an unsatisfied step"

    targets = planned_targets(result, cell, result.parents.iloc[0])
    assert {t.feature for t in targets.spans} == {"acronym"}


def test_planned_targets_never_asks_for_what_the_plan_could_not_serve():
    """d52(f): acceptance targets are every requirement except the ones
    nothing could serve."""
    cell = _cell(
        "leaky",
        AxisBand(column=f"{MARK}acronym", at_least=1),
        AxisBand(column=f"{MARK}negation", at_least=1),
    )
    pool = _rows(**{f"{MARK}acronym": 2.0, f"{MARK}negation": 0.0})
    result = plan(cell, pool, _weavable("negation"))

    targets = planned_targets(result, cell, result.parents.iloc[0])
    expected = len(requirements(cell)) - len(result.unsatisfied)
    assert len(targets.spans) + len(targets.stats) == expected


def test_planned_targets_resolves_an_any_of_family_via_bank():
    """The happy path the docstring promises: `bank` names the alternative
    actually woven, so the OTHER sibling's target is never asked for."""
    cell = _cell(
        "family",
        any_of=(
            AxisBand(column=f"{IDENT}uuid", at_least=1),
            AxisBand(column=f"{IDENT}api_key", at_least=1),
        ),
    )
    assert len(requirements(cell)) == 1, "the family is one requirement"
    parent = pd.Series({"bank": "api_key"})
    empty_plan = CellPlan(cell.name, pd.DataFrame(), (), ())
    targets = planned_targets(empty_plan, cell, parent)
    assert {t.feature for t in targets.spans} == {"api_key"}


@pytest.mark.parametrize("bank_value", [None, float("nan")])
def test_planned_targets_falls_back_to_the_first_alternative_when_bank_is_unresolved(
    bank_value,
):
    """Missing `bank` and NaN `bank` both take the `requirement[0]` branch —
    coincidentally correct only because nothing checks that band actually
    matches what was woven; a family ordered the other way would silently
    check the wrong alternative."""
    cell = _cell(
        "family",
        any_of=(
            AxisBand(column=f"{IDENT}uuid", at_least=1),
            AxisBand(column=f"{IDENT}api_key", at_least=1),
        ),
    )
    parent = pd.Series({"bank": bank_value}) if bank_value is not None else pd.Series()
    empty_plan = CellPlan(cell.name, pd.DataFrame(), (), ())
    targets = planned_targets(empty_plan, cell, parent)
    assert {t.feature for t in targets.spans} == {"uuid"}


# --- determinism -----------------------------------------------------------


def test_plan_is_deterministic_across_repeated_calls():
    """Decorate's `eligible()` shuffles with `config.seed`, the same
    mechanism Inject relies on — a plan must not depend on call order."""
    cell = _cell("politeness_only", AxisBand(column=f"{MARK}politeness", at_least=1))
    pool = _rows(checkable=[True] * 5, query_id=[f"q{i}" for i in range(5)])
    operators = (DecorateOperator(AugmentationConfig()),)

    first = plan(cell, pool, operators)
    second = plan(cell, pool, operators)
    pd.testing.assert_frame_equal(
        first.parents.reset_index(drop=True), second.parents.reset_index(drop=True)
    )


# --- degenerate inputs -------------------------------------------------


def test_empty_pool_plans_cleanly_to_nothing():
    """An empty frame (right columns, zero rows) must not raise — every
    band's mask still needs its column to exist, just no rows to hold it."""
    cell = _cell("select_only", AxisBand(column=f"{MARK}acronym", at_least=1))
    empty = pd.DataFrame({
        f"{MARK}acronym": pd.Series(dtype=float),
        "checkable": pd.Series(dtype=bool),
        "dataset": pd.Series(dtype=object),
        "query_id": pd.Series(dtype=object),
    })
    result = plan(cell, empty, default_operators())
    assert not result.servable and not result.mints and not result.unsatisfied


def test_a_select_band_missing_from_the_frame_raises_a_clear_key_error():
    """SELECT filtering reads the band's column straight off the frame with
    no existence check, so a missing column fails fast and loud, not a
    silent all-True or all-False verdict."""
    cell = _cell("select_only", AxisBand(column=f"{MARK}acronym", at_least=1))
    pool = _rows(checkable=True)  # no acronym column at all
    with pytest.raises(KeyError):
        plan(cell, pool, default_operators())


def test_every_requirement_unsatisfiable_still_reports_cleanly():
    """A cell where the SELECT band starves the pool AND the REBUILD band
    is vetoed: nothing crashes, and the two failure modes are visible
    exactly where each one is tracked (unsatisfied vs. empty parents)."""
    cell = _cell(
        "hopeless",
        AxisBand(column=f"{MARK}acronym", at_least=1),
        AxisBand(column=f"{MARK}negation", at_least=1),
    )
    pool = _rows(**{f"{MARK}acronym": 0.0, f"{MARK}negation": 0.0})
    result = plan(cell, pool, _weavable("negation"))
    assert not result.servable
    assert [s.stage for s in result.unsatisfied] == [Stage.REBUILD]


# --- the top-level report --------------------------------------------------


def test_plan_report_action_reflects_each_plan_state():
    """`select` / `augment` / `construct` must track `needs_selection` /
    `servable` exactly — this is the sentence a human reads off the sheet."""
    select_cell = _cell("already_true", AxisBand(column=f"{MARK}acronym", at_least=1))
    augment_cell = _cell(
        "politeness_only", AxisBand(column=f"{MARK}politeness", at_least=1)
    )
    # unreachable needs MORE acronyms than any row carries, and nothing mints
    # them, so its SELECT stage empties the pool
    dead_cell = _cell("unreachable", AxisBand(column=f"{MARK}acronym", at_least=9))
    cells = {c.name: c for c in (select_cell, augment_cell, dead_cell)}

    pool = _rows(**{
        f"{MARK}acronym": [2.0, 0.0], f"{MARK}politeness": [0.0, 0.0],
        "checkable": [True, True],
    })
    sheet = pd.DataFrame([
        {"floor": name, "missing": 5.0} for name in cells
    ])
    report = plan_report(sheet, pool, cells, (DecorateOperator(AugmentationConfig()),))
    actions = dict(zip(report["cell"], report["action"]))

    assert actions["already_true"] == "select"
    assert actions["politeness_only"] == "augment"
    assert actions["unreachable"] == "construct"


def test_unreachable_is_false_for_a_non_cutting_call():
    """Only a CONSTRAIN call has a ceiling to break — an additive call is
    never flagged, whatever the parent carries."""
    band = AxisBand(column=f"{IDENT}uuid", at_least=1)
    call = Call(steps=(Step((band,), Stage.CORPUS, None),), verified=((band,),))
    heavy = pd.Series({"surfaces": ("one two three four five",)})
    assert not unreachable(call, heavy)


def test_unreachable_fires_when_mandatory_words_already_break_the_ceiling():
    """A mint's own surfaces are literal and cannot be dropped, so a ceiling
    below their combined length is broken before any rewrite (d55-followup) —
    catching the case an LLM cannot solve by construction, at zero spend."""
    band = AxisBand(column="length.length_words", below=4)
    call = Call(steps=(Step((band,), Stage.CONSTRAIN, None),), verified=((band,),))
    assert unreachable(call, pd.Series({"surfaces": ("foo bar baz qux",)}))
    assert not unreachable(call, pd.Series({"surfaces": ("v1",)}))
    assert not unreachable(call, pd.Series({"surfaces": ()}))


def _pinned_plan() -> tuple[ArchetypeCell, CellPlan, Step, Step]:
    """`version_pinned_technical`'s exact shape: a CORPUS span mint plus a
    two-sided scalar the direction table declares both ways."""
    version_band = AxisBand(column=f"{IDENT}version_string", at_least=1)
    length_band = AxisBand(column="length.length_words", at_least=3, below=10)
    cell = _cell("pinned", version_band, length_band)
    ops = default_operators()
    inject = next(op for op in ops if op.declaration.operator == "inject")
    stat = next(op for op in ops if op.declaration.operator == "stat_rewrite")
    version_step = Step((version_band,), Stage.CORPUS, inject)
    length_step = Step((length_band,), Stage.QUERY_ONLY, stat)
    return cell, CellPlan("pinned", pd.DataFrame(), (version_step, length_step), ()), version_step, length_step


def test_calls_for_drops_a_scalar_mint_the_parent_already_satisfies():
    """A parent already short enough needs no rewrite once Inject's own word
    is counted — StatRewrite is dropped from the call, though its requirement
    stays VERIFIED so a wrong guess is still caught (2026-08 follow-up: most
    parents for this cell shape need no editing at all, measured 689/4,802)."""
    cell, result, *_ = _pinned_plan()
    easy = pd.Series({"length.length_words": 6.0, "surfaces": ("v1",)})
    calls = calls_for(result, cell, easy)
    assert len(calls) == 1
    assert [op.declaration.operator for op in calls[0].operators] == ["inject"]
    assert set(calls[0].verified) == {
        (cell.predicate[0],), (cell.predicate[1],),
    }, "the dropped requirement must still be part of what gets verified"


def test_calls_for_keeps_the_scalar_mint_when_genuinely_needed():
    """A parent already at the ceiling still gets both steps bundled — the
    free path is additive, it never removes editing where editing is real."""
    cell, result, *_ = _pinned_plan()
    hard = pd.Series({"length.length_words": 10.0, "surfaces": ("v1",)})
    calls = calls_for(result, cell, hard)
    assert len(calls) == 1
    assert {op.declaration.operator for op in calls[0].operators} == {
        "inject", "stat_rewrite",
    }


def test_calls_for_never_drops_a_constrain_step():
    """The free-path check only ever applies to additive steps — a CONSTRAIN
    step is the one case where "already holds" should be structurally
    impossible anyway (its own eligibility already filtered to violators),
    but this pins that the code never even asks the question for one."""
    band = AxisBand(column="length.length_words", below=4)
    cell = _cell("cut_only", band)
    cut_step = Step((band,), Stage.CONSTRAIN, StatRewrite(AugmentationConfig()))
    result = CellPlan("cut_only", pd.DataFrame(), (cut_step,), ())
    already_short = pd.Series({"length.length_words": 2.0, "surfaces": ()})
    calls = calls_for(result, cell, already_short)
    assert len(calls) == 1 and calls[0].cuts


def test_calls_for_returns_nothing_when_every_mint_is_free():
    """The degenerate case: every additive mint this parent needed turns out
    already satisfied, so there is nothing left to call at all."""
    band = AxisBand(column="length.length_words", at_least=3, below=10)
    cell = _cell("length_only", band)
    step = Step((band,), Stage.QUERY_ONLY, StatRewrite(AugmentationConfig()))
    result = CellPlan("length_only", pd.DataFrame(), (step,), ())
    already_fits = pd.Series({"length.length_words": 6.0, "surfaces": ()})
    assert calls_for(result, cell, already_fits) == ()
