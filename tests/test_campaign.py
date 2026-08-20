"""`AugmentationCampaign.plan()` (2026-08 follow-up): it must resolve a real
operator/gate for a CELL floor via `loop.demand()`, the same cell-aware path
`run()` already uses — not `operator_for()`, which only ever understood bare
floor labels and reported every cell name as unservable.
"""

import pandas as pd

from augmentation.campaign import (
    DROPPED_EXHAUSTED_CHANCES,
    FAULT_STREAK,
    MAX_CHANCES,
    PRODUCE,
    SKIP_NEEDS_SELECTION,
    SKIP_NO_OPERATOR,
    AugmentationCampaign,
    RowBudget,
)
from augmentation.config import AugmentationConfig, AugmentationPaths
from augmentation.core import CreditGate
from augmentation.engine import AugmentationOutcome, ErrorCase
from augmentation.loop import AugmentationLoop
from augmentation.operators import DecorateOperator, InjectOperator, StatRewrite
from augmentation.pool import GeneratedPool
from augmentation.qrels import AugmentationQrels
from query_taxonomy.features import FeatureExtractor
from scripts.run_v3_generation import synthetic_rung
from taxonomy_generators.verify import VerifyReport, verify

IDENT = "structured_identifiers."


class ModelDecorate(DecorateOperator):
    """Decorate with its deterministic path switched off. The scheduler's
    contract is about ENGINE faults, which only a model-served family can
    raise — stat_rewrite UP is the real one, decorate is the cheap stand-in."""

    def apply(self, parent, floor, text, requirement=()):
        return None


class AlwaysFaultsEngine:
    """Every call is a fault (rounds_exhausted) — proves the scheduler's K
    ceiling holds independent of any particular drop reason."""

    def __init__(self) -> None:
        self.calls = 0

    def run(self, *args, **kwargs):
        self.calls += 1
        return AugmentationOutcome(
            text=None, accepted=False, attempts=1, error=ErrorCase.ROUNDS_EXHAUSTED,
        )

    def accept(self, text, targets):
        """Every path faults, including the deterministic one — otherwise a
        family that stopped calling the model would quietly stop faulting."""
        return VerifyReport(passed=False, checks=())


class FaultsThenSucceeds:
    """Faults exactly `n_fail` times, then a real, verifiable accept —
    proves a floor that recovers gets banked rather than just dropped."""

    def __init__(self, n_fail: int, text: str) -> None:
        self.n_fail = n_fail
        self.calls = 0
        self.text = text
        self._extractor = FeatureExtractor(engines=None)

    def run(self, instruction, prompt, targets, *, tool_loop=False):
        self.calls += 1
        if self.calls <= self.n_fail:
            return AugmentationOutcome(
                text=None, accepted=False, attempts=1,
                error=ErrorCase.ROUNDS_EXHAUSTED,
            )
        report = verify(self.text, targets, extractor=self._extractor)
        return AugmentationOutcome(
            text=self.text, accepted=report.passed, attempts=1, checks=report.checks,
        )


def _many(n: int) -> pd.DataFrame:
    return pd.DataFrame([
        {
            "dataset": "beir-nfcorpus", "query_id": f"q{i}",
            "query": f"query number {i}", "checkable": True, "floors": [],
        }
        for i in range(n)
    ])


class PassThroughInject(InjectOperator):
    def eligible(self, selection, floor):
        return selection


class OneParent:
    def __init__(self, row: dict) -> None:
        self.row = row

    def available(self):
        return pd.DataFrame([self.row])

    def hydrate(self, frame):
        return frame

    def gold_text(self, parent, *, chars: int = 1200):
        return ""


def _sheet(floor: str) -> pd.DataFrame:
    return pd.DataFrame([{
        "slice": "cell", "floor": floor, "amount": 10.0,
        "credit": 0.0, "missing": 5.0, "reason": "exhausted",
    }])


def test_plan_resolves_a_cell_floor_through_demand_not_operator_for(tmp_path):
    """The exact bug: `version_pinned_technical` is a cell name, not a bare
    floor label — `operator_for()` alone reports it unservable."""
    paths = AugmentationPaths(data_dir=tmp_path)
    paths.catalog.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{
        "dataset": "beir-nfcorpus", "query_id": "q1", "checkable": True,
        "length.length_words": 2.0,
    }]).to_parquet(paths.catalog, index=False)
    config = AugmentationConfig(paths=paths)

    parent = {
        "dataset": "beir-nfcorpus", "query_id": "q1", "checkable": True,
        "query": "nginx config", "floors": [],
        "surfaces": ("2.1.3",), "bank": "version_string",
        "grounding_doc_id": "DOC-1", "length.length_words": 2.0,
    }
    sheet_path = tmp_path / "sheet.parquet"
    _sheet("version_pinned_technical").to_parquet(sheet_path, index=False)
    loop = AugmentationLoop(
        pd.DataFrame(columns=["dataset", "query_id", "query", "checkable"]),
        config=config,
        operators=(PassThroughInject(config), StatRewrite(config)),
        sheet_path=sheet_path,
        pool=GeneratedPool(paths),
        qrels=AugmentationQrels(paths),
        parents=OneParent(parent),
    )

    plan = AugmentationCampaign(loop).plan()
    row = plan.iloc[0]
    assert row["floor"] == "version_pinned_technical"
    assert row["action"] == PRODUCE
    assert "inject" in row["operator"]
    assert row["gate"] == str(CreditGate.COHERENCE_GATE)


def test_plan_still_resolves_a_bare_floor_label_gate_free(tmp_path):
    """The pre-existing, non-cell path (`marker:greeting`, served by the
    gate-free Decorate) must keep working exactly as before."""
    paths = AugmentationPaths(data_dir=tmp_path)
    config = AugmentationConfig(paths=paths)
    # five parents for a `missing` of five: the assertion below is about the
    # pilot cap not applying, so supply must not be what caps it
    selection = pd.DataFrame([{
        "dataset": "beir-nfcorpus", "query_id": f"q{i}",
        "query": "hello there, what is the capital of France",
        "checkable": True,
    } for i in range(5)])
    sheet_path = tmp_path / "sheet.parquet"
    _sheet("marker:greeting").to_parquet(sheet_path, index=False)
    loop = AugmentationLoop(
        selection,
        config=config,
        operators=(ModelDecorate(config),),
        sheet_path=sheet_path,
        pool=GeneratedPool(paths),
        qrels=AugmentationQrels(paths),
    )

    plan = AugmentationCampaign(loop).plan()
    row = plan.iloc[0]
    assert row["floor"] == "marker:greeting"
    assert row["operator"] == "decorate"
    assert row["gate"] == str(CreditGate.NONE)
    assert row["action"] == PRODUCE
    assert row["target_rows"] == 5   # gate-free: full `missing`, no pilot cap


def test_a_cell_short_of_parents_routes_the_remainder_to_synthesis(tmp_path):
    """`version_pinned_technical` needs more rows than it has parents, so the
    plan must split the demand rather than claim parents can cover it."""
    paths = AugmentationPaths(data_dir=tmp_path)
    paths.catalog.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{
        "dataset": "beir-nfcorpus", "query_id": "q1", "checkable": True,
        "length.length_words": 2.0,
    }]).to_parquet(paths.catalog, index=False)
    config = AugmentationConfig(paths=paths)
    parent = {
        "dataset": "beir-nfcorpus", "query_id": "q1", "checkable": True,
        "query": "nginx config", "floors": [],
        "surfaces": ("2.1.3",), "bank": "version_string",
        "grounding_doc_id": "DOC-1", "length.length_words": 2.0,
    }
    sheet_path = tmp_path / "sheet.parquet"
    _sheet("version_pinned_technical").to_parquet(sheet_path, index=False)
    loop = AugmentationLoop(
        pd.DataFrame(columns=["dataset", "query_id", "query", "checkable"]),
        config=config,
        operators=(PassThroughInject(config), StatRewrite(config)),
        sheet_path=sheet_path, pool=GeneratedPool(paths),
        qrels=AugmentationQrels(paths), parents=OneParent(parent),
    )

    row = AugmentationCampaign(loop, pilot_n=5).plan().iloc[0]
    assert row["target_rows"] == 1, "one parent reaches one row"
    assert row["synthetic_rows"] == 4, "the rest is the synthetic rung's"
    assert row["action"] == PRODUCE


def test_a_cell_its_own_parents_satisfy_is_selection_not_synthesis(tmp_path):
    """`damage_free_query`'s unspent parents already measure into it, so the
    line must never be priced as generation: paying to mint a clean query buys
    supply the pool is already holding, and the plan once read all 2,436 of
    them as owed to the synthetic rung."""
    paths = AugmentationPaths(data_dir=tmp_path)
    paths.catalog.parent.mkdir(parents=True, exist_ok=True)
    config = AugmentationConfig(paths=paths)
    parent = {
        "dataset": "beir-nfcorpus", "query_id": "q1", "checkable": True,
        "query": "how do antibiotics work", "floors": [],
        "derived.corruption_spans": 0.0,
    }
    sheet_path = tmp_path / "sheet.parquet"
    _sheet("damage_free_query").to_parquet(sheet_path, index=False)
    loop = RecordingLoop(
        pd.DataFrame(columns=["dataset", "query_id", "query", "checkable"]),
        config=config, operators=(StatRewrite(config),),
        sheet_path=sheet_path, pool=GeneratedPool(paths),
        qrels=AugmentationQrels(paths), parents=OneParent(parent),
    )

    plan = AugmentationCampaign(loop).plan()
    synthetic_rung(loop, plan, None, RowBudget())

    row = plan.iloc[0]
    assert row["action"] == SKIP_NEEDS_SELECTION
    assert row["synthetic_rows"] == 0 and row["target_rows"] == 0
    assert loop.synthesized == [], "a selection shortfall costs no completion"


class RecordingLoop(AugmentationLoop):
    """Records what the campaign asks the synthetic rung for, without paying
    for it."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.synthesized: list[tuple[str, int, str]] = []

    def synthesize(self, floor, n, *, source_dataset, **kwargs):
        self.synthesized.append((floor, n, source_dataset))
        return pd.DataFrame([{"query_id": f"syn-{floor}-{i}"} for i in range(n)])


def test_stage_5_mints_the_remainder_the_campaign_leaves_alone(tmp_path):
    """The gate is a fork, not a stop sign: what rung 1 cannot reach must
    actually reach the rung that can, with a lane to borrow distractors from —
    and exactly once, from the driver's stage 5, never also from `run()`."""
    paths = AugmentationPaths(data_dir=tmp_path)
    paths.catalog.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{
        "dataset": "beir-nfcorpus", "query_id": "q1", "checkable": True,
        "length.length_words": 2.0,
    }]).to_parquet(paths.catalog, index=False)
    config = AugmentationConfig(paths=paths)
    parent = {
        "dataset": "beir-nfcorpus", "query_id": "q1", "checkable": True,
        "query": "nginx config", "floors": [],
        "surfaces": ("2.1.3",), "bank": "version_string",
        "grounding_doc_id": "DOC-1", "length.length_words": 2.0,
    }
    sheet_path = tmp_path / "sheet.parquet"
    _sheet("version_pinned_technical").to_parquet(sheet_path, index=False)
    loop = RecordingLoop(
        pd.DataFrame(columns=["dataset", "query_id", "query", "checkable"]),
        config=config, engine=AlwaysFaultsEngine(),
        operators=(PassThroughInject(config), StatRewrite(config)),
        sheet_path=sheet_path, pool=GeneratedPool(paths),
        qrels=AugmentationQrels(paths), parents=OneParent(parent),
    )

    # the driver's own sequence: price it (stage 3), produce from parents
    # (stage 4), then mint the remainder (stage 5)
    campaign = AugmentationCampaign(loop, pilot_n=5)
    plan = campaign.plan()
    campaign.run()
    assert loop.synthesized == [], "the campaign never mints from nothing"

    synthetic_rung(loop, plan, None, RowBudget())

    assert loop.synthesized == [("version_pinned_technical", 4, "beir-nfcorpus")], (
        "the remainder must reach synthesize() once, with the cell's own lane"
    )


def test_a_cell_with_no_parents_is_not_given_an_arbitrary_lane(tmp_path):
    """No parents means no lane to borrow distractors from. Picking one would
    be inventing policy, so stage 5 reports and leaves it."""
    paths = AugmentationPaths(data_dir=tmp_path)
    config = AugmentationConfig(paths=paths)
    sheet_path = tmp_path / "sheet.parquet"
    _sheet("version_pinned_technical").to_parquet(sheet_path, index=False)
    loop = RecordingLoop(
        pd.DataFrame(columns=["dataset", "query_id", "query", "checkable"]),
        config=config, engine=AlwaysFaultsEngine(),
        operators=(PassThroughInject(config), StatRewrite(config)),
        sheet_path=sheet_path, pool=GeneratedPool(paths),
        qrels=AugmentationQrels(paths),
    )

    synthetic_rung(
        loop, AugmentationCampaign(loop, pilot_n=5).plan(), None, RowBudget()
    )

    assert loop.synthesized == []


def test_a_cell_whose_band_nobody_serves_reaches_zero_however_many_parents():
    """The case parent-count gating waves straight through: an unserved band
    means every row lands in some other cell, so rung 1 fills this one never."""
    from augmentation.dispatch import CellPlan, Stage, Step

    crowded = pd.DataFrame([{"query_id": f"q{i}"} for i in range(400_000)])
    unserved = Step((), Stage.CORPUS, None)
    assert CellPlan("c", crowded, (), ()).reachable == 400_000
    assert CellPlan("c", crowded, (), (unserved,)).reachable == 0


def test_plan_reports_no_operator_for_an_unregistered_bare_floor(tmp_path):
    """A bare floor nothing serves must still say so — distinctly from a
    cell whose stages leave no servable parent."""
    paths = AugmentationPaths(data_dir=tmp_path)
    config = AugmentationConfig(paths=paths)
    selection = pd.DataFrame(columns=["dataset", "query_id", "query", "checkable"])
    sheet_path = tmp_path / "sheet.parquet"
    _sheet("nonexistent:floor").to_parquet(sheet_path, index=False)
    loop = AugmentationLoop(
        selection, config=config, operators=(ModelDecorate(config),),
        sheet_path=sheet_path, pool=GeneratedPool(paths), qrels=AugmentationQrels(paths),
    )

    plan = AugmentationCampaign(loop).plan()
    row = plan.iloc[0]
    assert row["action"] == SKIP_NO_OPERATOR
    assert row["operator"] is None


def test_the_row_budget_ends_the_round_before_the_next_floor(tmp_path):
    """`--limit n`: the round produces n rows and stops. Not "n per floor" and
    not "finish the floor you started" — the second floor must never cost a
    single call."""
    paths = AugmentationPaths(data_dir=tmp_path)
    config = AugmentationConfig(paths=paths)
    sheet_path = tmp_path / "sheet.parquet"
    pd.concat([
        _sheet("marker:greeting"),                          # missing 5, served first
        _sheet("marker:politeness").assign(missing=4.0),
    ], ignore_index=True).to_parquet(sheet_path, index=False)
    engine = FaultsThenSucceeds(
        n_fail=0, text="hello there, what is the capital of France",
    )
    loop = AugmentationLoop(
        _many(15), config=config, engine=engine,
        operators=(ModelDecorate(config),),
        sheet_path=sheet_path, pool=GeneratedPool(paths), qrels=AugmentationQrels(paths),
    )
    budget = RowBudget(2)

    summary = AugmentationCampaign(loop).run(budget)

    assert int(summary["accepted"].sum()) == 2, "the budget is the whole round"
    assert engine.calls == 2, "the second floor must not be attempted at all"
    assert budget.exhausted()


def test_stage_5_mints_only_what_the_budget_has_left(tmp_path):
    """One budget across both spending stages: stage 5 asks for the remainder,
    never for its own full share."""
    paths = AugmentationPaths(data_dir=tmp_path)
    config = AugmentationConfig(paths=paths)
    sheet_path = tmp_path / "sheet.parquet"
    _sheet("marker:greeting").to_parquet(sheet_path, index=False)
    loop = RecordingLoop(
        _many(15), config=config, engine=AlwaysFaultsEngine(),
        operators=(ModelDecorate(config),),
        sheet_path=sheet_path, pool=GeneratedPool(paths), qrels=AugmentationQrels(paths),
    )
    plan = pd.DataFrame([{
        "floor": "version_pinned_technical", "missing": 10.0, "action": PRODUCE,
        "synthetic_rows": 8, "source_dataset": "beir-nfcorpus",
    }])
    budget = RowBudget(3)
    budget.add(1)   # as if stage 4 had banked one row

    synthetic_rung(loop, plan, None, budget)

    assert loop.synthesized == [("version_pinned_technical", 2, "beir-nfcorpus")]


def test_a_floor_that_always_faults_costs_at_most_the_k_ceiling(tmp_path):
    """The load-bearing guarantee (d59): whatever else might be wrong, no
    floor can ever cost more than MAX_CHANCES x FAULT_STREAK faulty attempts
    before being dropped — proven directly against real supply, not inferred."""
    paths = AugmentationPaths(data_dir=tmp_path)
    config = AugmentationConfig(paths=paths)
    sheet_path = tmp_path / "sheet.parquet"
    _sheet("marker:greeting").to_parquet(sheet_path, index=False)
    engine = AlwaysFaultsEngine()
    loop = AugmentationLoop(
        _many(15), config=config, engine=engine,
        operators=(ModelDecorate(config),),
        sheet_path=sheet_path, pool=GeneratedPool(paths), qrels=AugmentationQrels(paths),
    )

    summary = AugmentationCampaign(loop).run()

    assert engine.calls == MAX_CHANCES * FAULT_STREAK
    row = summary.iloc[0]
    assert row["action"] == DROPPED_EXHAUSTED_CHANCES
    assert row["accepted"] == 0


def test_run_alone_honours_the_ceiling_without_the_campaign(tmp_path):
    """`--floor` calls run() directly. Its ceiling used to default to None, so
    the 'worst-case-provable ceiling on wasted spend' held only via the
    campaign and a manual run chewed through every eligible parent."""
    paths = AugmentationPaths(data_dir=tmp_path)
    config = AugmentationConfig(paths=paths)
    sheet_path = tmp_path / "sheet.parquet"
    _sheet("marker:greeting").to_parquet(sheet_path, index=False)
    engine = AlwaysFaultsEngine()
    loop = AugmentationLoop(
        _many(50), config=config, engine=engine,
        operators=(ModelDecorate(config),),
        sheet_path=sheet_path, pool=GeneratedPool(paths),
        qrels=AugmentationQrels(paths),
    )

    banked = loop.run("marker:greeting", n=5)

    assert banked.empty
    assert engine.calls == FAULT_STREAK, (
        f"run() stopped after {engine.calls} faults, not {FAULT_STREAK} — "
        "50 parents were available to burn"
    )
    # opting out explicitly still burns the whole queue, for a caller that
    # genuinely wants that
    loop.run("marker:greeting", n=5, max_consecutive_faults=None)
    assert engine.calls > FAULT_STREAK * 2


def test_a_floor_that_recovers_after_one_lost_chance_still_gets_fully_served(tmp_path):
    """Losing a chance is not the end — a floor that starts failing and then
    genuinely recovers must bank its full need, not just escape with a few
    rows, and must never be reported as dropped."""
    paths = AugmentationPaths(data_dir=tmp_path)
    config = AugmentationConfig(paths=paths)
    sheet_path = tmp_path / "sheet.parquet"
    _sheet("marker:greeting").to_parquet(sheet_path, index=False)
    # 3 faults burn chance 1 (requeued); the 4th call (a FRESH parent, thanks
    # to `exclude`) faults once more into chance 2, then every call after
    # that succeeds — proving recovery, not just a lucky single retry
    engine = FaultsThenSucceeds(
        n_fail=FAULT_STREAK + 1, text="hello there, what is the capital of France",
    )
    loop = AugmentationLoop(
        _many(15), config=config, engine=engine,
        operators=(ModelDecorate(config),),
        sheet_path=sheet_path, pool=GeneratedPool(paths), qrels=AugmentationQrels(paths),
    )

    summary = AugmentationCampaign(loop).run()

    row = summary.iloc[0]
    assert row["action"] == PRODUCE
    assert row["accepted"] == 5   # the floor's full `missing`
