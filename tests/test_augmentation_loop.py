"""The loop's acceptance path with the LLM stubbed out: a hand-written child
stands in for the model, so every check between the completion and the banked
row is exercised without spend. The model's own competence is not testable here
and needs one real call.
"""

import time

import pandas as pd
import pytest
from query_taxonomy.features import FeatureExtractor

from augmentation.config import AugmentationConfig, AugmentationPaths
from augmentation.engine import AugmentationOutcome, Spend
from augmentation.loop import AugmentationLoop
from augmentation.operators import InjectOperator, StatRewrite, default_operators
from augmentation.pool import GeneratedPool
from augmentation.qrels import AugmentationQrels
from taxonomy_generators.verify import Targets, verify


class ScriptedAugmenter:
    """Returns a fixed text and accepts exactly as the real engine does — the
    local re-measure, never the transcript."""

    def __init__(self, text: str) -> None:
        self.text = text
        self.calls: list[tuple[str, str, Targets]] = []
        self._extractor = FeatureExtractor(engines=None)

    def run(self, instruction, prompt, targets, *, tool_loop=False):
        self.calls.append((instruction, prompt, targets))
        report = verify(self.text, targets, extractor=self._extractor)
        return AugmentationOutcome(
            text=self.text,
            accepted=report.passed,
            attempts=1,
            checks=report.checks,
        )


def _loop(tmp_path, text: str, sheet: pd.DataFrame, parents) -> AugmentationLoop:
    paths = AugmentationPaths(data_dir=tmp_path)
    sheet_path = tmp_path / "sheet.parquet"
    sheet.to_parquet(sheet_path, index=False)
    config = AugmentationConfig(paths=paths)
    return AugmentationLoop(
        pd.DataFrame(columns=["dataset", "query_id", "query", "checkable"]),
        config=config,
        engine=ScriptedAugmenter(text),
        operators=_operators(config),
        sheet_path=sheet_path,
        pool=GeneratedPool(paths),
        qrels=AugmentationQrels(paths),
        parents=parents,
    )


class PassThroughInject(InjectOperator):
    """Inject with its lane join stubbed: the real instruction, targets,
    structural check and candidate, on a parent handed straight through."""

    def eligible(self, selection, floor):
        return selection


def _operators(config=None):
    """Inject's eligibility stubbed, every other family real."""
    config = config or AugmentationConfig()
    return (PassThroughInject(config), StatRewrite(config))


class OneParent:
    """A ParentPool stand-in: one hydrated parent, no catalog or corpus."""

    def __init__(self, row: dict, gold: str = "") -> None:
        self.row = row
        self.gold = gold

    def available(self):
        return pd.DataFrame([self.row])

    def hydrate(self, frame):
        return frame

    def gold_text(self, parent, *, chars: int = 1200):
        return self.gold


IDENT = "structured_identifiers."


def _sheet(cell: str, missing: float = 5.0) -> pd.DataFrame:
    return pd.DataFrame([{
        "slice": "cell", "floor": cell, "amount": 10.0,
        "credit": 0.0, "missing": missing, "reason": "exhausted",
    }])


def test_two_surfaces_reach_a_count_band_and_bank_one_row(tmp_path):
    """d54: a `code_identifier >= 2` cell is served by one parent carrying two
    distinct surfaces, and the accepted row is banked with a minted key."""
    parent = {
        "dataset": "beir-nfcorpus", "query_id": "q1", "checkable": True,
        "query": "alternative medicine", "floors": [],
        "surfaces": ("cPGES", "lipoxinA4"), "bank": "code_identifier",
        "grounding_doc_id": "MED-1",
        f"{IDENT}code_identifier": 0.0,
        "natural_language_signal.natural_language_share": 0.0,
        "length.length_words": 2.0,
    }
    # zero function words: 1 in 9 tokens is already 0.111 > 0.1
    child = "cPGES lipoxinA4"
    loop = _loop(tmp_path, child, _sheet("symbol_pile_no_grammar"), OneParent(parent))

    banked = loop.run("symbol_pile_no_grammar", n=1)
    assert len(banked) == 1, "the composed request should be satisfiable"
    row = banked.iloc[0]
    assert row["provenance"] == "doc_grounded"
    assert row["answer_key"] == "minted"
    assert row["grounding_doc_id"] == "MED-1"

    keys = loop.qrels.load()
    assert list(keys["source"]) == ["constructed"]
    assert list(keys["doc_id"]) == ["MED-1"]

    asked = loop.engine.calls[0][2]
    count = next(t for t in asked.spans if t.feature == "code_identifier")
    assert count.min_count == 2, "the band's count must reach the target"


def test_a_child_missing_one_surface_is_refused(tmp_path):
    """Inject's literal-containment check covers every surface, so dropping one
    during a cut is caught — the guard the composed cut relies on (d53g)."""
    parent = {
        "dataset": "beir-nfcorpus", "query_id": "q1", "checkable": True,
        "query": "alternative medicine", "floors": [],
        "surfaces": ("cPGES", "lipoxinA4"), "bank": "code_identifier",
        "grounding_doc_id": "MED-1",
        f"{IDENT}code_identifier": 0.0,
        "natural_language_signal.natural_language_share": 0.0,
        "length.length_words": 2.0,
    }
    inject = next(
        op for op in default_operators()
        if op.declaration.operator == "inject"
    )
    asked = Targets(spans=())
    problems = inject.structural(pd.Series(parent), "cPGES alone", asked)
    assert any("lipoxinA4" in p for p in problems)


def test_a_free_scalar_mint_costs_zero_extra_calls(tmp_path):
    """2026-08 follow-up: `version_pinned_technical` needs Inject
    (version_string) AND a 3-9 word length band. When Inject's own surface
    already pushes the bare query into that band, StatRewrite must never be
    called — one call, not two."""
    paths = AugmentationPaths(data_dir=tmp_path)
    paths.catalog.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{
        "dataset": "beir-nfcorpus", "query_id": "q1", "checkable": True,
        "length.length_words": 2.0,
    }]).to_parquet(paths.catalog, index=False)

    parent = {
        "dataset": "beir-nfcorpus", "query_id": "q1", "checkable": True,
        "query": "nginx config", "floors": [],
        "surfaces": ("2.1.3",), "bank": "version_string",
        "grounding_doc_id": "DOC-1",
        "length.length_words": 2.0,
    }
    child = "nginx config 2.1.3"
    loop = _loop(
        tmp_path, child, _sheet("version_pinned_technical"), OneParent(parent)
    )

    banked = loop.run("version_pinned_technical", n=1)
    assert len(banked) == 1, "Inject alone should satisfy the whole cell"
    assert len(loop.engine.calls) == 1, (
        "StatRewrite must be skipped — Inject's own surface already lands "
        "the length in [3, 10), so a second call would edit nothing"
    )


class PoisonEngine:
    """Fails the test if the loop ever calls it — proof that an incompatible
    parent is refused before any spend, not merely rejected after one."""

    def run(self, *args, **kwargs):
        raise AssertionError("engine.run() was called on an incompatible parent")


def test_incompatible_parent_never_reaches_the_engine(tmp_path):
    """`bare_acronym` (length_words < 3, CONSTRAIN, no corpus mint) handed a
    parent whose mandatory surface is already 5 words: no rewrite can keep
    that surface AND land under 3 words, so the row is refused pre-flight."""
    parent = {
        "dataset": "beir-nfcorpus", "query_id": "q1", "checkable": True,
        "query": "an acronym like NASA", "floors": [],
        "surfaces": ("one two three four five",),
        "sentence_markers.acronym": 1.0, "length.length_words": 5.0,
    }
    paths = AugmentationPaths(data_dir=tmp_path)
    sheet_path = tmp_path / "sheet.parquet"
    _sheet("bare_acronym").to_parquet(sheet_path, index=False)
    loop = AugmentationLoop(
        pd.DataFrame(columns=["dataset", "query_id", "query", "checkable"]),
        config=AugmentationConfig(paths=paths),
        engine=PoisonEngine(),
        operators=default_operators(),
        sheet_path=sheet_path,
        pool=GeneratedPool(paths),
        qrels=AugmentationQrels(paths),
        parents=OneParent(parent),
    )

    banked = loop.run("bare_acronym", n=1)
    assert banked.empty, "an incompatible parent must never be banked"


@pytest.mark.parametrize("text,expect", [
    ("cPGES lipoxinA4 alternative medicine", True),
    ("cPGES alternative medicine", False),
])
def test_the_count_target_is_what_decides(text, expect):
    """Two distinct code identifiers pass, one does not — the contradiction d54
    removed was asking for two while offering one."""
    targets = Targets.model_validate({
        "spans": [{"feature": "code_identifier", "min_count": 2}]
    })
    report = verify(text, targets, extractor=FeatureExtractor(engines=None))
    assert report.passed is expect


def test_spend_add_accumulates_across_outcomes():
    """A drop still paid for its hops — add() has no accepted/rejected
    branch, it just sums what the outcome actually cost."""
    spend = Spend()
    spend.add(AugmentationOutcome(
        text="a", accepted=True, attempts=1, hops=1, tokens=100, elapsed_s=1.0,
    ))
    spend.add(AugmentationOutcome(
        text=None, accepted=False, attempts=1, hops=2, tokens=250, elapsed_s=3.5,
    ))
    assert (spend.hops, spend.tokens, spend.elapsed_s) == (3, 350, 4.5)


def test_spend_stamp_credits_the_running_total_not_the_last_call():
    """A cell's second call (e.g. the cut) only knows its OWN cost; stamp()
    must overwrite it with everything spent across the whole sequence."""
    spend = Spend()
    spend.add(AugmentationOutcome(text="a", accepted=True, attempts=1, hops=1, tokens=100))
    last_call = AugmentationOutcome(
        text="b", accepted=True, attempts=1, hops=1, tokens=50, elapsed_s=0.2,
    )
    spend.add(last_call)
    stamped = spend.stamp(last_call)
    assert (stamped.hops, stamped.tokens) == (2, 150)
    assert stamped.text == "b"   # every other field survives untouched


def test_spend_summary_reports_the_per_row_rate():
    spend = Spend()
    spend.add(AugmentationOutcome(text="a", accepted=True, attempts=1, hops=4, tokens=1000))
    spend.add(AugmentationOutcome(text=None, accepted=False, attempts=1, hops=6, tokens=2000))
    summary = spend.summary(accepted=1)
    assert summary.startswith("10 hops, 3,000 tokens, ")
    assert "of it LLM)" in summary
    assert summary.endswith("-> 10.0 hops/row, 3,000 tokens/row, "
                             f"{spend.wall_s:.1f}s/row")


def test_spend_summary_handles_nothing_accepted():
    spend = Spend()
    spend.add(AugmentationOutcome(text=None, accepted=False, attempts=1, hops=3, tokens=90))
    summary = spend.summary(accepted=0)
    assert summary.startswith("3 hops, 90 tokens, ")
    assert summary.endswith("(nothing accepted)")


def test_spend_wall_s_measures_from_construction_not_from_llm_time():
    """The whole point: `wall_s` ticks even when nothing is ever add()-ed —
    it is measuring the SCOPE's real time, not summing reported hop costs."""
    spend = Spend()
    time.sleep(0.02)
    assert spend.wall_s >= 0.02
    assert spend.elapsed_s == 0.0   # no hop was ever reported


def test_spend_report_includes_wall_s_but_stamp_input_does_not():
    """`report()` is for a floor-level readout; `as_dict()` (what `stamp()`
    uses) must stay outcome-compatible — `AugmentationOutcome` has no
    `wall_s` field, so leaking it into stamp() would crash the model_copy."""
    spend = Spend()
    assert "wall_s" in spend.report()
    assert "wall_s" not in spend.as_dict()
