"""The coherence gate's LLM judge with the model stubbed: verdicts banked
beside the pool, and the two consumers that read them — admission and the
campaign's pilot clamp. Nothing here calls a model or a network.
"""

import pandas as pd

from augmentation.campaign import PRODUCE, SKIP_STAGED, AugmentationCampaign
from augmentation.config import AugmentationConfig, AugmentationPaths
from augmentation.constructed import ConstructedDocs
from augmentation.core import AnswerKeyPath, AugmentedCandidate, CreditGate
from augmentation.engine import Spend
from augmentation.judge import CoherenceJudge
from augmentation.loop import AugmentationLoop
from augmentation.operators import StatRewrite
from augmentation.pool import GeneratedPool
from augmentation.qrels import AugmentationQrels
from composition.composer import V3Composition
from tests.test_augmentation_loop import OneParent, PassThroughInject, _sheet

CELL = "version_pinned_technical"
DOC = "nginx 2.1.3 changed the default worker_connections to 1024."
YES = "yes - the passage answers the query"
NO = "no - the document never mentions the version"


class ScriptedJudge:
    """One canned reply per call, in order — `ask` is the judge's only model
    contact, so an exhausted script proves no extra call happened."""

    model = "stub-judge"

    def __init__(self, *replies: str) -> None:
        self.replies = list(replies)
        self.prompts: list[str] = []

    def ask(self, instruction: str, prompt: str) -> tuple[str, Spend]:
        self.prompts.append(prompt)
        return self.replies.pop(0), Spend()


def _candidate(
    query_id: str,
    *,
    floor: str = CELL,
    gate: CreditGate = CreditGate.COHERENCE_GATE,
    doc: str | None = "DOC-1",
) -> AugmentedCandidate:
    return AugmentedCandidate(
        query_id=query_id, query="nginx 2.1.3 worker_connections", floor=floor,
        operator="inject", provenance="doc_grounded", generated_from="q1",
        parent_dataset="beir-nfcorpus", home_lane="beir-nfcorpus",
        meaning_preserved=False, answer_key=AnswerKeyPath.MINTED, attempts=1,
        grounding_doc_id=doc, credit_gate=str(gate),
    )


def _pool(*candidates: AugmentedCandidate) -> pd.DataFrame:
    return pd.DataFrame([c.model_dump() for c in candidates])


def _judge(tmp_path, engine, *, docs_for=(), parents=None) -> CoherenceJudge:
    paths = AugmentationPaths(data_dir=tmp_path)
    docs = ConstructedDocs(paths)
    for query_id in docs_for:
        docs.add(query_id=query_id, source_dataset="beir-nfcorpus", text=DOC)
    return CoherenceJudge(
        engine,
        config=AugmentationConfig(paths=paths, llm_workers=1),
        docs=docs,
        parents=parents,
    )


def test_the_judge_banks_one_verdict_per_gated_row(tmp_path):
    engine = ScriptedJudge(YES)
    judge = _judge(tmp_path, engine, docs_for=["syn-1"])

    counts = judge.run(_pool(
        _candidate("syn-1"), _candidate("syn-2"), _candidate("free", gate=CreditGate.NONE),
    ))

    assert counts == {
        "candidates": 2, "skipped": 0, "judged": 1, "passed": 1, "failed": 0,
        "unparsed": 0, "no_evidence": 1,
    }
    banked = judge.load()
    assert list(banked["query_id"]) == ["syn-1"]
    assert bool(banked.iloc[0]["verdict"]) is True
    assert banked.iloc[0]["reason"] == "the passage answers the query"
    assert banked.iloc[0]["floor"] == CELL
    assert banked.iloc[0]["model"] == "stub-judge"
    assert judge.passed() == {"syn-1"}
    assert DOC in engine.prompts[0]


def test_a_rerun_judges_nothing_twice(tmp_path):
    """Idempotent on query_id: the second run's engine has no reply left, so
    any call at all raises."""
    judge = _judge(tmp_path, ScriptedJudge(YES), docs_for=["syn-1"])
    pool = _pool(_candidate("syn-1"))
    judge.run(pool)

    again = _judge(tmp_path, ScriptedJudge(), docs_for=["syn-1"])
    counts = again.run(pool)

    assert counts["skipped"] == 1 and counts["judged"] == 0
    assert len(again.load()) == 1


def test_an_unparseable_reply_leaves_the_row_unjudged(tmp_path):
    """Never a silent pass: the row stays gated and the next run retries it."""
    judge = _judge(tmp_path, ScriptedJudge("hard to say, maybe?"), docs_for=["syn-1"])
    pool = _pool(_candidate("syn-1"))

    counts = judge.run(pool)

    assert counts["unparsed"] == 1 and counts["judged"] == 0
    assert judge.load().empty and judge.passed() == set()

    retried = _judge(tmp_path, ScriptedJudge(NO), docs_for=["syn-1"])
    assert retried.run(pool)["failed"] == 1
    assert retried.passed() == set()


def test_a_grounded_row_is_judged_against_its_lane_document(tmp_path):
    """Inject's evidence is the document its surfaces were copied from, not a
    constructed one."""
    judge = _judge(tmp_path, ScriptedJudge(YES), parents=OneParent({}, gold=DOC))

    counts = judge.run(_pool(_candidate("aug-1")))

    assert counts["judged"] == 1 and counts["no_evidence"] == 0
    assert judge.passed() == {"aug-1"}


def test_only_a_passed_coherence_row_becomes_admissible(tmp_path):
    """Admission's gate seam: a cleared coherence row is admissible, a failed
    or unjudged one is not, and declaration_audit never opens this way."""
    pool = _pool(
        _candidate("free", gate=CreditGate.NONE),
        _candidate("passed"),
        _candidate("failed"),
        _candidate("audited", gate=CreditGate.DECLARATION_AUDIT),
    )
    selection = pd.DataFrame({"query_id": []})
    sheet = pd.DataFrame([{"floor": CELL, "missing": 5.0}])

    today = V3Composition._admissible(pool, selection, sheet)
    assert set(today["query_id"]) == {"free"}

    with_judge = V3Composition._admissible(
        pool, selection, sheet, {"passed", "audited"}
    )
    assert set(with_judge["query_id"]) == {"free", "passed"}


def _campaign_loop(tmp_path, staged: list[AugmentedCandidate]) -> AugmentationLoop:
    paths = AugmentationPaths(data_dir=tmp_path)
    paths.catalog.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{
        "dataset": "beir-nfcorpus", "query_id": "q1", "checkable": True,
        "length.length_words": 2.0,
    }]).to_parquet(paths.catalog, index=False)
    config = AugmentationConfig(paths=paths, llm_workers=1)
    sheet_path = tmp_path / "sheet.parquet"
    _sheet(CELL).to_parquet(sheet_path, index=False)
    pool = GeneratedPool(paths)
    pool.append(staged)
    return AugmentationLoop(
        pd.DataFrame(columns=["dataset", "query_id", "query", "checkable"]),
        config=config,
        operators=(PassThroughInject(config), StatRewrite(config)),
        sheet_path=sheet_path,
        pool=pool,
        qrels=AugmentationQrels(paths),
        parents=OneParent({
            "dataset": "beir-nfcorpus", "query_id": "q1", "checkable": True,
            "query": "nginx config", "floors": [],
            "surfaces": ("2.1.3",), "bank": "version_string",
            "grounding_doc_id": "DOC-1", "length.length_words": 2.0,
        }),
    )


def _plan(tmp_path, *replies: str, with_judge: bool = True) -> pd.Series:
    staged = [_candidate("syn-1"), _candidate("syn-2")]
    loop = _campaign_loop(tmp_path, staged)
    judge = _judge(
        tmp_path, ScriptedJudge(*replies), docs_for=["syn-1", "syn-2"]
    )
    judge.run(_pool(*staged))
    campaign = AugmentationCampaign(
        loop, pilot_n=2, judge=judge if with_judge else None
    )
    return campaign.plan().iloc[0]


def test_a_pilot_that_passes_the_dial_unclamps_its_floor(tmp_path):
    """Both staged rows judged and passed = pass_rate 1.0 >= 0.9, so the floor
    plans its full missing credit instead of the pilot cap."""
    row = _plan(tmp_path, YES, YES)

    assert row["gate"] == str(CreditGate.COHERENCE_GATE)
    assert row["gate_state"] == "open"
    assert row["action"] == PRODUCE
    assert row["target_rows"] + row["synthetic_rows"] == 5


def test_a_pilot_under_the_dial_stays_clamped(tmp_path):
    """One pass in two is 0.5 — under the 0.9 dial, so the pilot cap holds and
    the staged floor produces nothing more."""
    row = _plan(tmp_path, YES, NO)

    assert row["gate_state"] == "held"
    assert row["action"] == SKIP_STAGED
    assert row["target_rows"] == 0


def test_without_the_judge_a_passed_pilot_still_clamps(tmp_path):
    """The default: verdicts exist on disk, but a campaign built without the
    judge behaves exactly as it did before it — the human audit still owns the
    gate."""
    row = _plan(tmp_path, YES, YES, with_judge=False)

    assert row["gate_state"] == "held"
    assert row["action"] == SKIP_STAGED
    assert row["target_rows"] == 0
