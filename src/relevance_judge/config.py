"""Input surface for the relevance-atom judge: paths, engine, gate dials."""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from augmentation.config import EngineSettings

_DATA = Path(__file__).resolve().parent.parent / "data"


def _judge_engine() -> EngineSettings:
    # ponytail: luna prices are the one calibration knob — the doc's ~$0.0003/call
    # is the anchor; set usd_per_mtok_* from luna's real card when it publishes.
    return EngineSettings(
        model="openrouter/openai/gpt-5.6-luna",
        usd_per_mtok_in=0.40,
        usd_per_mtok_out=1.60,
        max_spend_usd=10.0,
    )


class RelevanceJudgeConfig(BaseModel):
    """Every knob the judge, queue, harness and scorer read."""

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    data_dir: Path = _DATA
    engine: EngineSettings = Field(default_factory=_judge_engine)
    llm_workers: int = 8

    reasoning_effort: str | None = "none"
    """OpenRouter reasoning effort, sent as `reasoning={"effort": ...}` (the payload
    llm_a_judge proved works on luna). The output contract is verdict-FIRST, so a
    leaked reasoning preamble fails the parser and silently drops the pair — a
    binary yes/no judge needs no reasoning, so default 'none' suppresses it. Set to
    'low'/'medium'/'high' to re-enable, or None to omit the param entirely."""

    min_relevance: int = 1
    """Binary bar the objective already consumes — a graded scale would redefine
    every existing score (doc §5a)."""

    excluded_referee_lanes: tuple[str, ...] = ("trec-dl-2022",)
    """Lanes barred from the validation referee set despite having negatives:
    trec-dl-2022's corpus↔qrels doc_id mapping is broken (it grades unrelated docs
    — e.g. a crossword page for 'who is gehan homes' — as highly relevant), so it
    cannot referee the judge. A dataset bug to fix upstream, not a judge failure."""

    min_precision_relevant: float = 0.95
    """PRIMARY hard gate: of pairs the judge calls relevant, the share humans agree
    on. A false positive injects wrong gold that corrupts every downstream score —
    an asymmetric risk a missed hole does not carry."""
    anchor_lanes: tuple[str, ...] = ("rarb-math", "miracl-en-dev", "beir-touche-2020")
    """Answer-oriented lanes where the dataset's 'relevant' means 'answers the
    query' — the same definition the judge applies. The non-degeneracy recall
    floor is measured HERE, so it catches an always-'no' judge without penalizing
    the strict-vs-topical definition gap on retrieval-style lanes (crumb/freshstack)."""
    min_anchor_recall: float = 0.6
    """SECOND hard gate: recall on `anchor_lanes` must clear this — a degenerate
    always-'no' judge scores ~0 even where it shares the definition."""
    min_agreement_overall: float = 0.90
    min_agreement_lane: float = 0.80
    """DIAGNOSTIC only, not enforced: a strict-but-precise judge disagrees with
    datasets that grade topical/partial matches relevant, which is safe for gold
    injection. Reported, never gated (precision + recall floor are the gate)."""
    min_tie_conversion: float = 0.10
    """§3a Phase-2 gate: minimum tie -> decisive/low-margin conversion before any
    dataset-wide spend."""

    @property
    def artifacts(self) -> Path:
        return self.data_dir / "relevance_judge"

    @property
    def judged_qrels(self) -> Path:
        return self.artifacts / "judged_qrels.parquet"

    @property
    def judge_runs(self) -> Path:
        return self.artifacts / "judge_runs.parquet"

    @property
    def validation_report(self) -> Path:
        return self.artifacts / "validation_report.json"

    @property
    def validation_predictions(self) -> Path:
        return self.artifacts / "validation_predictions.parquet"

    @property
    def false_positives_audit(self) -> Path:
        return self.artifacts / "false_positives_audit.parquet"

    @property
    def sample_test_report(self) -> Path:
        return self.artifacts / "sample_test_report.json"

    @property
    def arch5k(self) -> Path:
        return self.data_dir / "legb_pilot" / "arch5k"

    @property
    def v2_100k(self) -> Path:
        return self.data_dir / "rungs" / "100k-v2"
