"""Input surface for the relevance-atom judge: paths, engine, gate dials."""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from augmentation.config import EngineSettings
from hybrid_search_rrf_dataset.paths import LanePaths

_DATA = Path(__file__).resolve().parent.parent / "data"


def _judge_engine() -> EngineSettings:
    # Rates from OpenRouter (2026-09-08). deepseek-chat-v3.1 chosen after the
    # v4-flash family probes showed high per-call latency (~1.3-2.2s even
    # reasoning-off) — v3.1 is the standard V3 chat model with reasoning
    # controllable via extra_body={reasoning: {enabled: False}}. Cost per pair
    # slightly HIGHER than luna ($0.00035 vs $0.00029) but wall-time is the
    # binding constraint; v3 tier probed at 2.0s/call, better format compliance
    # expected. Precision to be re-validated by the gate before any pilot spend.
    return EngineSettings(
        model="openrouter/deepseek/deepseek-chat-v3.1",
        usd_per_mtok_in=0.25,
        usd_per_mtok_out=0.95,
        max_spend_usd=10.0,
    )


class RelevanceJudgeConfig(BaseModel):
    """Every knob the judge, queue, harness and scorer read."""

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    data_dir: Path = _DATA
    engine: EngineSettings = Field(default_factory=_judge_engine)
    llm_workers: int = 12
    """Concurrent calls in flight. 32 provoked a steady stream of provider errors
    (throughput swinging 7-26 pair/s with hundreds of drops), and a dropped pair
    is PAID work discarded — so concurrency past the provider's tolerance costs
    money rather than saving time. 12 keeps a 3,600-pair gate near 6 min at the
    measured ~1.2s/call. The progress bar shows `drop=N` live: if it climbs,
    come down further; if it stays 0, this can be raised again."""

    max_answer_tokens: int = 256
    """Completion cap. Was 128, which fit the two-line reply only if the model
    honoured "at most 20 words" — it does not, and an over-long ASKED line
    truncated the VERDICT line away, losing 249 PAID pairs (5.2%) as unparseable.
    Output is billed per token actually produced, so a bigger cap costs nothing
    on the replies that stay short."""

    unreadable_samples: int = 5
    """How many raw unparseable replies to keep for diagnosis. They used to be
    discarded, which is why a format regression looked like a provider blip."""

    connect_retries: int = 3
    """Retries for a TRANSPORT failure (refused/reset/disconnected). Such a call
    never reached the model, so it cost nothing and the pair is pure lost work —
    one run lost 2,115 of 3,600 this way. Provider rejections of the request
    itself are NOT retried; they would fail identically and only waste time."""

    connect_backoff_s: float = 1.0
    """Linear backoff between connection retries (1s, 2s, 3s). Linear, not
    exponential: these faults arrive in bursts under load and clear in seconds."""

    request_timeout_s: float = 120.0
    """Per-call ceiling, set on litellm's MODULE-level `request_timeout` (see
    judge.py): the per-call `timeout=` kwarg does not override it on the
    OpenRouter path — measured, a `timeout=30` call still ran 60.9s against the
    6000s default. 120s, not 30s: luna answers a yes/no in ~60s, so a tight
    ceiling would time out work that was about to succeed. A timeout raises into
    TRANSIENT_PROVIDER_ERRORS, counting the pair unreadable and moving on."""

    reasoning_effort: str | None = "none"
    """Reasoning effort, sent BOTH as the top-level `reasoning_effort` param and
    as `extra_body={"reasoning": {"effort": ...}}` — the nested form alone left
    latency at ~60s/call, the top-level one drops it to ~1.2s. The precision cost
    is real and measured: 'none' scores 0.932 where full reasoning scored 0.955,
    the gap concentrated in code-documentation lanes. 'low'/'medium'/'high' are
    the untested middle; None omits the param and restores full reasoning."""

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

    oracle_stages: tuple[str, ...] = ("natural", "supplemented")
    """Labelling stages searched for a lane's persisted `route_rankings`,
    precedence first — data, not a hardcoded candidate list."""

    # ---- inputs the judge reads ----------------------------------------
    @property
    def lanes(self) -> LanePaths:
        """The shared per-lane layout on this config's root — one `data_dir`, so
        a `data_dir=tmp_path` config redirects lane reads too."""
        return LanePaths(data_dir=self.data_dir)

    @property
    def arch5k(self) -> Path:
        return self.data_dir / "legb_pilot" / "arch5k"

    @property
    def draw(self) -> Path:
        """The row population and its l2 score triple — the only artifact that
        carries l2, whose ranked lists were never persisted."""
        return self.arch5k / "rows.json"

    @property
    def depth_probe(self) -> Path:
        return self.arch5k / "depth_probe.parquet"

    @property
    def v2_100k(self) -> Path:
        return self.data_dir / "rungs" / "100k-v2"

    @property
    def labels(self) -> Path:
        return self.v2_100k / "labeling" / "labels.parquet"

    @property
    def manifest(self) -> Path:
        return self.v2_100k / "candidate_manifests.parquet"

    def oracle_caches(self, dataset: str) -> list[Path]:
        """Caches that may hold a lane's persisted retrieval results, precedence
        first: this rung's labelling stages, then the standing cache."""
        lanes = self.lanes
        rung = [
            lanes.oracle_rows(dataset, under=self.v2_100k / "labeling" / stage)
            for stage in self.oracle_stages
        ]
        return [*rung, lanes.oracle_rows(dataset)]

    # ---- artifacts the judge writes ------------------------------------
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
