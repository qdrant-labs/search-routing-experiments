"""Cost instrumentation on `Augmenter` (2026-08 follow-up): every
`completion()` round-trip reports elapsed time and tokens, and `run()` sums
them correctly across attempts/rounds. `completion()` itself is monkeypatched
— no network call anywhere in this file.
"""

from __future__ import annotations

from types import SimpleNamespace

from augmentation.engine import Augmenter, Spend
from taxonomy_generators.verify import Targets


def _message(content: str | None = "ok", tool_calls=None):
    return SimpleNamespace(
        content=content,
        tool_calls=tool_calls,
        model_dump=lambda: {"role": "assistant", "content": content},
    )


def _response(message, total_tokens: int | None = 42):
    usage = None if total_tokens is None else SimpleNamespace(total_tokens=total_tokens)
    return SimpleNamespace(choices=[SimpleNamespace(message=message)], usage=usage)


def test_completion_reports_elapsed_time_and_tokens(monkeypatch):
    monkeypatch.setattr(
        "augmentation.engine.completion",
        lambda **kwargs: _response(_message()),
    )
    _, elapsed, tokens = Augmenter._completion(model="x", messages=[])
    assert tokens == 42
    assert elapsed >= 0.0


def test_completion_tolerates_a_response_with_no_usage(monkeypatch):
    """Some providers/models omit `usage` entirely — must report 0, not crash."""
    monkeypatch.setattr(
        "augmentation.engine.completion",
        lambda **kwargs: _response(_message(), total_tokens=None),
    )
    _, _, tokens = Augmenter._completion(model="x", messages=[])
    assert tokens == 0


def test_run_single_shot_costs_exactly_one_hop(monkeypatch):
    """Single-shot mode (tool_loop=False): one completion() call is one hop,
    regardless of `max_attempts` — the empty Targets() accepts on the first
    reply, so there is only one to count."""
    monkeypatch.setattr(
        "augmentation.engine.completion",
        lambda **kwargs: _response(_message(content="rewritten query")),
    )
    outcome = Augmenter().run("instruction", "prompt", Targets())
    assert outcome.accepted is True
    assert outcome.hops == 1
    assert outcome.tokens == 42
    assert outcome.elapsed_s >= 0.0


def test_run_tool_loop_counts_one_hop_per_round(monkeypatch):
    """Tool-loop mode: a round with no tool call at all still costs a hop —
    here the model replies with plain text on round 1 (protocol failure,
    NO_TEXT is not raised since the reply is non-empty), so hops == 1."""
    monkeypatch.setattr(
        "augmentation.engine.completion",
        lambda **kwargs: _response(_message(content="plain reply, no tool call")),
    )
    outcome = Augmenter().run("instruction", "prompt", Targets(), tool_loop=True)
    assert outcome.hops == 1
    assert outcome.tokens == 42


def test_spend_add_hop_counts_one_hop_regardless_of_token_count():
    spend = Spend()
    spend.add_hop(1.5, 500)
    spend.add_hop(0.5, 0)
    assert (spend.hops, spend.tokens, spend.elapsed_s) == (2, 500, 2.0)


def test_spend_add_merges_another_spend_or_an_outcome_alike():
    """Both an outcome and a Spend expose hops/tokens/elapsed_s — add() must
    not care which one it's handed."""
    from augmentation.engine import AugmentationOutcome

    total = Spend()
    round_spend = Spend()
    round_spend.add_hop(1.0, 10)
    total.add(round_spend)
    total.add(AugmentationOutcome(text="x", accepted=True, attempts=1, hops=1, tokens=20))
    assert (total.hops, total.tokens, total.elapsed_s) == (2, 30, 1.0)
