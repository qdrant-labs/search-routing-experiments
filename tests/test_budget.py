"""The run's dollar ceiling: priced from real prompt/completion counts, and
refusing the call that would breach it rather than reporting the breach after
the money is gone."""

import pytest

from augmentation.engine import Augmenter, Budget, BudgetExceeded


def _budget(max_usd: float) -> Budget:
    return Budget(max_usd, usd_per_mtok_in=1.0, usd_per_mtok_out=5.0)


class _Usage:
    def __init__(self, prompt: int, completion: int) -> None:
        self.prompt_tokens = prompt
        self.completion_tokens = completion
        self.total_tokens = prompt + completion


class _Response:
    def __init__(self, prompt: int, completion: int) -> None:
        self.usage = _Usage(prompt, completion)


def test_output_is_priced_five_times_input():
    """A blended total would call these two calls equal; they are not."""
    cheap, dear = _budget(10.0), _budget(10.0)
    cheap.charge(1_000_000, 0)
    dear.charge(0, 1_000_000)
    assert cheap.spent_usd == pytest.approx(1.0)
    assert dear.spent_usd == pytest.approx(5.0)


def test_reserve_refuses_before_the_call_that_would_breach():
    budget = _budget(0.01)
    budget.charge(0, 1_800)  # $0.009 of the $0.01 ceiling
    with pytest.raises(BudgetExceeded, match="stopping before it runs"):
        budget.reserve(["x" * 300], max_tokens=1024)
    assert budget.calls == 1, "the refused call must not be billed"


def test_charge_is_the_backstop_when_the_estimate_was_low():
    budget = _budget(0.001)
    with pytest.raises(BudgetExceeded, match="over the"):
        budget.charge(0, 1_000_000)


def test_a_call_within_the_ceiling_passes_both_gates():
    budget = _budget(10.0)
    budget.reserve(["a short prompt"], max_tokens=1024)
    budget.charge(700, 120)
    assert budget.spent_usd == pytest.approx(700 / 1e6 + 120 * 5 / 1e6)
    assert budget.calls == 1


def test_no_budget_means_uncapped(monkeypatch):
    """Every caller predating budgets keeps its old behaviour."""
    engine = Augmenter.__new__(Augmenter)
    engine._budget = None
    engine.model = "test/model"
    monkeypatch.setattr(
        "augmentation.engine.completion", lambda **kw: _Response(9_000_000, 9_000_000)
    )
    _, _, prompt, answer = engine._completion(messages=[], max_tokens=8)
    assert (prompt, answer) == (9_000_000, 9_000_000)


def test_the_engine_charges_what_the_provider_reported(monkeypatch):
    budget = _budget(10.0)
    engine = Augmenter.__new__(Augmenter)
    engine._budget = budget
    engine.model = "test/model"
    monkeypatch.setattr(
        "augmentation.engine.completion", lambda **kw: _Response(1_000, 200)
    )
    engine._completion(messages=["hi"], max_tokens=1024)
    assert budget.spent_usd == pytest.approx(1_000 / 1e6 + 200 * 5 / 1e6)
    assert budget.calls == 1


def test_the_engine_never_calls_the_provider_once_the_ceiling_is_reached(monkeypatch):
    budget = _budget(0.001)
    budget.charge(0, 200)  # $0.001 — exactly at the ceiling
    called = []
    engine = Augmenter.__new__(Augmenter)
    engine._budget = budget
    engine.model = "test/model"
    monkeypatch.setattr(
        "augmentation.engine.completion",
        lambda **kw: called.append(kw) or _Response(1, 1),
    )
    with pytest.raises(BudgetExceeded):
        engine._completion(messages=["hi"], max_tokens=1024)
    assert called == [], "the provider must never be reached"
