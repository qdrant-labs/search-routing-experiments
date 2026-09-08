from types import SimpleNamespace

import pandas as pd
import pytest

from scripts.apply_os_distill_sparse import _pilot, _plan, _positive_int


class _Client:
    """Reports which collections exist; that is all `dense_source`/`_plan` ask."""

    def __init__(self, live: set[str]) -> None:
        self.live = live

    def collection_exists(self, name: str) -> bool:
        return name in self.live

    def get_collections(self):
        return SimpleNamespace(collections=[SimpleNamespace(name=n) for n in self.live])


@pytest.fixture(autouse=True)
def _openrouter_key(monkeypatch):
    monkeypatch.setenv("OPEN_ROUTER_API_KEY", "sk-test")


def _cfg(tmp_path):
    labels = pd.DataFrame({
        "dataset": ["scirgen-geo-en", "scirgen-geo-en", "antique", "antique", "quest"],
        "shape": ["all_zero", "routes_differ", "all_tied", "all_zero", "all_zero"],
    })
    labels.to_parquet(tmp_path / "labels.parquet")
    return SimpleNamespace(labels=tmp_path / "labels.parquet")


def test_plan_puts_dense_reusing_lanes_first(tmp_path):
    """The gemini-hosted lanes are free (dense copied) and must lead, so an
    abort banks them before any paid lane runs."""
    gemini = "scirgen-geo-en_legb_gemini-embedding-001_routes"
    pilot = _pilot(_Client(live={gemini}), ("scirgen-geo-en", "antique", "quest"))
    plan = _plan(_cfg(tmp_path), pilot)
    assert list(plan["lane"])[0] == "scirgen-geo-en"
    assert bool(plan.iloc[0]["reuses_dense"]) is True
    assert set(plan[~plan["reuses_dense"]]["lane"]) == {"antique", "quest"}


def test_plan_counts_only_unreached_rows(tmp_path):
    """A lane's weight is its all_zero + all_tied rows — the ones the later
    relabel exists to move — not its whole label count."""
    pilot = _pilot(_Client(live=set()), ("scirgen-geo-en", "antique", "quest"))
    plan = _plan(_cfg(tmp_path), pilot).set_index("lane")
    assert plan.loc["scirgen-geo-en", "unreached"] == 1  # one all_zero; routes_differ ignored
    assert plan.loc["antique", "unreached"] == 2         # all_tied + all_zero


def test_index_refuses_a_paid_lane_without_authorisation(tmp_path):
    """A lane with no gemini vectors would embed dense through the provider;
    the driver drives LegBPilot.index, whose guard must hold that back."""
    pilot = _pilot(_Client(live=set()), ("antique",))
    with pytest.raises(RuntimeError, match="allow_paid_dense"):
        pilot.index("antique")


def test_pilot_passes_the_cli_sparse_batch_size_to_the_distill_encoder():
    pilot = _pilot(_Client(live=set()), ("antique",), sparse_batch_size=8)
    assert pilot._sparse.sentence_transformers_batch_size == 8


@pytest.mark.parametrize("value", ["0", "-1"])
def test_sparse_batch_size_must_be_positive(value):
    with pytest.raises(Exception, match="at least 1"):
        _positive_int(value)
