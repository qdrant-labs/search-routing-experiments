"""Provider blips are failed attempts, never crashes, and a forced rebuild
sees labels that landed after the pool was first read — the two failure
modes of the 2026-08-20 crank."""

import pandas as pd
from litellm.exceptions import InternalServerError

from augmentation.engine import Augmenter, Spend
from composition.pool_v3 import LabelledPool
from taxonomy_generators.verify import Targets


def _overloaded(*args, **kwargs):
    raise InternalServerError(
        message="overloaded", llm_provider="anthropic", model="m"
    )


def test_run_returns_a_failed_attempt_on_provider_overload(monkeypatch):
    engine = Augmenter()
    monkeypatch.setattr(engine, "_single_shot", _overloaded)
    outcome = engine.run("instruction", "prompt", Targets())
    assert not outcome.accepted
    assert outcome.error == "provider_fault"


def test_ask_returns_no_reply_on_provider_overload(monkeypatch):
    engine = Augmenter()
    monkeypatch.setattr(engine, "_single_shot", _overloaded)
    text, spend = engine.ask("instruction", "prompt")
    assert text is None and isinstance(spend, Spend)


def test_refresh_makes_a_rebuild_see_new_labels(tmp_path):
    (tmp_path / "v3").mkdir()
    path = tmp_path / "v3" / "labels_rederived.parquet"
    pd.DataFrame({
        "dataset": ["a"], "query_id": ["q1"], "checkable": [True],
    }).to_parquet(path, index=False)
    pool = LabelledPool(data_dir=tmp_path)
    assert len(pool.labels()) == 1
    first = pool.labels
    # the cached frame is the composer-facing surface; labels() is uncached,
    # so pin the cache itself
    pool._frame = pd.DataFrame({"stale": [True]})
    pd.DataFrame({
        "dataset": ["a"], "query_id": ["q2"], "checkable": [True],
    }).to_parquet(tmp_path / "v3" / "labels.parquet", index=False)
    assert "stale" in pool.frame().columns, "cache still served"
    assert pool.refresh() is pool and pool._frame is None
    assert len(first()) == 2
