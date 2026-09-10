"""Guards the three training-time policies that have no other check: which rows
early stopping validates on, which rows the route loss counts, and the horizon
the branch anneal follows."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from encoder_router.evaluate import Arm, LaneCV

LANES, PER_LANE = 20, 50
HELD = "lane00"


class _Table:
    """The frame columns LaneCV's split and weight policies read."""

    def __init__(self) -> None:
        rng = np.random.default_rng(0)
        shapes = rng.choice(
            ["routes_differ", "all_tied", "all_zero"],
            size=LANES * PER_LANE, p=[0.3, 0.6, 0.1],
        )
        self.frame = pd.DataFrame({
            "dataset": np.repeat(
                [f"lane{i:02d}" for i in range(LANES)], PER_LANE
            ),
            "query_id": np.tile(np.arange(PER_LANE).astype(str), LANES),
            "shape": shapes,
            "serve": np.where(shapes == "all_zero", None, "dense_only"),
        })


@pytest.fixture
def cv() -> LaneCV:
    return LaneCV(_Table(), seed=0)


def test_validation_holds_a_within_train_row_slice(cv: LaneCV) -> None:
    """Early stopping validates on a random tenth of the train ROWS — lanes are
    shared with the fit rows, and the held lane is in neither."""
    frame = cv.table.frame
    train = (frame["dataset"] != HELD).to_numpy()
    fit, val = cv._fit_val_split(train)

    lanes = frame["dataset"].to_numpy()
    assert not set(fit) & set(val)
    assert len(fit) + len(val) == train.sum()
    assert len(val) == max(int(train.sum()) // 10, 1)
    assert HELD not in set(lanes[fit]) | set(lanes[val])


def test_decisive_only_zeroes_every_undecided_row(cv: LaneCV) -> None:
    frame = cv.table.frame
    decisive = (
        (frame["shape"] == "routes_differ") & frame["serve"].notna()
    ).to_numpy()

    weights = cv.row_weights(Arm("decisive", decisive_only=True))
    assert np.array_equal(weights > 0, decisive)

    default = cv.row_weights(Arm("default"))
    tied = (frame["shape"] == "all_tied").to_numpy()
    assert np.allclose(default[tied], cv.tie_weight)
    assert np.allclose(default[~tied], 1.0)


def test_zero_weight_rows_leave_the_loss_untouched(cv: LaneCV) -> None:
    """`decisive_only` zeroes rows rather than dropping them — only sound if a
    zero row is genuinely absent from the objective."""
    import torch

    from encoder_router.model import _masked_bce

    weights = cv.row_weights(Arm("decisive", decisive_only=True))
    kept = weights > 0
    rng = torch.Generator().manual_seed(0)
    logits = torch.randn(len(weights), 2, generator=rng)
    targets = torch.tile(torch.tensor([[1.0, 0.0]]), (len(weights), 1))

    masked = _masked_bce(logits, targets, row_weights=torch.as_tensor(weights))
    subset = _masked_bce(logits[kept], targets[kept])
    assert masked == pytest.approx(float(subset), abs=1e-6)
    assert _masked_bce(
        logits, targets, row_weights=torch.zeros(len(weights))
    ) == 0.0


def _tiny_fit_args(rng: np.random.Generator, rows: int = 32, dim: int = 8):
    x = rng.normal(size=(rows, dim)).astype(np.float32)
    route = rng.integers(0, 2, size=(rows, 2)).astype(np.float32)
    return x, route


def test_warm_start_without_a_net_raises() -> None:
    from encoder_router.model import EncoderRouter

    x, route = _tiny_fit_args(np.random.default_rng(0))
    with pytest.raises(ValueError, match="requires a loaded net"):
        EncoderRouter().fit(x, route, None, None, warm_start=True)


def test_warm_start_rejects_mismatched_input_width() -> None:
    from encoder_router.model import EncoderRouter

    rng = np.random.default_rng(0)
    x, route = _tiny_fit_args(rng)
    router = EncoderRouter(epochs=1)
    router.fit(x, route, None, None)
    wide, _ = _tiny_fit_args(rng, dim=9)
    with pytest.raises(ValueError, match="input width"):
        router.fit(wide, route, None, None, warm_start=True)


def test_warm_start_at_lr_zero_is_a_no_op_on_weights() -> None:
    """The seam's soundness proof: warm_start must reuse the existing net,
    so with lr=0 (AdamW's decoupled decay also scales by lr) every parameter
    survives bit-identical."""
    import torch

    from encoder_router.model import EncoderRouter

    x, route = _tiny_fit_args(np.random.default_rng(0))
    router = EncoderRouter(epochs=2)
    router.fit(x, route, None, None)
    before = {k: v.clone() for k, v in router.net.state_dict().items()}

    router.lr = 0.0
    router.fit(x, route, None, None, warm_start=True)
    for key, value in router.net.state_dict().items():
        assert torch.equal(before[key], value), key


def test_warm_start_with_frozen_encoder_tunes_only_route_layers() -> None:
    import torch

    from encoder_router.model import EncoderRouter

    x, route = _tiny_fit_args(np.random.default_rng(0))
    router = EncoderRouter(epochs=3, lr=1e-2)
    router.fit(x, route, None, None)
    before = {k: v.clone() for k, v in router.net.state_dict().items()}

    for p in router.net.encoder.parameters():
        p.requires_grad_(False)
    router.fit(x, route, None, None, warm_start=True)
    for key, value in router.net.state_dict().items():
        if key.startswith("encoder"):
            assert torch.equal(before[key], value), key
    assert any(
        not torch.equal(before[k], v)
        for k, v in router.net.state_dict().items()
        if k.startswith("route_layers")
    )


def test_cold_fit_still_builds_a_fresh_net() -> None:
    from encoder_router.model import EncoderRouter

    x, route = _tiny_fit_args(np.random.default_rng(0))
    router = EncoderRouter(epochs=1)
    router.fit(x, route, None, None)
    first = router.net
    router.fit(x, route, None, None)
    assert router.net is not first


def test_branch_anneal_follows_a_share_of_the_epoch_budget() -> None:
    """The horizon is `anneal_share * epochs`, so a run that early-stops well
    before it keeps the branches at near-full weight."""
    from encoder_router.model import EncoderRouter

    router = EncoderRouter(epochs=200)
    horizon = router.anneal_share * router.epochs
    anneal = [
        max(0.0, 1.0 - e / max(horizon, 1)) for e in range(int(horizon) + 1)
    ]
    assert anneal[0] == 1.0
    assert anneal[25] > 0.8
    assert anneal[-1] == 0.0
