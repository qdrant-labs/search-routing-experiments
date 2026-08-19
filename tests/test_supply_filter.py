"""Mined surfaces outlive the bank that claimed them: a bank repair leaves
`surfaces.parquet` full of spans nothing claims any more, and eligibility
selects them anyway. The load-time filter drops what can no longer serve the
target it gets selected for.
"""

import pandas as pd
import pytest

from augmentation.config import AugmentationPaths
from augmentation.supply import SupplyIndex

# version_string was repaired after the corpora were mined, so its supply is
# the live example: `p = 0.05`-shaped decimals it no longer claims
STALE = "3.25"
GENUINE = "1.10.2"
COLLATERAL = "0.7.0-2.el7.noarch.rpm"


@pytest.fixture(scope="module")
def index() -> SupplyIndex:
    return SupplyIndex(AugmentationPaths())


def test_a_surface_its_bank_no_longer_claims_is_dropped(index):
    assert not index._claims("version_string", STALE)


def test_a_genuine_surface_survives(index):
    assert index._claims("version_string", GENUINE)


def test_a_surface_that_drags_in_another_floor_is_dropped(index):
    """It IS a version string, but it also claims code_identifier and number —
    and gaining an unasked identifier floor is Inject's largest structural
    veto class, so it is cheaper never to select it."""
    assert not index._claims("version_string", COLLATERAL)


def _mined(*surfaces: str) -> pd.DataFrame:
    return pd.DataFrame([
        {"doc_id": f"d{i}", "bank": "version_string",
         "floor": "id:tech", "surface": s}
        for i, s in enumerate(surfaces)
    ])


def test_the_filter_keeps_the_serviceable_rows_and_reports(index, capsys):
    kept = index.claimable(_mined(STALE, GENUINE, COLLATERAL), "a-lane")
    assert list(kept["surface"]) == [GENUINE]
    out = capsys.readouterr().out
    assert "dropped 2/3" in out
    assert "version_string" in out, "the rotten bank must be named, not just counted"


def test_an_all_good_frame_is_returned_untouched_and_silent(index, capsys):
    mined = _mined(GENUINE, "2.0.0")
    kept = index.claimable(mined, "a-lane")
    assert len(kept) == len(mined)
    assert capsys.readouterr().out == "", "no drop, no noise"


def test_verdicts_are_memoized_per_surface_not_per_frame(tmp_path):
    """`eligible()` re-slices per floor, so the memo has to key on the pair —
    keying on the frame both re-resolves and, worse, returns another frame's
    answer when two demands draw the same bank."""
    paths = AugmentationPaths(data_dir=tmp_path)
    lane_dir = tmp_path / "lane"
    lane_dir.mkdir()
    _mined(STALE, GENUINE).to_parquet(lane_dir / "surfaces.parquet", index=False)
    idx = SupplyIndex(paths)

    assert list(idx.claimable(idx.load("lane"), "lane")["surface"]) == [GENUINE]
    assert idx._verdicts == {
        ("version_string", STALE): False,
        ("version_string", GENUINE): True,
    }
    # a DIFFERENT slice of the same lane and bank must be filtered on its own
    # merits, not handed the previous frame back
    other = idx.claimable(_mined(GENUINE, "2.0.0"), "lane")
    assert len(other) == 2


def test_a_missing_lane_is_empty_not_an_error(tmp_path):
    idx = SupplyIndex(AugmentationPaths(data_dir=tmp_path))
    assert idx.load("nope").empty
