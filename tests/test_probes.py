from hybrid_search_rrf_dataset.fusion import StrategyName
from hybrid_search_rrf_dataset.probes import ARCHETYPE_PROBES, probe


class _Fixed:
    """A router that serves one route regardless of the query."""

    def __init__(self, route: StrategyName) -> None:
        self.route = route

    def predict(self, query: str) -> StrategyName:
        del query
        return self.route


def test_probe_scores_every_case():
    frame = probe(_Fixed(StrategyName.SPARSE_ONLY))
    assert len(frame) == len(ARCHETYPE_PROBES)
    assert frame["agrees"].eq(True).sum() == 3  # the three sparse probes
    assert frame["agrees"].isna().sum() == 1  # HTTP 502 has no expected route


def test_probes_are_well_formed():
    for case in ARCHETYPE_PROBES:
        assert case.expected is None or isinstance(case.expected, StrategyName)
        assert case.query and case.why
