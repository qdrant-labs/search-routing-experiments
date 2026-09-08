"""Offline invariants for the dataset registry: enum/instance bijection,
card validity, parquet cache behavior, index-based sampling. Network loads
are exercised in notebooks, not here.
"""

from collections.abc import Iterator
from pathlib import Path

import ir_datasets
import pytest

from dataset_registry import (
    DATASETS,
    Availability,
    DatasetCard,
    DatasetName,
    DatasetRegistry,
    Grounding,
    IRDatasetsBacked,
    MiraclDev,
    Query,
    QueryProvenance,
    RegistryDataset,
    Scope,
    SourceKind,
)


class FakeDataset(RegistryDataset):
    """In-memory dataset for cache/sampling/facade tests; borrows a real name."""

    def __init__(self, queries: list[str], cache_dir: Path) -> None:
        super().__init__(cache_dir)
        self._queries = queries
        self.fetch_calls = 0

    @property
    def card(self) -> DatasetCard:
        return DatasetCard(
            name=DatasetName.MSMARCO_PASSAGE_DEV,
            source=SourceKind.URL,
            grounding=Grounding.QO,
            llm_target=False,
            query_provenance=QueryProvenance.HUMAN,
            scope=Scope.GENERAL,
            non_trivial=False,
            multilingual=False,
            multimodal=False,
            availability=Availability.OPEN,
            homepage="https://example.com",
        )

    def _fetch_queries(self) -> Iterator[Query]:
        self.fetch_calls += 1
        for index, text in enumerate(self._queries):
            yield Query(str(index), text)


class FailingDataset(FakeDataset):
    """Dies mid-stream after yielding a few queries."""

    def _fetch_queries(self) -> Iterator[Query]:
        yield from list(super()._fetch_queries())[:2]
        raise ConnectionError("stream died")


def test_registered_names_are_bijective_with_enum():
    names = [dataset.card.name for dataset in DATASETS]
    assert len(names) == len(set(names)), "duplicate dataset names"
    assert set(names) == set(DatasetName), "enum and DATASETS out of sync"


def test_every_card_validates():
    for dataset in DATASETS:
        assert isinstance(dataset.card, DatasetCard)


def test_query_provenance_is_declared_and_pins_the_machine_written_sources():
    """The two sources whose queries no human wrote — a showcase that reports
    them as real user queries is lying about 73% of its natural supply."""
    by_name = {dataset.card.name: dataset.card for dataset in DATASETS}
    for card in by_name.values():
        assert isinstance(card.query_provenance, QueryProvenance)
    assert by_name[DatasetName.SCIRGEN_GEO_EN].query_provenance is QueryProvenance.LLM
    assert by_name[DatasetName.LIMIT].query_provenance is QueryProvenance.TEMPLATE


def test_irds_ids_resolve_in_catalog():
    for dataset in DATASETS:
        if isinstance(dataset, IRDatasetsBacked):
            loaded = ir_datasets.load(dataset.irds_id)
            assert loaded.has_queries()


def test_miracl_unknown_language_rejected(tmp_path):
    with pytest.raises(ValueError):
        MiraclDev("xx", cache_dir=tmp_path)


def test_miracl_unregistered_language_fails_card_validation(tmp_path):
    # 'fr' is a valid MIRACL language but has no DatasetName member yet;
    # the enum lookup in the card raises before pydantic even sees it
    with pytest.raises(ValueError, match="miracl-fr-dev"):
        _ = MiraclDev("fr", cache_dir=tmp_path).card


def test_registry_rejects_duplicate_names(tmp_path):
    fake = FakeDataset(["a"], tmp_path)
    with pytest.raises(ValueError, match="duplicate"):
        DatasetRegistry(datasets=[fake, fake])


def test_source_is_fetched_exactly_once(tmp_path):
    fake = FakeDataset([f"query {i}" for i in range(10)], tmp_path)

    first = list(fake.load_queries())
    second = list(fake.load_queries())
    fake.sample_queries(3)

    assert fake.fetch_calls == 1, "cache must absorb repeated loads"
    assert first == second
    assert fake.cache_path.exists()


def test_crashed_fetch_leaves_no_cache(tmp_path):
    failing = FailingDataset([f"query {i}" for i in range(10)], tmp_path)

    with pytest.raises(ConnectionError):
        list(failing.load_queries())

    assert not failing.cache_path.exists(), "truncated cache must not survive"
    assert list(tmp_path.iterdir()) == [], "tmp file must be cleaned up"

    # a later retry starts clean and succeeds
    fake = FakeDataset(["a", "b"], tmp_path)
    assert len(list(fake.load_queries())) == 2


def test_sample_is_seeded_and_index_based(tmp_path):
    fake = FakeDataset([f"query {i}" for i in range(100)], tmp_path)
    first = fake.sample_queries(10, seed=42)
    second = fake.sample_queries(10, seed=42)
    other_seed = fake.sample_queries(10, seed=7)

    assert first == second, "same seed must reproduce the sample"
    assert first != other_seed, "different seed should differ on 100 items"
    assert len(first) == 10
    assert len({q.query_id for q in first}) == 10, "sample must not repeat items"
    ids = [int(q.query_id) for q in first]
    assert ids == sorted(ids), "sample keeps source order"
    assert fake.fetch_calls == 1, "sampling must run against the cache"


def test_sample_spans_multiple_record_batches(tmp_path):
    import random

    from dataset_registry.core import _WRITE_BATCH_SIZE

    n_rows = _WRITE_BATCH_SIZE + 1_000  # cache gets 2 row groups
    fake = FakeDataset([f"query {i}" for i in range(n_rows)], tmp_path)

    sample = fake.sample_queries(50, seed=3)

    expected = sorted(random.Random(3).sample(range(n_rows), 50))
    assert [int(q.query_id) for q in sample] == expected
    assert [q.text for q in sample] == [f"query {i}" for i in expected]
    assert any(i >= _WRITE_BATCH_SIZE for i in expected), (
        "seed must draw from the second batch for this test to mean anything"
    )


def test_sample_smaller_population_returns_all(tmp_path):
    fake = FakeDataset(["a", "b", "c"], tmp_path)
    assert len(fake.sample_queries(10)) == 3
    assert len(fake.sample_queries(None)) == 3


def test_profile_runs_taxonomy_over_sample(tmp_path):
    fake = FakeDataset(
        [
            "upgrade qdrant to v1.9.2",
            "what is CVE-2024-3094",
            "plain question with no identifiers at all",
        ],
        tmp_path,
    )
    registry = DatasetRegistry(datasets=[fake])
    corpus_ids = registry.profile(DatasetName.MSMARCO_PASSAGE_DEV)

    assert len(corpus_ids.queries) == 3
    tagged = [q for q in corpus_ids.queries if q.spans]
    # all three: two carry identifiers, the third's "no" is a negation marker
    assert len(tagged) == 3
