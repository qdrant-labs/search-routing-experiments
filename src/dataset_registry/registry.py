from collections.abc import Iterable, Iterator

from query_taxonomy.features import CorpusIdentifierExtractor, CorpusIdentifiers
from tqdm.auto import tqdm

from dataset_registry.core import (
    DatasetCard,
    DatasetName,
    Query,
    RegistryDataset,
)
from dataset_registry.hf import MiraclDev
from dataset_registry.irds import BeirNFCorpus, MSMarcoPassageDev, TrecDL2022

# Configured instances, not classes: parameterized datasets (MiraclDev per
# language) register several entries from one class.
DATASETS: tuple[RegistryDataset, ...] = (
    MSMarcoPassageDev(),
    TrecDL2022(),
    BeirNFCorpus(),
    MiraclDev("en"),
)


class DatasetRegistry:
    """
    Facade over the registered datasets: lookup by name, seeded sampling,
    and taxonomy profiling of a sample in one call.
    """

    def __init__(
        self,
        datasets: Iterable[RegistryDataset] = DATASETS,
        extractor: CorpusIdentifierExtractor | None = None,
    ) -> None:
        self._extractor = extractor or CorpusIdentifierExtractor()
        self._datasets: dict[DatasetName, RegistryDataset] = {}
        for dataset in datasets:
            name = dataset.card.name
            if name in self._datasets:
                raise ValueError(f"duplicate dataset name: {name}")
            self._datasets[name] = dataset

    def __getitem__(self, name: DatasetName) -> RegistryDataset:
        return self._datasets[name]

    def __iter__(self) -> Iterator[RegistryDataset]:
        return iter(self._datasets.values())

    def __len__(self) -> int:
        return len(self._datasets)

    def cards(self) -> tuple[DatasetCard, ...]:
        return tuple(dataset.card for dataset in self._datasets.values())

    def sample(
        self, name: DatasetName, n: int | None = None, *, seed: int = 0
    ) -> list[Query]:
        return self[name].sample_queries(n, seed=seed)

    def profile(
        self, name: DatasetName, n: int | None = None, *, seed: int = 0
    ) -> CorpusIdentifiers:
        """Run a query sample through the identifier banks — the statistical
        approximation used to decide which datasets are worth acquiring fully."""
        texts = [query.text for query in self.sample(name, n, seed=seed)]
        # extract() consumes the sequence in a single enumerate pass, so a
        # tqdm wrapper tracks the actual regex work query by query
        progress = tqdm(
            texts,
            desc=f"profile {name.value} (n={len(texts)}, seed={seed})",
            unit="query",
        )
        return self._extractor.extract(progress)
