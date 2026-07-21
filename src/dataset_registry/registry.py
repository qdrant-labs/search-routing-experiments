from collections.abc import Iterable, Iterator

import pandas as pd
from query_taxonomy.features import FeatureExtractor, CorpusFeatures
from tqdm.auto import tqdm

from dataset_registry.core import (
    DatasetCard,
    DatasetName,
    Query,
    RegistryDataset,
)
from dataset_registry.hf import (
    CRUMB_TASKS,
    BrightSplit,
    CrumbTask,
    MiraclDev,
    Quest,
    RarbPool,
)
from dataset_registry.irds import (
    BeirNFCorpus,
    DBPediaEntity,
    MSMarcoPassageDev,
    Orcas,
    TrecDL2022,
)
from dataset_registry.url import Limit

# Configured instances, not classes: parameterized datasets (MiraclDev per
# language) register several entries from one class.
DATASETS: tuple[RegistryDataset, ...] = (
    MSMarcoPassageDev(),
    TrecDL2022(),
    BeirNFCorpus(),
    MiraclDev("en"),
    Orcas(),
    DBPediaEntity(),
    BrightSplit("leetcode"),
    BrightSplit("aops"),
    BrightSplit("theoremqa_questions"),
    Quest(),
    *(CrumbTask(task) for task in sorted(CRUMB_TASKS)),
    RarbPool("math"),
    RarbPool("code"),
    Limit(),
)


class DatasetRegistry:
    """
    Facade over the registered datasets: lookup by name, seeded sampling,
    and taxonomy profiling of a sample in one call.
    """

    def __init__(
        self,
        datasets: Iterable[RegistryDataset] = DATASETS,
        extractor: FeatureExtractor | None = None,
    ) -> None:
        self._extractor = extractor or FeatureExtractor()
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

    def overview(self) -> pd.DataFrame:
        """One row per dataset: the card's dimensions plus the live cached
        query count (empty until that dataset is fetched). Offline — reads
        cache metadata only."""
        return pd.DataFrame(
            [
                {
                    **dataset.card.model_dump(),
                    "queries_cached": dataset.queries_cached,
                }
                for dataset in self
            ]
        )

    def sample(
        self, name: DatasetName, n: int | None = None, *, seed: int = 0
    ) -> list[Query]:
        return self[name].sample_queries(n, seed=seed)

    def profile(
        self, name: DatasetName, n: int | None = None, *, seed: int = 0
    ) -> CorpusFeatures:
        """Run a query sample through every registered feature group — the
        statistical approximation used to decide which datasets are worth
        acquiring fully."""
        texts = [query.text for query in self.sample(name, n, seed=seed)]
        # extract() consumes the sequence in a single enumerate pass, so a
        # tqdm wrapper tracks the actual regex work query by query
        progress = tqdm(
            texts,
            desc=f"profile {name.value} (n={len(texts)}, seed={seed})",
            unit="query",
        )
        return self._extractor.extract(progress)
