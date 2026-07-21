"""Grounded-snapshot profiles (SPEC d27-28): one CorpusFeatures JSON plus
its CorpusReport text per RetrievalDataset snapshot, under
src/data/profiles/ — so profiles align with the exact query sets the
demo/NDCG framework runs on.

`ProfileStore` owns the artifact the way `FeatureTable` owns the feature
parquets: per-dataset files built lazily and skipped when already on disk.
A bare run never materializes a snapshot; naming a dataset does (one-time
network fetch) before profiling it.

    poetry run python src/scripts/profile_datasets.py                     # existing snapshots only
    poetry run python src/scripts/profile_datasets.py msmarco-passage-dev # materialize + profile one
    poetry run python src/scripts/profile_datasets.py --force             # re-extract existing

Explore in the notebook:

    from query_taxonomy.features import CorpusFeatures
    profile = CorpusFeatures.model_validate_json(
        (PROFILES_DIR / "trec-dl-2022.json").read_text()
    )
"""

import argparse
from pathlib import Path

import pandas as pd
from tqdm.auto import tqdm

from hybrid_search_rrf_dataset.retrieval import (
    MSMarcoDev,
    NFCorpus,
    RetrievalDataset,
    SciFact,
    TrecDL2022,
)
from query_taxonomy.features import CorpusFeatures, FeatureExtractor
from query_taxonomy.reporting import CorpusReport

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
PROFILES_DIR = DATA_DIR / "profiles"

DATASETS: tuple[RetrievalDataset, ...] = (
    TrecDL2022(30000),
    NFCorpus(),
    SciFact(),
    MSMarcoDev(5000),
)


class ProfileStore:
    """Artifact owner for the snapshot profiles: builds each dataset's
    CorpusFeatures JSON + summary text from its saved snapshot, lazily."""

    def __init__(
        self,
        datasets: tuple[RetrievalDataset, ...] = DATASETS,
        extractor: FeatureExtractor | None = None,
        data_dir: Path = DATA_DIR,
        profiles_dir: Path = PROFILES_DIR,
    ) -> None:
        # all engines: the signal scalars need spaCy, not just the regex banks
        self._extractor = extractor or FeatureExtractor(engines=None)
        self._datasets = {dataset.name: dataset for dataset in datasets}
        self._data_dir = data_dir
        self._profiles_dir = profiles_dir

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(self._datasets)

    def profile_path(self, name: str) -> Path:
        return self._profiles_dir / f"{name}.json"

    def report_path(self, name: str) -> Path:
        return self._profiles_dir / f"{name}.summary.txt"

    def snapshot_path(self, name: str) -> Path:
        return self._data_dir / name / "queries.parquet"

    def build(self, name: str, *, force: bool = False) -> CorpusFeatures:
        """One dataset's profile: extract if missing (or `force`), else
        read the JSON already on disk. Materializes the snapshot first
        when it does not exist yet."""
        path = self.profile_path(name)
        if path.exists() and not force:
            tqdm.write(f"[{name}] profile hit -> {path} (no extraction)")
            return CorpusFeatures.model_validate_json(path.read_text())
        profile = self._extract(name)
        self._persist(name, profile)
        return profile

    def build_existing(self, *, force: bool = False) -> None:
        """Profile every dataset that already has a snapshot; the rest are
        skipped with a hint instead of triggering a network fetch."""
        for name in self.names:
            if not self.snapshot_path(name).exists():
                tqdm.write(
                    f"[skip] {name}: no snapshot yet — "
                    f"`poetry run python src/scripts/profile_datasets.py {name}` "
                    f"materializes it"
                )
                continue
            self.build(name, force=force)

    def _snapshot_texts(self, name: str) -> list[str]:
        if not self.snapshot_path(name).exists():
            dataset = self._datasets[name]
            dataset.materialize()
            dataset.save(self._data_dir)
        return pd.read_parquet(self.snapshot_path(name))["text"].tolist()

    def _extract(self, name: str) -> CorpusFeatures:
        texts = self._snapshot_texts(name)
        progress = tqdm(texts, desc=f"profile {name} (n={len(texts)})", unit="query")
        return self._extractor.extract(progress)

    def _persist(self, name: str, profile: CorpusFeatures) -> None:
        self._profiles_dir.mkdir(parents=True, exist_ok=True)
        self.profile_path(name).write_text(profile.model_dump_json())
        report = CorpusReport(profile).text()
        self.report_path(name).write_text(report + "\n")
        tqdm.write(f"\n{'=' * 72}\n{name}\n{'=' * 72}\n{report}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Profile grounded snapshots into src/data/profiles/."
    )
    parser.add_argument(
        "names",
        nargs="*",
        metavar="DATASET",
        help="datasets to profile, materializing their snapshots if missing "
        "(default: every dataset whose snapshot already exists)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="re-extract profiles that already exist",
    )
    args = parser.parse_args()

    store = ProfileStore()
    unknown = set(args.names) - set(store.names)
    if unknown:
        parser.error(
            f"unknown datasets: {sorted(unknown)}; registered: {sorted(store.names)}"
        )

    if args.names:
        for name in args.names:
            store.build(name, force=args.force)
    else:
        store.build_existing(force=args.force)


if __name__ == "__main__":
    main()
