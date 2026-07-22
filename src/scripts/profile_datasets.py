"""Registry profiles (SPEC d27-28): one CorpusFeatures JSON plus its
CorpusReport text per registered dataset, under src/data/profiles/ —
the human-readable view harvest-target ratification (SPEC d7) reads.

`ProfileStore` owns the artifact the way `FeatureTable` owns the feature
parquets: per-dataset files built lazily and skipped when already on disk.
Extraction runs over the registry's (sample-capped) cached queries — ORCAS
profiles its 100K sample, never the full 10.4M. A bare run touches only
datasets whose query cache already exists; naming a dataset fills its
cache first (one-time network fetch).

    poetry run python src/scripts/profile_datasets.py         # cached datasets only
    poetry run python src/scripts/profile_datasets.py quest   # fetch + profile one
    poetry run python src/scripts/profile_datasets.py --force # re-extract existing

Explore in the notebook:

    from query_taxonomy.features import CorpusFeatures
    profile = CorpusFeatures.model_validate_json(
        (PROFILES_DIR / "quest.json").read_text()
    )
"""

import argparse
from pathlib import Path

from tqdm.auto import tqdm

from dataset_registry import DATASETS, RegistryDataset
from query_taxonomy.features import CorpusFeatures, FeatureExtractor
from query_taxonomy.reporting import CorpusReport

PROFILES_DIR = Path(__file__).resolve().parent.parent / "data" / "profiles"


class ProfileStore:
    """Artifact owner for the registry profiles: builds each dataset's
    CorpusFeatures JSON + summary text from its cached query sample,
    lazily."""

    def __init__(
        self,
        datasets: tuple[RegistryDataset, ...] = DATASETS,
        extractor: FeatureExtractor | None = None,
        profiles_dir: Path = PROFILES_DIR,
    ) -> None:
        # all engines: the signal scalars need spaCy, not just the regex banks
        self._extractor = extractor or FeatureExtractor(engines=None)
        self._datasets = {dataset.name: dataset for dataset in datasets}
        self._profiles_dir = profiles_dir

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(self._datasets)

    def profile_path(self, name: str) -> Path:
        return self._profiles_dir / f"{name}.json"

    def report_path(self, name: str) -> Path:
        return self._profiles_dir / f"{name}.summary.txt"

    def build(self, name: str, *, force: bool = False) -> CorpusFeatures:
        """One dataset's profile: extract if missing (or `force`), else
        read the JSON already on disk. Fills the registry query cache
        first when it does not exist yet."""
        path = self.profile_path(name)
        if path.exists() and not force:
            tqdm.write(f"[{name}] profile hit -> {path} (no extraction)")
            return CorpusFeatures.model_validate_json(path.read_text())
        profile = self._extract(name)
        self._persist(name, profile)
        return profile

    def build_existing(self, *, force: bool = False) -> None:
        """Profile every dataset whose query cache already exists; the
        rest are skipped with a hint instead of triggering a network
        fetch."""
        for name, dataset in self._datasets.items():
            if not dataset.cache_path.exists():
                tqdm.write(
                    f"[skip] {name}: no query cache yet — "
                    f"`poetry run python src/scripts/profile_datasets.py {name}` "
                    f"fetches it"
                )
                continue
            self.build(name, force=force)

    def _extract(self, name: str) -> CorpusFeatures:
        texts = [query.text for query in self._datasets[name].sample_queries()]
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
