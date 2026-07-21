"""Feature table (SPEC d29): one extraction pass per registered dataset ->
per-query parquet rows the composition stage reads. Extraction is the
expensive leg (~minutes of spaCy per 100K queries); selection and the d30
pilots re-read the parquet in seconds, so recipe tweaks never re-pay
extraction.

`FeatureTable` owns the artifact the way `RegistryDataset` owns its query
cache: per-dataset parquet files plus a concatenated catalog under one
directory, built lazily and skipped when already on disk.

Columns follow the SPEC's own notation: span counts as `<group>.<type>`
(e.g. "logical_structures.temporal" = 2 means two temporal spans in that
query), stat scalars as `<bank>.<stat>` (e.g.
"coordination.widest_list_size"). Identity columns: dataset, query_id,
checkable.

`checkable` is card-level for now — True when the dataset ships qrels
(grounding == QQ), same value for every row of a dataset. The registry is
queries-only, so the per-query ">=1 judged doc" flag of d30(c) waits on
the deferred qrels decision.

    poetry run python src/scripts/feature_table.py                  # all registered
    poetry run python src/scripts/feature_table.py --only beir-nfcorpus
    poetry run python src/scripts/feature_table.py --force          # rebuild
"""

import argparse
from pathlib import Path

import pandas as pd
from tqdm.auto import tqdm

from dataset_registry.core import DatasetName, Grounding, Query, RegistryDataset
from dataset_registry.registry import DatasetRegistry
from query_taxonomy.features import FeatureExtractor, QueryFeatures

DEFAULT_TABLE_DIR = Path(__file__).resolve().parent.parent / "data" / "feature_table"


class FeatureTable:
    """The d29 parquet artifact: builds per-dataset tables from the
    registry's cached queries, concatenates them into a catalog, and reads
    them back for selection and the d30 pilots."""

    ID_COLUMNS = ("dataset", "query_id", "checkable")

    def __init__(
        self,
        registry: DatasetRegistry | None = None,
        extractor: FeatureExtractor | None = None,
        table_dir: Path | None = None,
    ) -> None:
        self._registry = registry if registry is not None else DatasetRegistry()
        # all engines: the five signals need spaCy, not just the regex banks
        self._extractor = extractor or FeatureExtractor(engines=None)
        self._table_dir = table_dir if table_dir is not None else DEFAULT_TABLE_DIR

    @property
    def catalog_path(self) -> Path:
        return self._table_dir / "catalog.parquet"

    def dataset_path(self, name: DatasetName) -> Path:
        return self._table_dir / f"{name.value}.parquet"

    def build(self, name: DatasetName, *, force: bool = False) -> pd.DataFrame:
        """One dataset's table: extract if missing (or `force`), else read
        the parquet already on disk."""
        path = self.dataset_path(name)
        if path.exists() and not force:
            tqdm.write(f"[{name.value}] table hit -> {path} (no extraction)")
            return pd.read_parquet(path)
        frame = self._extract(self._registry[name])
        self._table_dir.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(path, index=False)
        tqdm.write(
            f"[{name.value}] {len(frame):,} rows x {frame.shape[1]} cols -> {path}"
        )
        return frame

    def build_all(
        self,
        only: tuple[DatasetName, ...] | None = None,
        *,
        force: bool = False,
    ) -> list[DatasetName]:
        """Build every registered dataset (or the `only` subset), then
        refresh the catalog. Returns the names that failed — one gated or
        broken source must not kill the batch."""
        failed: list[DatasetName] = []
        for dataset in self._registry:
            name = dataset.card.name
            if only is not None and name not in only:
                continue
            try:
                self.build(name, force=force)
            except Exception as error:
                failed.append(name)
                tqdm.write(f"[{name.value}] FAILED: {error}")
        catalog = self.catalog()
        tqdm.write(
            f"catalog: {len(catalog):,} rows x {catalog.shape[1]} cols "
            f"({catalog['dataset'].nunique()} datasets) -> {self.catalog_path}"
        )
        return failed

    def catalog(self) -> pd.DataFrame:
        """Concatenate every per-dataset table on disk, write the catalog
        parquet, and return it. Column union across datasets: a span type
        one corpus never fires becomes 0.0 there, so selection filters see
        a uniform schema."""
        paths = sorted(
            path
            for path in self._table_dir.glob("*.parquet")
            if path != self.catalog_path
        )
        catalog = self._ordered(
            pd.concat(
                [pd.read_parquet(path) for path in paths], ignore_index=True
            ).fillna(0.0)
        )
        catalog.to_parquet(self.catalog_path, index=False)
        return catalog

    def _extract(self, dataset: RegistryDataset) -> pd.DataFrame:
        """One extraction pass over the dataset's (capped) query sample."""
        card = dataset.card
        checkable = card.grounding is Grounding.QQ
        queries = tqdm(
            dataset.sample_queries(),
            desc=f"extract {card.name.value}",
            unit="query",
        )
        rows = [
            self._row(card.name, checkable, query, self._extractor.resolve(query.text))
            for query in queries
        ]
        # span columns exist only where a bank fired; absent = zero spans
        return self._ordered(pd.DataFrame(rows).fillna(0.0))

    def _row(
        self,
        name: DatasetName,
        checkable: bool,
        query: Query,
        features: QueryFeatures,
    ) -> dict[str, object]:
        row: dict[str, object] = {
            "dataset": name.value,
            "query_id": query.query_id,
            "checkable": checkable,
        }
        for group, counts_by_type in features.tfs.items():
            for type_, count in counts_by_type.items():
                row[f"{group.value}.{type_}"] = count
        for stats_by_bank in features.stats.values():
            for bank_name, stats in stats_by_bank.items():
                for stat in stats:
                    row[f"{bank_name}.{stat.name}"] = stat.value
        return row

    def _ordered(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Identity columns first, feature columns sorted — one stable
        order for per-dataset files and the catalog alike."""
        features = sorted(c for c in frame.columns if c not in self.ID_COLUMNS)
        return frame[[*self.ID_COLUMNS, *features]]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build the SPEC d29 feature table from the registry."
    )
    parser.add_argument(
        "--only",
        nargs="*",
        type=DatasetName,
        metavar="DATASET",
        help="build only these dataset names (default: all registered)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="rebuild per-dataset tables that already exist",
    )
    args = parser.parse_args()

    failed = FeatureTable().build_all(
        tuple(args.only) if args.only else None, force=args.force
    )
    if failed:
        print(f"failed ({len(failed)}): {', '.join(n.value for n in failed)}")


if __name__ == "__main__":
    main()
