"""The parent pool (d51g): catalog rows the dataset has never spent. Excluding
every selected query — and reserving each one already used as a parent — is
what makes a parent/child near-duplicate pair impossible rather than unlikely.
"""

from __future__ import annotations

from collections.abc import Mapping

import pandas as pd

from augmentation.config import AugmentationPaths
from augmentation.pool import GeneratedPool
from composition.compose import RegistryDataset, join_text
from composition.floors import identifier_floor_key
from query_taxonomy.taxonomy import FeatureGroup

_IDENTIFIERS = f"{FeatureGroup.STRUCTURED_IDENTIFIERS.value}."


def _key(frame: pd.DataFrame) -> pd.Series:
    return frame["dataset"].astype(str) + "\x00" + frame["query_id"].astype(str)


def identifier_floors(catalog: pd.DataFrame) -> pd.Series:
    """Each row's id-floor keys, read off the catalog's span columns — the
    `floors` column Inject's structural check compares against."""
    columns = [c for c in catalog.columns if c.startswith(_IDENTIFIERS)]
    keys = [identifier_floor_key(c.rpartition(".")[2]) for c in columns]
    present = catalog[columns].to_numpy() > 0
    return pd.Series(
        [
            sorted({k for k, on in zip(keys, row, strict=True) if on})
            for row in present
        ],
        index=catalog.index,
    )


class ParentPool:
    """The unspent side of the catalog. Owns no artifact — it reads the
    catalog, the selection and the generated pool."""

    def __init__(
        self,
        catalog: pd.DataFrame,
        selection: pd.DataFrame,
        datasets: Mapping[str, RegistryDataset],
        *,
        pool: GeneratedPool | None = None,
        paths: AugmentationPaths | None = None,
    ) -> None:
        self._catalog = catalog
        self._selection = selection
        self._datasets = datasets
        self._pool = pool or GeneratedPool(paths or AugmentationPaths())

    def reserved(self) -> set[str]:
        """Queries already used as a parent — spent for good, so a later fill
        cannot select one and recreate the pair it was chosen to avoid."""
        pool = self._pool.load()
        if pool.empty:
            return set()
        used = pool.dropna(subset=["generated_from"])
        return set(
            used["parent_dataset"].astype(str)
            + "\x00"
            + used["generated_from"].astype(str)
        )

    def available(self) -> pd.DataFrame:
        """Checkable catalog rows the selection has not touched and no child
        already claims. Every row is natural by construction — the catalog
        holds only source queries — which is the first-generation guard,
        structurally rather than by a column test."""
        spent = set(_key(self._selection)) | self.reserved()
        return self._catalog[
            self._catalog["checkable"] & ~_key(self._catalog).isin(spent)
        ]

    def hydrate(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Attach the columns an operator reads — query text and id floors —
        to the rows dispatch actually chose. Kept off `available()` because
        the text join hits one parquet per lane."""
        rows = frame.copy()
        rows["floors"] = identifier_floors(rows)
        rows["query"] = join_text(rows, self._datasets)
        return rows
