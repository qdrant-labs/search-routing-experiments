"""Floor vocabulary (d33a): what a floor is, how identifier banks group
into floor keys, and the catalog columns no bank emits."""

from __future__ import annotations

from functools import lru_cache

import pandas as pd
from pydantic import BaseModel, ConfigDict

from composition.catalog_axes import DEPTH_AXIS, LENGTH_AXIS, NL_SHAPE_AXIS
from query_taxonomy.banks import BANKS
from query_taxonomy.taxonomy import Domain, FeatureGroup


@lru_cache(maxsize=1)
def _bank_domains() -> dict[str, Domain]:
    return {bank().name.value: bank().domain for bank in BANKS}


def identifier_floor_key(bank_name: str) -> str:
    """The id-floor key for one identifier bank (d33a grouping): `*_like`
    banks pool into id:shape_guess, GENERAL-domain banks split out as
    id:<bank>, field-domain banks aggregate as id:<domain>. The single
    home of this mapping — the supply index and the derivers share it."""
    if bank_name.endswith("_like"):
        return "id:shape_guess"
    domain = _bank_domains()[bank_name]
    if domain is Domain.GENERAL:
        return f"id:{bank_name}"
    return f"id:{domain.value}"

SPAN_PREFIXES: tuple[str, ...] = tuple(
    f"{group.value}."
    for group in (
        FeatureGroup.STRUCTURED_IDENTIFIERS,
        FeatureGroup.SENTENCE_MARKERS,
        FeatureGroup.LOGICAL_STRUCTURES,
    )
)
STAT_AXES = (LENGTH_AXIS, DEPTH_AXIS, NL_SHAPE_AXIS)


class FloorSpec(BaseModel):
    """One floor. `amount` is an evidence amount in weight currency
    (d33a); `credit` is the weight a pool row earns the floor when
    selected (0 = not a member); `cap_waived` when a single dataset
    carries the whole supply (d29)."""

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    key: str
    amount: float
    credit: pd.Series
    cap_waived: bool


IDENTIFIER_SPANS = "derived.identifier_spans"
"""Total structured-identifier spans on a row, so "carries no identifier of
any kind" is one band instead of one per bank. Deliberately OUTSIDE the span
prefixes: `SpanCountAxis` sums everything under a group prefix, and a total
living there would be counted twice."""

CORRUPTION_SPANS = "derived.corruption_spans"
"""Total corruption spans on a row, so the damage LADDER (clean / one span /
two or more) is one band — `AxisBand` reads a single column and cannot sum the
four corruption kinds."""

DERIVED_TOTALS = {
    IDENTIFIER_SPANS: FeatureGroup.STRUCTURED_IDENTIFIERS,
    CORRUPTION_SPANS: FeatureGroup.CORRUPTION,
}


def read_catalog(path) -> pd.DataFrame:
    """The catalog with string query ids, retyped ONE column at a time: a
    frame-wide `astype({"query_id": str})` rebuilds all ~118 columns as
    separate blocks, and every later `assign` pays for that fragmentation."""
    return pd.read_parquet(path).assign(
        query_id=lambda frame: frame["query_id"].astype(str)
    )


def with_derived(catalog: pd.DataFrame) -> pd.DataFrame:
    """The catalog plus the columns no bank emits — derived on every read so
    a total can never disagree with the parts it sums."""
    return catalog.assign(**{
        name: catalog[
            [c for c in catalog.columns if c.startswith(f"{group.value}.")]
        ].sum(axis=1)
        for name, group in DERIVED_TOTALS.items()
    })
