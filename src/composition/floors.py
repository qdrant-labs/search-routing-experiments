"""Floor derivation (d32c, d33a-b). Span-type floors live in evidence
currency over the span pool; stat-band floors live on raw scalar bands
over the zero-span pool. Ports the audit notebook's target-column logic
into the package."""

from __future__ import annotations

import pandas as pd
from pydantic import BaseModel, ConfigDict

from composition.catalog_axes import DEPTH_AXIS, LENGTH_AXIS, NL_SHAPE_AXIS
from query_taxonomy.banks import BANKS
from query_taxonomy.taxonomy import Domain, FeatureGroup

from composition.recipe import Recipe

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


def _spec(
    key: str, amount: float, credit: pd.Series, datasets: pd.Series,
) -> FloorSpec:
    suppliers = datasets[credit > 0].nunique()
    return FloorSpec(
        key=key, amount=amount, credit=credit, cap_waived=suppliers <= 1,
    )


class SpanFloorDeriver:
    """Span-type floors: certified identifier banks grouped by domain
    (general banks split out — number would drown any aggregate), all
    `*_like` banks pooled into the domain-agnostic shape_guess floor,
    marker and logical types one floor each. Credit = discount of the
    most rigid qualifying bank that fired (d33a)."""

    def __init__(self, recipe: Recipe) -> None:
        self._recipe = recipe

    def derive(self, pool: pd.DataFrame) -> list[FloorSpec]:
        floors = []
        for key, columns in sorted(self._column_weights(pool).items()):
            credit = pd.Series(0.0, index=pool.index)
            for column, discount in columns:
                fired = pool[column] > 0
                credit = credit.where(~fired | (credit >= discount), discount)
            floors.append(_spec(
                key, self._recipe.floor_rules.span_floor_weight,
                credit, pool["dataset"],
            ))
        return floors

    def _column_weights(
        self, pool: pd.DataFrame,
    ) -> dict[str, list[tuple[str, float]]]:
        by_name = {bank.name.value: bank for bank in (b() for b in BANKS)}
        out: dict[str, list[tuple[str, float]]] = {}
        for column in pool.columns:
            group, _, type_ = column.partition(".")
            if group == FeatureGroup.STRUCTURED_IDENTIFIERS.value:
                bank = by_name[type_]
                discount = self._recipe.floor_rules.discounts[bank.ambiguity]
                if type_.endswith("_like"):
                    key = "id:shape_guess"
                elif bank.domain is Domain.GENERAL:
                    key = f"id:{type_}"
                else:
                    key = f"id:{bank.domain.value}"
            elif group == FeatureGroup.SENTENCE_MARKERS.value:
                key, discount = f"marker:{type_}", 1.0
            elif group == FeatureGroup.LOGICAL_STRUCTURES.value:
                key, discount = f"logical:{type_}", 1.0
            else:
                continue
            out.setdefault(key, []).append((column, discount))
        return out


class StatFloorDeriver:
    """One floor per raw scalar band (d33b) across the three stat axes.
    Zero-span rows credit 1.0 to every band they fall in — one band per
    axis, so up to three floors per row. Floor amounts derive from the
    recipe (slice rows / band count), so they can never drift from the
    axis definitions."""

    def __init__(self, recipe: Recipe) -> None:
        self._recipe = recipe

    def derive(self, pool: pd.DataFrame) -> list[FloorSpec]:
        band_count = sum(len(axis.labels) for axis in STAT_AXES)
        amount = round(self._recipe.stat_strata_rows / band_count)
        floors = []
        for axis in STAT_AXES:
            bands = axis.bands(pool)
            for label in axis.labels:
                clean = str(label).replace("\n", " ")
                credit = (bands == label).astype(float)
                floors.append(_spec(
                    f"{axis.title}:{clean}", amount, credit, pool["dataset"],
                ))
        return floors


def span_mask(catalog: pd.DataFrame) -> pd.Series:
    """Rows carrying >=1 span of any group — the entity population."""
    columns = [c for c in catalog.columns if c.startswith(SPAN_PREFIXES)]
    return catalog[columns].sum(axis=1) > 0
