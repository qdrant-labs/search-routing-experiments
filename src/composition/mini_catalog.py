"""Catalog-shaped rows for texts that never went through the feature table.
Generated children must be measured by the same extractor configuration as the
base catalog, or their cell membership and floor credit mean something else.
"""

from __future__ import annotations

import pandas as pd

from query_taxonomy.features import FeatureExtractor, QueryFeatures

from composition.floors import with_derived


def feature_columns(features: QueryFeatures) -> dict[str, float]:
    """The catalog column convention (scripts/feature_table.py): span counts as
    `<group>.<type>`, stat scalars as `<bank>.<stat>`."""
    row: dict[str, float] = {}
    for group, counts_by_type in features.tfs.items():
        for type_, count in counts_by_type.items():
            row[f"{group.value}.{type_}"] = count
    for stats_by_bank in features.stats.values():
        for bank_name, stats in stats_by_bank.items():
            for stat in stats:
                row[f"{bank_name}.{stat.name}"] = stat.value
    return row


def mini_catalog(
    rows: pd.DataFrame,
    extractor: FeatureExtractor,
    *,
    columns: tuple[str, ...] = (),
) -> pd.DataFrame:
    """Pool rows as catalog rows, keyed on the lane they will be labelled in.
    `columns` names span columns the caller reads regardless: a text that
    exhibits none of a feature yields no column for it, and a predicate over
    the absent column would raise rather than read the zero it means."""
    records = []
    for row in rows.itertuples(index=False):
        record: dict[str, object] = {
            "dataset": row.home_lane,
            "query_id": row.query_id,
            "checkable": True,
        }
        record.update(feature_columns(extractor.resolve(str(row.query))))
        records.append(record)
    mini = pd.DataFrame(records)
    absent = [column for column in columns if column not in mini.columns]
    filled = mini.reindex(columns=[*mini.columns, *absent]).fillna(0.0)
    return with_derived(filled)
