"""v4 candidate catalog — EVERY query, pre-label features only.

The composition universe: every materialized lane's queries.parquet plus the
augmentation pool, with cells / corruption degree / corpus strata derived from
the same deterministic extractor the v3 catalog uses. Reads no label file and no
v3 selected artifact, so it is byte-identical whether or not any route-label
file exists — the invariant that keeps newly materialized lanes visible.

Memory: joins the ~440K-row v2 feature catalog; run under `memguard`.

    ~/.claude/bin/memguard poetry run python src/scripts/build_v4_catalog.py --force
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from composition.cells import CELLS, CELLS_V3
from composition.floors import CORRUPTION_SPANS, read_catalog, with_derived
from composition.mini_catalog import feature_columns
from composition.pool_v3 import CORPUS_AXES, STRATA, UNKNOWN
from query_taxonomy.banks import BANKS
from query_taxonomy.core import Engine
from query_taxonomy.features import FeatureExtractor
from query_taxonomy.taxonomy import FeatureGroup

DATA = Path(__file__).resolve().parent.parent / "data"
V2_CATALOG = DATA / "feature_table" / "catalog.parquet"
QUERY_CORPUS_STATS = DATA / "route_labels" / "query_corpus_stats.parquet"
POOL = DATA / "augmentation" / "pool.parquet"
OUT = DATA / "v4" / "catalog_v4.parquet"

IDENTITY = ["dataset", "query_id", "query", "provenance", "operator", "family", "home_lane"]

# The taxonomy groups the v2 catalog was built without; computed fresh per query.
NEW_GROUPS = [
    FeatureGroup.CORRUPTION,
    FeatureGroup.STATISTICAL_METRICS,
    FeatureGroup.STRUCTURED_IDENTIFIERS,
]
NEW_PREFIXES = (
    "corruption.", "unknown_token_rate.", "term_rarity.", "subword_fragmentation.",
)
IDENTIFIER_PREFIX = f"{FeatureGroup.STRUCTURED_IDENTIFIERS.value}."
CORPUS_STAT_COLS = {"corpus_idf": "avg_idf", "corpus_oov": "oov_share", "corpus_pmi": "min_pmi"}


def _natural() -> pd.DataFrame:
    """Every materialized lane's queries — a lane is a dir holding queries.parquet."""
    frames = []
    for path in sorted(DATA.glob("*/queries.parquet")):
        lane = path.parent.name
        q = pd.read_parquet(path).rename(columns={"text": "query"})
        q = q.astype({"query_id": str}).assign(
            dataset=lane, provenance="natural", operator=None,
            family=None, home_lane=lane,
        )
        frames.append(q[IDENTITY])
    return pd.concat(frames, ignore_index=True)


def _augmented() -> pd.DataFrame:
    """The generated/corrupted pool, keyed on the lane it will be labelled in.
    `generated_from` is the anti-clumping family; natural rows have none."""
    if not POOL.exists():
        return pd.DataFrame(columns=IDENTITY)
    p = pd.read_parquet(
        POOL,
        columns=["query_id", "query", "operator", "provenance", "generated_from", "home_lane"],
    ).astype({"query_id": str})
    p = p.rename(columns={"generated_from": "family", "home_lane": "dataset"})
    p["home_lane"] = p["dataset"]
    return p[IDENTITY]


def _universe() -> pd.DataFrame:
    """All candidates, deduped on (dataset, query_id) with natural winning ties."""
    universe = pd.concat([_natural(), _augmented()], ignore_index=True)
    before = len(universe)
    universe = universe.drop_duplicates(["dataset", "query_id"], keep="first")
    dropped = before - len(universe)
    if dropped:
        print(f"  dropped {dropped:,} duplicate (dataset, query_id) keys")
    return universe.reset_index(drop=True)


def _absent_identifier_columns(present: set[str]) -> list[str]:
    """Identifier banks the v2 join carries no column for — zero-filled so a
    cell banding on an absent column reads the zero it means, not a KeyError."""
    shipped = {f"{IDENTIFIER_PREFIX}{bank().name.value}" for bank in BANKS}
    return sorted(shipped - present)


def _new_columns(universe: pd.DataFrame, absent_ids: list[str]) -> pd.DataFrame:
    """The taxonomy columns the v2 catalog lacks, per query — regex/statistical
    only, so no spaCy or model download is needed to build the catalog."""
    extractor = FeatureExtractor(engines=(Engine.REGEX, Engine.WORDFREQ, Engine.TOKENIZER))
    keep = (*NEW_PREFIXES, *absent_ids)
    rows = []
    for text in tqdm(universe["query"].fillna("").astype(str), unit="query"):
        cols = feature_columns(extractor.resolve(text, groups=NEW_GROUPS))
        rows.append({k: v for k, v in cols.items() if k.startswith(keep)})
    new = pd.DataFrame(rows)
    missing = [c for c in absent_ids if c not in new.columns]
    return new.reindex(columns=[*new.columns, *missing]).fillna(0.0)


def _active_cells(columns: set[str]):
    """v2 + v3 cells whose every banded column the catalog carries."""
    return tuple(
        cell for cell in (*CELLS, *CELLS_V3)
        if all(band.column in columns for band in cell.bands)
    )


def _cells(catalog: pd.DataFrame) -> list[list[str]]:
    """Multi cell membership per row via the same predicates the v3 catalog uses."""
    per_row: list[set[str]] = [set() for _ in range(len(catalog))]
    for cell in _active_cells(set(catalog.columns)):
        for i in np.nonzero(cell.select(catalog).to_numpy())[0]:
            per_row[i].add(cell.name)
    return [sorted(names) for names in per_row]


def _corpus_strata(catalog: pd.DataFrame, in_qcs: np.ndarray, p25: float, p75: float) -> dict[str, np.ndarray]:
    """The three marginal corpus axes — IDF quartile band (edges measured off
    query_corpus_stats), OOV presence, PMI sentinel. Rows the stats file does
    not cover are UNKNOWN: counted, never floored."""
    idf = catalog["query_corpus.avg_idf"].to_numpy(dtype=float)
    oov = catalog["query_corpus.oov_share"].to_numpy(dtype=float)
    pmi = catalog["query_corpus.min_pmi"].to_numpy(dtype=float)
    absent = ~in_qcs
    return {
        "corpus_idf": np.where(
            absent | np.isnan(idf), UNKNOWN,
            np.where(idf < p25, "low_idf", np.where(idf >= p75, "high_idf", "mid_idf")),
        ),
        "corpus_oov": np.where(
            absent | np.isnan(oov), UNKNOWN,
            np.where(oov > 0, "has_oov", "in_vocab"),
        ),
        # -1.0 is PMIBank's measured "never co-occurs"; NaN is never measurable.
        "corpus_pmi": np.where(
            absent, UNKNOWN,
            np.where(np.isnan(pmi), "unmeasured",
                     np.where(pmi == -1.0, "never_co_occurs", "co_occurring")),
        ),
    }


def build_v4_catalog(*, force: bool = False) -> pd.DataFrame:
    if OUT.exists() and not force:
        return pd.read_parquet(OUT)
    universe = _universe()
    print(f"  universe: {len(universe):,} candidates across {universe['dataset'].nunique()} lanes")

    v2 = read_catalog(V2_CATALOG)
    base = universe[["dataset", "query_id"]].merge(v2, on=["dataset", "query_id"], how="left")
    absent_ids = _absent_identifier_columns(set(base.columns))
    new = _new_columns(universe, absent_ids)
    # A refreshed feature table can already carry some NEW_PREFIXES columns (banks
    # that shipped after the v2 catalog was frozen); drop those from the base so
    # the freshly computed, all-row values win and the concat has no duplicates.
    base = base.drop(columns=[c for c in new.columns if c in base.columns])
    features = pd.concat([base.reset_index(drop=True), new.reset_index(drop=True)], axis=1)

    in_qcs = np.zeros(len(universe), dtype=bool)
    p25 = p75 = float("nan")
    if QUERY_CORPUS_STATS.exists():
        qcs = pd.read_parquet(QUERY_CORPUS_STATS).astype({"query_id": str})
        p25, p75 = qcs["avg_idf"].quantile([0.25, 0.75])
        stat_cols = [c for c in qcs.columns if c not in ("dataset", "query_id")]
        qcs = qcs.rename(columns={c: f"query_corpus.{c}" for c in stat_cols})
        features = features.merge(qcs, on=["dataset", "query_id"], how="left")
        key = pd.MultiIndex.from_arrays([universe["dataset"], universe["query_id"]])
        in_qcs = key.isin(
            pd.MultiIndex.from_arrays([qcs["dataset"], qcs["query_id"]])
        )
    for col in ("query_corpus.avg_idf", "query_corpus.oov_share", "query_corpus.min_pmi"):
        if col not in features.columns:
            features[col] = np.nan

    assert features.columns.is_unique, (
        f"duplicate feature columns: {features.columns[features.columns.duplicated()].tolist()}"
    )
    taxonomy_cols = [c for c in features.columns if c.startswith((*NEW_PREFIXES, *absent_ids))]
    features[taxonomy_cols] = features[taxonomy_cols].fillna(0.0)
    features = with_derived(features)

    degree = pd.cut(
        features[CORRUPTION_SPANS], [-np.inf, 0.5, 1.5, np.inf],
        labels=STRATA["corruption_degree"],
    ).astype(object)
    corpus = _corpus_strata(features, np.asarray(in_qcs), p25, p75)

    catalog = universe.assign(
        row_id=universe["dataset"].astype(str) + ":" + universe["query_id"].astype(str),
        cells=_cells(features),
        corruption_degree=pd.Series(degree).fillna(UNKNOWN).to_numpy(),
        **corpus,
    )
    columns = [
        "dataset", "query_id", "query", "row_id", "cells", "corruption_degree",
        *CORPUS_AXES, "provenance", "operator", "family", "home_lane",
    ]
    catalog = catalog[columns]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    catalog.to_parquet(OUT, index=False)
    return catalog


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    catalog = build_v4_catalog(force=args.force)
    with_cells = int(catalog["cells"].map(bool).sum())
    print(f"v4 catalog: {len(catalog):,} rows x {catalog.shape[1]} cols -> {OUT}")
    print(f"  rows with >=1 cell: {with_cells:,}")
    print(f"  corruption_degree: {dict(catalog['corruption_degree'].value_counts())}")
    print(f"  lanes: {catalog['dataset'].nunique()}  provenance: {dict(catalog['provenance'].value_counts())}")


if __name__ == "__main__":
    main()
