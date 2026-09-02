"""Version-neutral candidate catalog for two-rung composition.

Sources: dataset_registry cache (natural) plus admission-gated augmentation pool
(generated). Every candidate carries a manifest guaranteeing its graded answer
docs are in the local corpus BEFORE labeling; rows that fail the guarantee are
dropped at acquisition, never silently converted into all-zero measurements.
This module reads no label file and knows nothing about dataset release versions.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from augmentation.core import CreditGate
from composition.cells import DECLARED_CELLS
from composition.floors import CORRUPTION_SPANS, read_catalog, with_derived
from composition.mini_catalog import feature_columns
from composition.strata import CORPUS_AXES, STRATA, UNKNOWN
from dataset_registry import DATASETS, RegistryDataset
from hybrid_search_rrf_dataset.lanes import LANES
from query_taxonomy.banks import BANKS
from query_taxonomy.core import Engine
from query_taxonomy.features import FeatureExtractor
from query_taxonomy.taxonomy import FeatureGroup

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA = REPO_ROOT / "src" / "data"
AUG = DATA / "augmentation"
FEATURE_CATALOG = DATA / "feature_table" / "catalog.parquet"

CATALOG_COLUMNS: tuple[str, ...] = (
    "row_id",
    "dataset",
    "query_id",
    "query",
    "provenance",
    "operator",
    "family",
    "debt_id",
    "home_lane",
    "cells",
    "corruption_degree",
    *CORPUS_AXES,
    "answer_covered",
    "answer_manifest_id",
    "content_fp",
)

MANIFEST_COLUMNS: tuple[str, ...] = ("manifest_id", "dataset", "query_id", "doc_id", "relevance")

_NEW_GROUPS = [
    FeatureGroup.CORRUPTION,
    FeatureGroup.STATISTICAL_METRICS,
    FeatureGroup.STRUCTURED_IDENTIFIERS,
]
_NEW_PREFIXES = (
    "corruption.",
    "unknown_token_rate.",
    "term_rarity.",
    "subword_fragmentation.",
)
_IDENTIFIER_PREFIX = f"{FeatureGroup.STRUCTURED_IDENTIFIERS.value}."


@dataclass(frozen=True)
class CatalogConfig:
    """Explicit acquisition policy (spec:97-102). Never inferred from historical
    label membership: expanding a lane cannot require a composer change."""

    per_lane_cap: dict[str, int] = field(default_factory=dict)
    default_per_lane_cap: int | None = None
    include_augmentation: bool = True
    seed: int = 0


class CandidateCatalog:
    """Assemble the version-neutral candidate universe. Public API:
    `build()` -> (catalog_df, manifests_df, catalog_fp)."""

    def __init__(
        self,
        config: CatalogConfig = CatalogConfig(),
        *,
        data_dir: Path = DATA,
        datasets: Iterable[RegistryDataset] = DATASETS,
    ) -> None:
        self._config = config
        self._data = data_dir
        self._datasets = list(datasets)

    def build(self) -> tuple[pd.DataFrame, pd.DataFrame, str]:
        rows_natural, manifests_natural, dropped_natural = self._natural()
        if self._config.include_augmentation:
            rows_generated, manifests_generated, dropped_generated = self._generated()
        else:
            rows_generated = pd.DataFrame(columns=_identity_columns())
            manifests_generated = pd.DataFrame(columns=list(MANIFEST_COLUMNS))
            dropped_generated = 0

        rows = pd.concat([rows_natural, rows_generated], ignore_index=True)
        before = len(rows)
        rows = rows.drop_duplicates(["dataset", "query_id"], keep="first").reset_index(drop=True)
        dedup_dropped = before - len(rows)

        rows = self._attach_strata(rows)
        rows["row_id"] = rows["dataset"].astype(str) + ":" + rows["query_id"].astype(str)
        rows["content_fp"] = _row_fingerprints(rows)
        rows = rows[list(CATALOG_COLUMNS)]

        manifests = pd.concat([manifests_natural, manifests_generated], ignore_index=True)
        manifests = manifests[list(MANIFEST_COLUMNS)]
        catalog_fp = _catalog_fingerprint(rows, self._config)

        rows.attrs["dropped_natural"] = dropped_natural
        rows.attrs["dropped_generated"] = dropped_generated
        rows.attrs["dropped_dedup"] = dedup_dropped
        return rows, manifests, catalog_fp

    # ------------------------------------------------------------- natural ---
    def _natural(self) -> tuple[pd.DataFrame, pd.DataFrame, int]:
        """Registry cache filtered by the per-lane answer-coverage guarantee.
        Skips lanes whose local corpus/qrels are absent — those cannot promise
        corpus inclusion at label time (spec:352)."""
        frames: list[pd.DataFrame] = []
        manifest_frames: list[pd.DataFrame] = []
        dropped = 0
        for dataset in self._datasets:
            covered, manifest = self._coverage_index(dataset.name)
            if covered is None:
                continue
            cap = self._config.per_lane_cap.get(dataset.name, self._config.default_per_lane_cap)
            queries = list(dataset.sample_queries(cap, seed=self._config.seed))
            frame = pd.DataFrame(
                (
                    {"query_id": str(q.query_id), "query": str(q.text)}
                    for q in queries
                    if str(q.query_id) in covered
                ),
                columns=["query_id", "query"],
            )
            dropped += len(queries) - len(frame)
            if frame.empty:
                continue
            frame = frame.assign(
                dataset=dataset.name,
                provenance="natural",
                operator=None,
                family=frame["query_id"].map(lambda qid: f"{dataset.name}:{qid}"),
                debt_id="",
                home_lane=dataset.name,
                answer_covered=True,
                answer_manifest_id=frame["query_id"].map(covered),
            )
            frames.append(frame[list(_identity_columns())])
            manifest_frames.append(
                manifest[manifest["query_id"].isin(frame["query_id"])].assign(dataset=dataset.name)
            )
        if not frames:
            return (
                pd.DataFrame(columns=_identity_columns()),
                pd.DataFrame(columns=list(MANIFEST_COLUMNS)),
                dropped,
            )
        return (
            pd.concat(frames, ignore_index=True),
            pd.concat(manifest_frames, ignore_index=True)[list(MANIFEST_COLUMNS)],
            dropped,
        )

    def _coverage_index(self, lane: str) -> tuple[dict[str, str] | None, pd.DataFrame]:
        """Per-lane {query_id -> manifest_id} for queries whose graded docs are
        all local, plus the manifest rows. Returns (None, empty) when the lane
        has no local qrels/corpus (unpromiseable at label time)."""
        lane_dir = self._data / lane
        qrels_path = lane_dir / "qrels.parquet"
        corpus_path = lane_dir / "corpus.parquet"
        if not qrels_path.exists() or not corpus_path.exists():
            return None, pd.DataFrame(columns=list(MANIFEST_COLUMNS))
        min_relevance = LANES[lane].min_relevance if lane in LANES else 1
        qrels = pd.read_parquet(qrels_path).astype({"query_id": str, "doc_id": str})
        graded = qrels[qrels["relevance"] >= min_relevance]
        if graded.empty:
            return {}, pd.DataFrame(columns=list(MANIFEST_COLUMNS))
        # Coverage is CONTENT-level, not id-level. An empty-body doc has a valid
        # doc_id but is unembeddable/unretrievable (cloud encoders 400 on it), so
        # it does NOT cover an answer — counting it as covered silently mints
        # false all-zero rows, the exact failure this module's docstring forbids.
        text_cols = ["doc_id", "text"] + (
            ["title"] if "title" in pq.read_schema(corpus_path).names else []
        )
        corpus = pd.read_parquet(corpus_path, columns=text_cols)
        embed_text = corpus["text"].fillna("")
        if "title" in corpus.columns:
            embed_text = corpus["title"].fillna("") + " " + embed_text
        corpus_docs = set(corpus.loc[embed_text.str.strip() != "", "doc_id"].astype(str))
        in_corpus = graded[graded["doc_id"].isin(corpus_docs)]
        # a query is admissible only if EVERY graded doc is in the corpus
        graded_counts = graded.groupby("query_id").size()
        kept_counts = in_corpus.groupby("query_id").size()
        full = kept_counts[kept_counts == graded_counts.reindex(kept_counts.index)].index
        manifest_rows = in_corpus[in_corpus["query_id"].isin(full)].copy()
        manifest_rows["manifest_id"] = manifest_rows.groupby("query_id")["doc_id"].transform(
            lambda docs: _manifest_id(lane, docs.iloc[0], docs)
        )
        covered = dict(zip(manifest_rows["query_id"], manifest_rows["manifest_id"]))
        return covered, manifest_rows.assign(dataset=lane)

    # ----------------------------------------------------------- generated ---
    def _generated(self) -> tuple[pd.DataFrame, pd.DataFrame, int]:
        """Augmentation pool filtered by (a) the credit-gate admission policy
        and (b) the augmentation-qrels answer-coverage guarantee."""
        pool_path = AUG / "pool.parquet"
        if not pool_path.exists():
            return (
                pd.DataFrame(columns=_identity_columns()),
                pd.DataFrame(columns=list(MANIFEST_COLUMNS)),
                0,
            )
        pool = pd.read_parquet(pool_path).astype({"query_id": str})
        raw_size = len(pool)
        pool = _admitted_augmentation(pool, AUG)
        covered, manifest = self._augmentation_coverage(pool)
        kept = pool[pool["query_id"].isin(covered)]
        dropped = raw_size - len(kept)
        if kept.empty:
            return (
                pd.DataFrame(columns=_identity_columns()),
                pd.DataFrame(columns=list(MANIFEST_COLUMNS)),
                dropped,
            )
        frame = pd.DataFrame({
            "dataset": kept["home_lane"].astype(str),
            "query_id": kept["query_id"].astype(str),
            "query": kept["query"].astype(str),
            "provenance": kept["provenance"].astype(str),
            "operator": kept["operator"].astype(str),
            "family": kept["generated_from"].astype(str).where(
                kept["generated_from"].notna() & (kept["generated_from"].astype(str) != ""),
                kept["query_id"].astype(str),
            ),
            # the shortage that asked for this row; already an `axis:name` key
            "debt_id": kept["floor"].astype(str),
            "home_lane": kept["home_lane"].astype(str),
            "answer_covered": True,
            "answer_manifest_id": kept["query_id"].map(covered),
        })
        return frame[list(_identity_columns())], manifest, dropped

    def _augmentation_coverage(
        self, pool: pd.DataFrame
    ) -> tuple[dict[str, str], pd.DataFrame]:
        qrels_path = AUG / "qrels.parquet"
        if pool.empty or not qrels_path.exists():
            return {}, pd.DataFrame(columns=list(MANIFEST_COLUMNS))
        qrels = pd.read_parquet(qrels_path).astype({"query_id": str, "doc_id": str})
        wanted_ids = set(pool["query_id"].astype(str))
        qrels = qrels[qrels["query_id"].isin(wanted_ids)]
        if qrels.empty:
            return {}, pd.DataFrame(columns=list(MANIFEST_COLUMNS))
        min_relevance = qrels["min_relevance"] if "min_relevance" in qrels.columns else 1
        graded = qrels[qrels["relevance"] >= min_relevance]
        # per-query lane binding via pool.home_lane; the aug qrels are lane-agnostic
        lane_of = dict(zip(pool["query_id"].astype(str), pool["home_lane"].astype(str)))
        lane_series = graded["query_id"].map(lane_of)
        corpus_by_lane: dict[str, set[str]] = {}
        for lane in lane_series.dropna().unique():
            corpus_path = self._data / lane / "corpus.parquet"
            if not corpus_path.exists():
                corpus_by_lane[lane] = set()
                continue
            corpus_by_lane[lane] = set(
                pd.read_parquet(corpus_path, columns=["doc_id"])["doc_id"].astype(str)
            )
        in_corpus = graded[
            [doc in corpus_by_lane.get(lane, set()) for doc, lane in zip(graded["doc_id"], lane_series)]
        ]
        graded_counts = graded.groupby("query_id").size()
        kept_counts = in_corpus.groupby("query_id").size()
        full = kept_counts[kept_counts == graded_counts.reindex(kept_counts.index)].index
        manifest_rows = in_corpus[in_corpus["query_id"].isin(full)].copy()
        manifest_rows["dataset"] = manifest_rows["query_id"].map(lane_of)
        manifest_rows["manifest_id"] = manifest_rows.groupby(["dataset", "query_id"])["doc_id"].transform(
            lambda docs: _manifest_id(str(docs.name), docs.iloc[0], docs)
        )
        covered = dict(zip(manifest_rows["query_id"], manifest_rows["manifest_id"]))
        return covered, manifest_rows[list(MANIFEST_COLUMNS)]

    # ------------------------------------------------------------- strata ---
    def _attach_strata(self, rows: pd.DataFrame) -> pd.DataFrame:
        """Compute cells, corruption_degree, and the three corpus bands per
        row. Bands the local join cannot resolve are `unknown` (spec:114-116)."""
        if rows.empty:
            for col in ("cells", "corruption_degree", *CORPUS_AXES):
                rows[col] = pd.Series(dtype=object)
            return rows

        extractor = FeatureExtractor(engines=(Engine.REGEX, Engine.WORDFREQ, Engine.TOKENIZER))
        absent_ids = _absent_identifier_columns()
        keep_prefixes = (*_NEW_PREFIXES, *absent_ids)
        feature_rows: list[dict[str, float]] = []
        for text in rows["query"].fillna("").astype(str):
            cols = feature_columns(extractor.resolve(text, groups=_NEW_GROUPS))
            feature_rows.append({k: v for k, v in cols.items() if k.startswith(keep_prefixes)})
        new = pd.DataFrame(feature_rows).reindex(
            columns=list({k for r in feature_rows for k in r} | set(absent_ids))
        ).fillna(0.0)

        if FEATURE_CATALOG.exists():
            cached = read_catalog(FEATURE_CATALOG)
            base = rows[["dataset", "query_id"]].merge(
                cached, on=["dataset", "query_id"], how="left"
            )
            base = base.drop(columns=[c for c in new.columns if c in base.columns])
            features = pd.concat([base.reset_index(drop=True), new.reset_index(drop=True)], axis=1)
        else:
            features = pd.concat(
                [rows[["dataset", "query_id"]].reset_index(drop=True), new.reset_index(drop=True)],
                axis=1,
            )

        taxonomy_cols = [c for c in features.columns if c.startswith(keep_prefixes)]
        features[taxonomy_cols] = features[taxonomy_cols].fillna(0.0)
        features = with_derived(features)

        cells = _cells(features)
        degree = pd.cut(
            features[CORRUPTION_SPANS],
            [-np.inf, 0.5, 1.5, np.inf],
            labels=STRATA["corruption_degree"],
        ).astype(object)
        bands = _corpus_bands(rows, self._data)

        return rows.assign(
            cells=cells,
            corruption_degree=pd.Series(degree).fillna(UNKNOWN).to_numpy(),
            **bands,
        )


# -------------------------------------------------------------- helpers ---


def _identity_columns() -> tuple[str, ...]:
    return (
        "dataset",
        "query_id",
        "query",
        "provenance",
        "operator",
        "family",
        "debt_id",
        "home_lane",
        "answer_covered",
        "answer_manifest_id",
    )


def _admitted_augmentation(pool: pd.DataFrame, aug_dir: Path) -> pd.DataFrame:
    """Filter pool to rows the shared coherence/credit policy passed (spec:123):
    ungated rows in, coherence-gated rows in iff the audit says True, and
    declaration-audited rows in iff their query_id was cleared."""
    gate = pool["credit_gate"].fillna("none").astype(str)
    admitted = gate == str(CreditGate.NONE)

    coherence_ids: set[str] = set()
    audit_path = aug_dir / "coherence_audit.parquet"
    if audit_path.exists():
        audit = pd.read_parquet(audit_path).astype({"query_id": str})
        coherence_ids = set(audit.loc[audit["verdict"].astype(bool), "query_id"])
    admitted |= (gate == str(CreditGate.COHERENCE_GATE)) & pool["query_id"].astype(str).isin(coherence_ids)

    declaration_ids: set[str] = set()
    passed_path = aug_dir / "declaration_audit_passed.txt"
    if passed_path.exists():
        declaration_ids = {
            line.strip()
            for line in passed_path.read_text().splitlines()
            if line.strip() and not line.strip().startswith("#")
        }
    admitted |= (gate == str(CreditGate.DECLARATION_AUDIT)) & pool["query_id"].astype(str).isin(declaration_ids)

    return pool[admitted]


def _absent_identifier_columns() -> list[str]:
    """Identifier banks that ship as columns but may not be in the feature cache.
    Zero-filled so a cell banding on the column reads the intended zero."""
    return sorted({f"{_IDENTIFIER_PREFIX}{bank().name.value}" for bank in BANKS})


def _cells(features: pd.DataFrame) -> list[list[str]]:
    active = tuple(
        cell for cell in DECLARED_CELLS
        if all(band.column in features.columns for band in cell.bands)
    )
    per_row: list[set[str]] = [set() for _ in range(len(features))]
    for cell in active:
        for i in np.nonzero(cell.select(features).to_numpy())[0]:
            per_row[i].add(cell.name)
    return [sorted(names) for names in per_row]


def _corpus_bands(rows: pd.DataFrame, data_dir: Path) -> dict[str, np.ndarray]:
    """The three marginal corpus bands from query_corpus_stats.parquet — an
    absent stats file leaves every row `unknown` (a coverage miss, never a
    band; spec:114-116)."""
    n = len(rows)
    unknown = np.full(n, UNKNOWN, dtype=object)
    qcs_path = data_dir / "route_labels" / "query_corpus_stats.parquet"
    if not qcs_path.exists():
        return {axis: unknown.copy() for axis in CORPUS_AXES}
    qcs = (
        pd.read_parquet(qcs_path)
        .astype({"query_id": str})
        .drop_duplicates(["dataset", "query_id"])
        .set_index(["dataset", "query_id"])
    )
    key = pd.MultiIndex.from_arrays([rows["dataset"].astype(str), rows["query_id"].astype(str)])
    absent = ~key.isin(qcs.index)
    joined = qcs.reindex(key)
    p25, p75 = qcs["avg_idf"].quantile([0.25, 0.75])
    idf = joined["avg_idf"].to_numpy(dtype=float)
    oov = joined["oov_share"].to_numpy(dtype=float)
    pmi = joined["min_pmi"].to_numpy(dtype=float)
    return {
        "corpus_idf": np.where(
            absent | np.isnan(idf),
            UNKNOWN,
            np.where(idf < p25, "low_idf", np.where(idf >= p75, "high_idf", "mid_idf")),
        ),
        "corpus_oov": np.where(
            absent | np.isnan(oov),
            UNKNOWN,
            np.where(oov > 0, "has_oov", "in_vocab"),
        ),
        "corpus_pmi": np.where(
            absent,
            UNKNOWN,
            np.where(
                np.isnan(pmi),
                "unmeasured",
                np.where(pmi == -1.0, "never_co_occurs", "co_occurring"),
            ),
        ),
    }


def _manifest_id(lane: str, first_doc: str, docs: pd.Series) -> str:
    """Stable id for a manifest = hash of (lane, sorted doc_ids). first_doc is
    unused; the signature matches a groupby-transform closure."""
    material = lane + "|" + "|".join(sorted(str(d) for d in docs))
    return hashlib.sha256(material.encode()).hexdigest()[:16]


def _row_fingerprints(rows: pd.DataFrame) -> list[str]:
    """content_fp per row = hash of query and every pre-label field the
    composition-and-gate pipeline reads. Cache status is deliberately absent."""
    cell_strs = ["|".join(sorted(cells)) for cells in rows["cells"]]
    return [
        hashlib.sha256(
            "\x1f".join((
                str(rows["dataset"].iloc[i]),
                str(rows["query_id"].iloc[i]),
                str(rows["query"].iloc[i]),
                str(rows["provenance"].iloc[i]),
                str(rows["operator"].iloc[i]),
                str(rows["family"].iloc[i]),
                str(rows["debt_id"].iloc[i]),
                str(rows["home_lane"].iloc[i]),
                cell_strs[i],
                str(rows["corruption_degree"].iloc[i]),
                *(str(rows[axis].iloc[i]) for axis in CORPUS_AXES),
                str(rows["answer_manifest_id"].iloc[i]),
            )).encode()
        ).hexdigest()[:16]
        for i in range(len(rows))
    ]


def _catalog_fingerprint(rows: pd.DataFrame, config: CatalogConfig) -> str:
    """Whole-catalog fp = sorted per-row fp digest + config digest. Two builds
    on the same catalog+config produce byte-identical fp (spec verification 12)."""
    row_material = "\n".join(sorted(rows["content_fp"].tolist()))
    config_material = repr(sorted(config.per_lane_cap.items())) + repr(
        (config.default_per_lane_cap, config.include_augmentation, config.seed)
    )
    return hashlib.sha256((row_material + "\x1f" + config_material).encode()).hexdigest()
