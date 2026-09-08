"""Shared synthetic-catalog helpers for the Rung A/B verification suite.

Every catalog row is hand-shaped so tests target the algorithm's invariants,
not the extractor's. Uses no fastembed, no spaCy, no wordfreq.
"""

from __future__ import annotations

import pandas as pd

from rungs.rung_a import RungAConfig


def make_row(
    row_id: str,
    dataset: str,
    query: str,
    cells: list[str],
    *,
    corruption: str = "clean",
    corpus_idf: str = "unknown",
    corpus_oov: str = "unknown",
    corpus_pmi: str = "unknown",
    provenance: str = "natural",
    operator: str | None = None,
    family: str | None = None,
    debt_id: str = "",
    answer_manifest_id: str | None = None,
    content_fp: str | None = None,
) -> dict[str, object]:
    return {
        "row_id": row_id,
        "dataset": dataset,
        "query_id": row_id.split(":", 1)[1],
        "query": query,
        "provenance": provenance,
        "operator": operator,
        "family": family or row_id,
        "debt_id": debt_id,
        "home_lane": dataset,
        "cells": list(cells),
        "corruption_degree": corruption,
        "corpus_idf": corpus_idf,
        "corpus_oov": corpus_oov,
        "corpus_pmi": corpus_pmi,
        "answer_covered": True,
        "answer_manifest_id": answer_manifest_id or f"mf:{row_id}",
        "content_fp": content_fp or f"fp:{row_id}",
    }


def make_catalog(rows: list[dict[str, object]]) -> pd.DataFrame:
    return pd.DataFrame(rows)


def default_config(
    catalog: pd.DataFrame,
    *,
    ceiling: int = 20,
    theta0: float = 0.7,
    theta_step: float = 0.1,
    floor_by_cell: int = 2,
    floor_by_lane: int = 1,
    lane_cap: int = 100,
) -> RungAConfig:
    cells = sorted({c for cs in catalog["cells"] for c in cs})
    floors: dict[tuple[str, str], int] = {}
    for cell in cells:
        floors[("cell", cell)] = floor_by_cell
    for lane in sorted(catalog["dataset"].astype(str).unique()):
        floors[("lane", lane)] = floor_by_lane
    return RungAConfig(
        planned_size_ceiling=ceiling,
        floors=floors,
        lane_budgets={lane: lane_cap for lane in catalog["dataset"].astype(str).unique()},
        theta0=theta0,
        theta_step=theta_step,
    )
