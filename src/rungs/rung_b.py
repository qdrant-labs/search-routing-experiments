"""Rung B: reporting-only validation over the completed planned set.

Membership is immutable — spec:302-305. Rung B may INSPECT post-label
outcomes but never ranks, optimizes, substitutes, or removes rows. Its only
failure modes are integrity conditions (missing rows, invalid fingerprints,
duplicate identities, broken answer manifests); quality judgments belong to
external acceptance (spec:319-322).
"""

from __future__ import annotations

import copy
import hashlib
import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass

import pandas as pd

ROUTES = ("dense_only", "pure_rrf", "sparse_only")
SCORE_COLUMNS = tuple(f"score_{route}" for route in ROUTES)


class RungBIntegrityError(RuntimeError):
    """The completed planned set fails an integrity precondition."""


@dataclass(frozen=True)
class RungBReport:
    """Ten metrics per spec:306-316. Every count is a scalar or a small
    marginal, so a report round-trips through JSON."""

    lane_representation: dict[str, int]
    lane_starvation: list[str]
    route_class_mix: dict[str, int]
    winner_mix: dict[str, int]
    winner_share: dict[str, float]
    outcome_shape_mix: dict[str, int]
    separation_summary: dict[str, float | int]
    breakdowns: dict[str, dict[str, dict[str, object]]]
    reference_comparisons: dict[str, dict[str, object]]
    margin_summary: dict[str, float]
    margin_distribution: list[float]
    judged_depth_distribution: dict[int, int]
    route_lane_coupling: dict[str, dict[str, int]]
    correctness_provenance: dict[str, int]
    cost_accounting: dict[str, int]
    execution_accounting: dict[str, object]
    all_zero_count: int
    missing_label_count: int
    counts: dict[str, int]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    def fingerprint(self) -> str:
        """Content identity of the exact assessment reviewed for acceptance."""
        payload = json.dumps(
            self.to_dict(),
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode()
        return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True)
class CompletedSet:
    """Frozen 1-to-1 join of planned_set and labels by durable identity."""

    rows: pd.DataFrame

    def __post_init__(self) -> None:
        missing = {"dataset", "query_id"} - set(self.rows.columns)
        if missing:
            raise RungBIntegrityError(f"completed set missing columns {sorted(missing)}")


class RungB:
    """Report over the completed planned set. Never mutates membership."""

    LANE_STARVATION_MIN: int = 5

    def __init__(
        self,
        *,
        decisive_margin: float = 0.4,
        tie_tolerance: float = 1e-9,
    ) -> None:
        if decisive_margin < 0:
            raise ValueError("decisive_margin must be non-negative")
        if tie_tolerance < 0:
            raise ValueError("tie_tolerance must be non-negative")
        self.decisive_margin = float(decisive_margin)
        self.tie_tolerance = float(tie_tolerance)

    def assess(
        self,
        planned_set: pd.DataFrame,
        labels: pd.DataFrame,
        *,
        cost_hits: int = 0,
        cost_misses: int = 0,
        references: Mapping[str, pd.DataFrame] | None = None,
        accounting: Mapping[str, object] | None = None,
    ) -> RungBReport:
        """Run the lightweight assessment over a frozen plan and its labels."""
        completed = self.join(
            planned_set,
            labels,
            cost_hits=cost_hits,
            cost_misses=cost_misses,
        )
        return self.report(
            completed,
            references=references,
            accounting=accounting,
        )

    def join(
        self,
        planned_set: pd.DataFrame,
        labels: pd.DataFrame,
        *,
        cost_hits: int = 0,
        cost_misses: int = 0,
    ) -> CompletedSet:
        """Exact 1-to-1 join on (dataset, query_id). Fails on any missing plan
        row, duplicate identity, or fp mismatch."""
        keys = list(zip(planned_set["dataset"].astype(str), planned_set["query_id"].astype(str)))
        if len(keys) != len(set(keys)):
            raise RungBIntegrityError("planned_set has duplicate (dataset, query_id) pairs")
        raw_label_keys = list(
            zip(labels["dataset"].astype(str), labels["query_id"].astype(str))
        )
        if len(raw_label_keys) != len(set(raw_label_keys)):
            raise RungBIntegrityError(
                "labels have duplicate (dataset, query_id) pairs; overlapping shards "
                "must be resolved before assessment"
            )
        label_keys = set(raw_label_keys)
        missing = [k for k in keys if k not in label_keys]
        if missing:
            raise RungBIntegrityError(
                f"{len(missing)} planned rows have no label — Rung B cannot run on incomplete set"
            )
        joined = planned_set.merge(
            labels,
            on=["dataset", "query_id"],
            how="left",
            validate="one_to_one",
            # Labels are the selected measurement. A legacy plan may carry a
            # stale copy of score/route columns, so overlapping plan fields
            # receive the suffix while label fields keep canonical names.
            suffixes=("_plan", ""),
            sort=False,
        )
        joined = joined.assign(_cost_hits=cost_hits, _cost_misses=cost_misses)
        return CompletedSet(rows=joined)

    def report(
        self,
        completed: CompletedSet,
        *,
        references: Mapping[str, pd.DataFrame] | None = None,
        accounting: Mapping[str, object] | None = None,
    ) -> RungBReport:
        rows = completed.rows
        measurements = _derive_measurements(
            rows,
            decisive_margin=self.decisive_margin,
            tie_tolerance=self.tie_tolerance,
        )
        lane_rep = rows["dataset"].astype(str).value_counts().to_dict()
        starvation = [lane for lane, count in lane_rep.items() if count < self.LANE_STARVATION_MIN]
        route_mix = _safe_value_counts(rows, "route_class")
        winner_mix = _winner_mix(measurements)
        winner_share = _mix_share(winner_mix, len(measurements))
        if not route_mix:
            route_mix = dict(winner_mix)
        kind_mix = {
            str(k): int(v)
            for k, v in measurements["_outcome_shape"].value_counts().to_dict().items()
        }
        margin_series = measurements["_margin"].dropna()
        margin_summary = _summary(margin_series) if not margin_series.empty else {}
        depth = pd.to_numeric(rows.get("depth"), errors="coerce") \
            if "depth" in rows.columns else pd.Series(dtype=float)
        depth_dist = depth.dropna().astype(int).value_counts().sort_index().to_dict() \
            if not depth.empty else {}
        route_lane = _cross_tab(rows, "route_class", "dataset")
        correctness = _safe_value_counts(rows, "scored_against")
        cost = {
            "hits": int(rows["_cost_hits"].iloc[0]) if len(rows) else 0,
            "misses": int(rows["_cost_misses"].iloc[0]) if len(rows) else 0,
        }
        execution_accounting = copy.deepcopy(dict(accounting or {}))
        execution_accounting.setdefault("cache", dict(cost))
        all_zero = int((measurements["_outcome_shape"] == "all_zero").sum())
        decisive_count = int(measurements["_decisive"].sum())
        decisive_winner_mix = _winner_mix(measurements.loc[measurements["_decisive"]])
        separation = {
            "threshold": self.decisive_margin,
            "decisive_count": decisive_count,
            "decisive_share": decisive_count / len(rows) if len(rows) else 0.0,
            "exact_tie_count": int((measurements["_outcome_shape"] == "all_tied").sum()),
            "all_zero_count": all_zero,
            "null_route_count": int((measurements["_winner"] == "null").sum()),
            "decisive_winner_mix": decisive_winner_mix,
            "decisive_winner_share": _mix_share(
                decisive_winner_mix,
                decisive_count,
            ),
        }
        breakdowns = _breakdowns(rows, measurements, self.decisive_margin)
        reference_comparisons = _reference_comparisons(
            rows,
            measurements,
            references or {},
            decisive_margin=self.decisive_margin,
            tie_tolerance=self.tie_tolerance,
        )
        missing = 0
        counts = {
            "planned": len(rows),
            "completed": int(rows[["dataset", "query_id"]].notna().all(axis=1).sum()),
        }
        return RungBReport(
            lane_representation={str(k): int(v) for k, v in lane_rep.items()},
            lane_starvation=[str(lane) for lane in starvation],
            route_class_mix=route_mix,
            winner_mix=winner_mix,
            winner_share=winner_share,
            outcome_shape_mix=kind_mix,
            separation_summary=separation,
            breakdowns=breakdowns,
            reference_comparisons=reference_comparisons,
            margin_summary=margin_summary,
            margin_distribution=[round(float(v), 12) for v in margin_series.tolist()],
            judged_depth_distribution={int(k): int(v) for k, v in depth_dist.items()},
            route_lane_coupling=route_lane,
            correctness_provenance=correctness,
            cost_accounting=cost,
            execution_accounting=execution_accounting,
            all_zero_count=all_zero,
            missing_label_count=missing,
            counts=counts,
        )


def _safe_value_counts(rows: pd.DataFrame, column: str) -> dict[str, int]:
    if column not in rows.columns:
        return {}
    return {str(k): int(v) for k, v in rows[column].astype(str).value_counts().to_dict().items()}


def _cross_tab(rows: pd.DataFrame, row_col: str, col_col: str) -> dict[str, dict[str, int]]:
    if row_col not in rows.columns or col_col not in rows.columns:
        return {}
    ct = pd.crosstab(rows[row_col].astype(str), rows[col_col].astype(str))
    return {r: {c: int(ct.loc[r, c]) for c in ct.columns} for r in ct.index}


def _summary(series: pd.Series) -> dict[str, float]:
    return {
        "n": int(len(series)),
        "mean": float(series.mean()),
        "std": float(series.std()) if len(series) > 1 else 0.0,
        "min": float(series.min()),
        "p25": float(series.quantile(0.25)),
        "p50": float(series.quantile(0.5)),
        "p75": float(series.quantile(0.75)),
        "max": float(series.max()),
    }


def _winner_mix(measurements: pd.DataFrame) -> dict[str, int]:
    counts = measurements["_winner"].value_counts().to_dict()
    return {route: int(counts.get(route, 0)) for route in (*ROUTES, "null")}


def _mix_share(mix: Mapping[str, int], total: int) -> dict[str, float]:
    return {
        route: mix[route] / total if total else 0.0
        for route in (*ROUTES, "null")
    }


def _breakdowns(
    rows: pd.DataFrame,
    measurements: pd.DataFrame,
    decisive_margin: float,
) -> dict[str, dict[str, dict[str, object]]]:
    dimensions = {
        "lane": _first_column(rows, "dataset"),
        "query_origin": _first_column(
            rows,
            "query_origin_plan",
            "query_origin",
            "provenance_plan",
            "provenance",
        ),
        "corpus_regime": _first_column(
            rows,
            "corpus_regime_plan",
            "corpus_regime",
            "scored_against",
        ),
        "trust_source": _first_column(
            rows,
            "trust_tier",
            "judgment_source",
            "qrel_source",
        ),
    }
    combined = rows.copy()
    for column in measurements.columns:
        combined[column] = measurements[column]

    result: dict[str, dict[str, dict[str, object]]] = {}
    for dimension, column in dimensions.items():
        if column is None:
            continue
        values = combined[column].where(combined[column].notna(), "unknown").astype(str)
        groups: dict[str, dict[str, object]] = {}
        for value in values.drop_duplicates().tolist():
            selected = measurements.loc[values == value]
            margins = selected["_margin"].dropna()
            shape_mix = {
                str(k): int(v)
                for k, v in selected["_outcome_shape"].value_counts().to_dict().items()
            }
            decisive_count = int(selected["_decisive"].sum())
            groups[value] = {
                "count": len(selected),
                "winner_mix": _winner_mix(selected),
                "outcome_shape_mix": shape_mix,
                "decisive_threshold": decisive_margin,
                "decisive_count": decisive_count,
                "decisive_share": decisive_count / len(selected),
                "all_zero_count": int((selected["_outcome_shape"] == "all_zero").sum()),
                "null_route_count": int((selected["_winner"] == "null").sum()),
                "margin_summary": _summary(margins) if not margins.empty else {},
            }
        result[dimension] = groups
    return result


def _first_column(rows: pd.DataFrame, *candidates: str) -> str | None:
    return next((column for column in candidates if column in rows.columns), None)


def _reference_comparisons(
    rows: pd.DataFrame,
    measurements: pd.DataFrame,
    references: Mapping[str, pd.DataFrame],
    *,
    decisive_margin: float,
    tie_tolerance: float,
) -> dict[str, dict[str, object]]:
    current = rows.loc[:, ["dataset", "query_id"]].astype(str).reset_index(drop=True)
    current_measurements = measurements.reset_index(drop=True).add_suffix("_current")
    current = pd.concat([current, current_measurements], axis=1)
    comparisons: dict[str, dict[str, object]] = {}

    for name, reference in references.items():
        _validate_reference(name, reference)
        normalized = reference.copy()
        normalized[["dataset", "query_id"]] = normalized[["dataset", "query_id"]].astype(str)
        reference_measurements = _derive_measurements(
            normalized,
            decisive_margin=decisive_margin,
            tie_tolerance=tie_tolerance,
        ).reset_index(drop=True).add_suffix("_reference")
        reference_frame = pd.concat(
            [normalized.loc[:, ["dataset", "query_id"]].reset_index(drop=True), reference_measurements],
            axis=1,
        )
        matched = current.merge(
            reference_frame,
            on=["dataset", "query_id"],
            how="inner",
            validate="one_to_one",
            sort=False,
        )
        current_metrics = _cohort_metrics(_side_measurements(matched, "current"))
        reference_metrics = _cohort_metrics(_side_measurements(matched, "reference"))
        comparisons[str(name)] = {
            "matched_count": len(matched),
            "current": current_metrics,
            "reference": reference_metrics,
            "delta": {
                "winner_share": {
                    route: current_metrics["winner_share"][route]
                    - reference_metrics["winner_share"][route]
                    for route in (*ROUTES, "null")
                },
                "decisive_share": current_metrics["decisive_share"]
                - reference_metrics["decisive_share"],
                "all_zero_share": current_metrics["all_zero_share"]
                - reference_metrics["all_zero_share"],
            },
        }
    return comparisons


def _validate_reference(name: str, reference: pd.DataFrame) -> None:
    missing = {"dataset", "query_id"} - set(reference.columns)
    if missing:
        raise RungBIntegrityError(f"reference {name!r} missing columns {sorted(missing)}")
    keys = list(zip(reference["dataset"].astype(str), reference["query_id"].astype(str)))
    if len(keys) != len(set(keys)):
        raise RungBIntegrityError(
            f"reference {name!r} has duplicate (dataset, query_id) pairs"
        )


def _side_measurements(matched: pd.DataFrame, side: str) -> pd.DataFrame:
    return matched.loc[
        :,
        [f"{column}_{side}" for column in ("_winner", "_outcome_shape", "_margin", "_decisive")],
    ].rename(columns=lambda column: column.removesuffix(f"_{side}"))


def _cohort_metrics(measurements: pd.DataFrame) -> dict[str, object]:
    count = len(measurements)
    mix = _winner_mix(measurements)
    return {
        "winner_share": {
            route: mix[route] / count if count else 0.0
            for route in (*ROUTES, "null")
        },
        "decisive_share": float(measurements["_decisive"].mean()) if count else 0.0,
        "all_zero_share": (
            float((measurements["_outcome_shape"] == "all_zero").mean())
            if count
            else 0.0
        ),
    }


def _derive_measurements(
    rows: pd.DataFrame,
    *,
    decisive_margin: float,
    tie_tolerance: float,
) -> pd.DataFrame:
    """Derive assessment-only views without changing the completed set."""
    derived = pd.DataFrame(index=rows.index)
    derived["_winner"] = _winner_series(rows)

    if all(column in rows.columns for column in SCORE_COLUMNS):
        scores = rows.loc[:, SCORE_COLUMNS].apply(pd.to_numeric, errors="coerce")
        ordered = pd.DataFrame(
            [sorted(values, reverse=True) for values in scores.to_numpy()],
            index=rows.index,
        )
        derived["_margin"] = ordered[0] - ordered[1]
        maximum = scores.max(axis=1)
        spread = maximum - scores.min(axis=1)
        derived["_outcome_shape"] = "routes_differ"
        derived.loc[spread <= tie_tolerance, "_outcome_shape"] = "all_tied"
        derived.loc[maximum <= tie_tolerance, "_outcome_shape"] = "all_zero"
    else:
        derived["_margin"] = (
            pd.to_numeric(rows["margin"], errors="coerce")
            if "margin" in rows.columns
            else pd.Series(float("nan"), index=rows.index)
        )
        derived["_outcome_shape"] = _shape_series(rows)

    derived["_decisive"] = (
        (derived["_outcome_shape"] == "routes_differ")
        & (derived["_margin"] >= decisive_margin)
    )
    return derived


def _winner_series(rows: pd.DataFrame) -> pd.Series:
    if "route" in rows.columns:
        winners = rows["route"]
    elif "route_class" in rows.columns:
        winners = rows["route_class"].replace(
            {"dense": "dense_only", "hybrid": "pure_rrf", "sparse": "sparse_only"}
        )
    else:
        winners = pd.Series(None, index=rows.index, dtype=object)
    return winners.where(winners.notna(), "null").astype(str)


def _shape_series(rows: pd.DataFrame) -> pd.Series:
    if "shape" in rows.columns:
        return rows["shape"].fillna("unknown").astype(str)
    if "kind" not in rows.columns:
        return pd.Series("unknown", index=rows.index, dtype=str)
    return rows["kind"].replace(
        {
            "decisive": "routes_differ",
            "undecisive": "routes_differ",
            "fake_tie": "routes_differ",
            "genuine_tie": "all_tied",
        }
    ).fillna("unknown").astype(str)
