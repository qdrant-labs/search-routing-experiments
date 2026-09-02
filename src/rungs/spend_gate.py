"""Pre-label spend gate: a read-only predicate over a frozen Rung A plan.

Answers "is this batch worth paying to label?" from pre-label data only — it
never selects, scores a candidate, or reads a route/margin/label. FAIL blocks
labeling; concentration and novelty remain raw facts until their thresholds are
measured (spec: pre-label-spend-gate-design).
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass

import pandas as pd

from rungs.rung_a import CoverageReport


@dataclass(frozen=True)
class FloorStatus:
    axis: str
    name: str
    floor: int
    achievable: int
    selected: int


@dataclass(frozen=True)
class AugmentationFunnelLine:
    debt_id: str
    admitted: int
    selected: int


@dataclass(frozen=True)
class NoveltySummary:
    theta0: float
    p10: float
    p50: float
    p90: float
    share_below_theta0: float


@dataclass(frozen=True)
class PredictedYield:
    status: str
    reason: str


@dataclass(frozen=True)
class SpendGateReport:
    """One deeply immutable spend decision with a stable content identity."""

    rows_planned: int
    reachable_floors_met: int
    reachable_floors_unmet: tuple[FloorStatus, ...]
    unreachable_floors: tuple[FloorStatus, ...]
    rows_answer_uncovered: int
    novelty: NoveltySummary | None
    effective_families: float
    largest_family_share: float
    largest_operator_share: float
    augmentation_funnel: tuple[AugmentationFunnelLine, ...]
    warnings: tuple[str, ...]
    predicted_yield: PredictedYield
    assessment_complete: bool
    verdict: str
    failed_conditions: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    def fingerprint(self) -> str:
        payload = json.dumps(
            self.to_dict(), sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
        return hashlib.sha256(payload).hexdigest()


class SpendGate:
    """Assess a frozen plan before paid labeling. Owns no state."""

    def assess(
        self,
        planned_set: pd.DataFrame,
        selection_trace: pd.DataFrame,
        coverage_report: CoverageReport,
        *,
        catalog: pd.DataFrame | None = None,
        theta0: float,
    ) -> SpendGateReport:
        if not math.isfinite(theta0) or not 0.0 <= theta0 <= 1.0:
            raise ValueError(f"theta0 must be finite and in [0,1]; got {theta0}")
        met, unmet, unreachable = _floor_status(coverage_report)
        uncovered = _answer_uncovered(planned_set)
        funnel = _augmentation_funnel(catalog, planned_set)

        failed: list[str] = []
        for line in unmet:
            failed.append(
                f"reachable floor {line.axis}:{line.name} unmet "
                f"({line.selected}/{line.floor}, {line.achievable} available)"
            )
        if uncovered:
            failed.append(f"{uncovered} planned rows lack answer coverage")
        assessment_complete = catalog is not None and "debt_id" in catalog.columns
        if not assessment_complete:
            failed.append("admitted candidate catalog with debt_id is required for assessment")

        family_share, n_eff = _concentration(planned_set["family"])
        operator_share, _ = _concentration(planned_set["operator"])

        return SpendGateReport(
            rows_planned=len(planned_set),
            reachable_floors_met=met,
            reachable_floors_unmet=unmet,
            unreachable_floors=unreachable,
            rows_answer_uncovered=uncovered,
            novelty=_novelty_summary(selection_trace, theta0),
            effective_families=n_eff,
            largest_family_share=family_share,
            largest_operator_share=operator_share,
            augmentation_funnel=funnel,
            warnings=(),
            predicted_yield=PredictedYield(
                status="not_activated",
                reason="held-out-lane estimator not yet measured",
            ),
            assessment_complete=assessment_complete,
            verdict="FAIL" if failed else "PASS",
            failed_conditions=tuple(failed),
        )


def _floor_status(
    coverage: CoverageReport,
) -> tuple[int, tuple[FloorStatus, ...], tuple[FloorStatus, ...]]:
    """A floor is reachable when admissible supply covers it. A reachable floor
    left unfilled is a spend failure; an unreachable one is debt, not a fault.

    `selected` undercounts when Rung A ran residual fill (`n` is not advanced
    there), so a residual-filled floor may read as unmet. The gate's own runs
    keep residual_fill off, so this is exact in practice.
    """
    met = 0
    unmet: list[FloorStatus] = []
    unreachable: list[FloorStatus] = []
    for line in coverage.strata:
        row = FloorStatus(
            axis=line.axis,
            name=line.name,
            floor=line.floor,
            achievable=line.achievable,
            selected=line.selected,
        )
        if line.achievable < line.floor:
            unreachable.append(row)
        elif line.selected < line.floor:
            unmet.append(row)
        else:
            met += 1
    return met, tuple(unmet), tuple(unreachable)


def _augmentation_funnel(
    catalog: pd.DataFrame | None, planned_set: pd.DataFrame
) -> tuple[AugmentationFunnelLine, ...]:
    """Report admitted versus selected generated rows by shortage identity.

    Natural rows carry `debt_id == ""` and are excluded. Skipped entirely when
    the catalog is not supplied; the caller marks that assessment incomplete.
    The funnel is informational: current floor status owns the spend verdict.
    """
    if catalog is None or "debt_id" not in catalog.columns:
        return ()
    admitted = catalog.loc[catalog["debt_id"].astype(str) != "", "debt_id"].astype(str)
    if admitted.empty:
        return ()
    selected = (
        planned_set["debt_id"].astype(str)
        if "debt_id" in planned_set.columns
        else pd.Series(dtype=str)
    )
    selected_counts = selected[selected != ""].value_counts()
    admitted_counts = admitted.value_counts()
    return tuple(
        AugmentationFunnelLine(
            debt_id=str(debt_id),
            admitted=int(admitted_counts[debt_id]),
            selected=int(selected_counts.get(debt_id, 0)),
        )
        for debt_id in sorted(admitted_counts.index)
    )


def _answer_uncovered(planned_set: pd.DataFrame) -> int:
    covered = planned_set["answer_covered"].fillna(False).astype(bool)
    manifest = planned_set["answer_manifest_id"].notna() & (
        planned_set["answer_manifest_id"].astype(str) != ""
    )
    return int((~(covered & manifest)).sum())


def _concentration(series: pd.Series) -> tuple[float, float]:
    """Largest single share and inverse-Simpson effective count over a family or
    operator column. Nulls (natural rows carry no operator) are ignored."""
    counts = series.dropna().value_counts()
    total = int(counts.sum())
    if total == 0:
        return 0.0, 0.0
    shares = counts / total
    largest = float(shares.iloc[0])
    n_eff = float(1.0 / (shares**2).sum())
    return largest, n_eff


def _novelty_summary(
    selection_trace: pd.DataFrame, theta0: float
) -> NoveltySummary | None:
    if "novelty" not in selection_trace.columns or selection_trace.empty:
        return None
    novelty = pd.to_numeric(selection_trace["novelty"], errors="coerce").dropna()
    if novelty.empty:
        return None
    return NoveltySummary(
        theta0=theta0,
        p10=float(novelty.quantile(0.10)),
        p50=float(novelty.quantile(0.50)),
        p90=float(novelty.quantile(0.90)),
        share_below_theta0=float((novelty < theta0).mean()),
    )
