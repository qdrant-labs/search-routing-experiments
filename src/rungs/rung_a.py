"""Rung A: label-blind, deterministic composer over the version-neutral
candidate catalog.

Reads only pre-label fields — never a route, margin, depth, or label
provenance. Cache status is deliberately absent from the trace: an empty
cache and a full cache must produce byte-identical artifacts (spec:274).
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import defaultdict
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd

from composition.strata import CORPUS_AXES, STRATA

Axis = str
Name = str
StratumKey = tuple[Axis, Name]

_TOKEN = re.compile(r"[A-Za-z0-9]+")

# Axes whose bands are cheap pre-label strata and therefore eligible for floor
# declaration. Cell and lane are declared separately by the caller because
# their member sets are catalog-dependent.
_BAND_AXES: tuple[str, ...] = ("corruption_degree", *CORPUS_AXES)


@dataclass(frozen=True)
class RungAConfig:
    """One run's dials. Every value is an explicit input — spec:180-183 forbids
    the composer from carrying θ0/θ_step/ρ/κ as implementation constants."""

    planned_size_ceiling: int
    floors: dict[StratumKey, int]
    lane_budgets: dict[str, int]
    theta0: float
    theta_step: float
    residual_fill: bool = False
    seed: int = 0

    def __post_init__(self) -> None:
        if not 0.0 <= self.theta0 <= 1.0:
            raise ValueError(f"theta0 must be in [0,1]; got {self.theta0}")
        if not 0.0 < self.theta_step <= 1.0:
            raise ValueError(f"theta_step must be in (0,1]; got {self.theta_step}")
        if self.planned_size_ceiling < 0:
            raise ValueError(f"planned_size_ceiling must be >=0; got {self.planned_size_ceiling}")


@dataclass(frozen=True)
class DebtLine:
    """One unreachable or exhausted floor. `debt_id` is the durable name of the
    shortage — generation stamps it onto the rows it produces, so the spend
    gate can group a generated row back to the deficit that asked for it."""

    debt_id: str
    axis: str
    name: str
    floor: int
    admissible_supply: int
    missing: int
    reason: str


@dataclass(frozen=True)
class CoverageLine:
    axis: str
    name: str
    floor: int
    pool: int
    achievable: int
    selected: int


@dataclass(frozen=True)
class CoverageReport:
    """`pool -> achievable -> planned -> completed` for every stratum plus
    build-level totals (spec:316)."""

    planned_size: int
    ceiling: int
    lanes_used: int
    families_used: int
    provenance_mix: dict[str, int]
    strata: list[CoverageLine]


@dataclass(frozen=True)
class RungAArtifacts:
    """The atomic output set (spec:264-271). `provenance` carries every fp the
    inputs stamp; `debt` is the generation queue the loop consumes next."""

    planned_set: pd.DataFrame
    selection_trace: pd.DataFrame
    coverage_report: CoverageReport
    generation_debt: pd.DataFrame
    provenance: dict[str, object]

    def write(self, out_dir: Path) -> None:
        out_dir.mkdir(parents=True, exist_ok=True)
        _atomic_parquet(self.planned_set, out_dir / "planned_set.parquet")
        _atomic_parquet(self.selection_trace, out_dir / "selection_trace.parquet")
        _atomic_parquet(self.generation_debt, out_dir / "generation_debt.parquet")
        _atomic_text(
            out_dir / "coverage_report.json",
            json.dumps(asdict(self.coverage_report), indent=2, default=str) + "\n",
        )
        _atomic_text(
            out_dir / "provenance.json",
            json.dumps(self.provenance, indent=2, default=str, sort_keys=True) + "\n",
        )


@dataclass
class _SelectionState:
    """Mutable, exact scoring state kept out of the pandas hot path.

    ``max_jaccard`` is updated after each pick for every same-lane candidate
    sharing a token with the picked row. Candidates with no shared token retain
    zero, which is exactly their maximum Jaccard similarity.
    """

    row_ids: list[str]
    lanes: list[str]
    families: list[str]
    provenances: list[str]
    strata: list[frozenset[StratumKey]]
    tokens: list[frozenset[str]]
    token_members: dict[str, dict[str, set[int]]]
    selected_counts: defaultdict[StratumKey, int]
    max_jaccard: list[float]

    @classmethod
    def from_candidates(cls, candidates: pd.DataFrame) -> _SelectionState:
        lanes = [str(value) for value in candidates["dataset"].tolist()]
        tokens = candidates["_tokens"].tolist()
        token_members: dict[str, dict[str, set[int]]] = defaultdict(lambda: defaultdict(set))
        for i, (lane, row_tokens) in enumerate(zip(lanes, tokens, strict=True)):
            for token in row_tokens:
                token_members[lane][token].add(i)
        return cls(
            row_ids=[str(value) for value in candidates["row_id"].tolist()],
            lanes=lanes,
            families=[str(value) for value in candidates["family"].tolist()],
            provenances=[str(value) for value in candidates["provenance"].tolist()],
            strata=candidates["_strata"].tolist(),
            tokens=tokens,
            token_members={lane: dict(index) for lane, index in token_members.items()},
            selected_counts=defaultdict(int),
            max_jaccard=[0.0] * len(candidates),
        )

    def novelty(self, i: int) -> float:
        return 1.0 - self.max_jaccard[i] if self.tokens[i] else 1.0

    def gain(self, i: int, config: RungAConfig) -> float:
        return _structural_gain_from_counts(self.strata[i], self.selected_counts, config)

    def record_pick(self, i: int) -> None:
        """Update sufficient statistics after selecting candidate ``i``."""
        for key in self.strata[i]:
            self.selected_counts[key] += 1
        tokens = self.tokens[i]
        if not tokens:
            return
        affected: set[int] = set()
        lane_index = self.token_members.get(self.lanes[i], {})
        for token in tokens:
            affected.update(lane_index.get(token, ()))
        for other_i in affected:
            other_tokens = self.tokens[other_i]
            similarity = len(tokens & other_tokens) / len(tokens | other_tokens)
            if similarity > self.max_jaccard[other_i]:
                self.max_jaccard[other_i] = similarity


class RungA:
    """Compose a planned set from an admissible candidate catalog. Owns no
    state: one instance may be reused across smoke and full runs."""

    def compose(
        self,
        catalog: pd.DataFrame,
        config: RungAConfig,
        *,
        catalog_fp: str,
        on_progress: Callable[[int, int], None] | None = None,
    ) -> RungAArtifacts:
        self._assert_shape(catalog)
        candidates = _prepare(catalog)
        state = _SelectionState.from_candidates(candidates)

        supply = _admissible_supply(candidates, config)
        debt, reachable_floors = _split_debt(config.floors, supply)
        n: dict[StratumKey, int] = defaultdict(int)
        # membership indexed by stratum -> lane -> {row_id}, so the selector can
        # cheaply filter to lanes with budget remaining
        member: dict[StratumKey, dict[str, set[int]]] = {
            key: defaultdict(set) for key in reachable_floors
        }
        for i, row_strata in enumerate(state.strata):
            for key in row_strata:
                if key in member:
                    member[key][state.lanes[i]].add(i)

        lane_used: defaultdict[str, int] = defaultdict(int)
        family_used: defaultdict[str, int] = defaultdict(int)
        prov_used: defaultdict[str, int] = defaultdict(int)
        chosen: list[int] = []
        trace: list[dict[str, object]] = []
        stalled: set[StratumKey] = set()
        log_every = max(1, config.planned_size_ceiling // 50 or 1)

        while (
            len(chosen) < config.planned_size_ceiling
            and len(stalled) < len(reachable_floors)
        ):
            weakest = _weakest_active(reachable_floors, n, stalled, member, lane_used, config.lane_budgets)
            if weakest is None:
                break
            pick = _pick_for(
                weakest,
                member,
                lane_used,
                family_used,
                state=state,
                selected_count=len(chosen),
                config=config,
            )
            if pick is None:
                stalled.add(weakest)
                continue
            i, gain, novelty, provenance_score, theta_used, reason = pick
            chosen.append(i)
            lane_used[state.lanes[i]] += 1
            family_used[state.families[i]] += 1
            prov_used[state.provenances[i]] += 1
            state.record_pick(i)
            for key in state.strata[i]:
                if key in member:
                    member[key][state.lanes[i]].discard(i)
                    n[key] += 1
            trace.append({
                "row_id": state.row_ids[i],
                "stratum_axis": weakest[0],
                "stratum_name": weakest[1],
                "gain": gain,
                "novelty": novelty,
                "provenance_score": provenance_score,
                "theta_effective": theta_used,
                "lane": state.lanes[i],
                "family": state.families[i],
                "reason": reason,
                "rank": len(chosen),
            })
            if on_progress is not None and len(chosen) % log_every == 0:
                on_progress(len(chosen), config.planned_size_ceiling)

        if config.residual_fill and len(chosen) < config.planned_size_ceiling:
            _residual_fill(
                candidates, state, chosen, trace, config, lane_used, family_used, prov_used
            )

        selected = candidates.iloc[chosen].reset_index(drop=True)
        planned_set = selected.drop(columns=["_strata", "_tokens"], errors="ignore")
        # ensure deterministic column and row order — the byte-identical invariant
        planned_set = planned_set.sort_values("row_id", kind="stable").reset_index(drop=True)
        trace_df = pd.DataFrame(trace, columns=[
            "rank", "row_id", "stratum_axis", "stratum_name",
            "gain", "novelty", "provenance_score", "theta_effective",
            "lane", "family", "reason",
        ])
        coverage = _coverage_report(
            candidates, chosen, config, supply, n, lane_used, prov_used, family_used
        )
        provenance = {
            "catalog_fp": catalog_fp,
            "config_fp": _config_fp(config),
            "planned_size": len(chosen),
            "debt_count": len(debt),
            "reachable_floors": len(reachable_floors),
            "unreachable_floors": len(debt),
        }
        return RungAArtifacts(
            planned_set=planned_set,
            selection_trace=trace_df,
            coverage_report=coverage,
            generation_debt=_debt_frame(debt),
            provenance=provenance,
        )

    @staticmethod
    def _assert_shape(catalog: pd.DataFrame) -> None:
        required = {"row_id", "dataset", "query_id", "query", "provenance", "family", "cells"}
        missing = required - set(catalog.columns)
        if missing:
            raise ValueError(f"catalog missing required columns: {sorted(missing)}")
        for forbidden in (
            "rung", "version", "route_class", "route_class_any", "margin",
            "oracle", "route_scores", "route_rankings", "kind", "depth",
            "certified", "is_decisive", "is_hybrid", "is_waste", "winner",
        ):
            if forbidden in catalog.columns:
                raise ValueError(
                    f"catalog carries post-label column {forbidden!r} — Rung A refuses to consume it"
                )


# ---------------------------------------------------------------- prepare ---


def _prepare(catalog: pd.DataFrame) -> pd.DataFrame:
    """Attach `_strata` (frozenset of (axis, name)) and `_tokens` (set) to each
    row. Stable sort by row_id keeps every downstream tie-break deterministic."""
    frame = catalog.sort_values("row_id", kind="stable").reset_index(drop=True).copy()
    strata: list[frozenset[StratumKey]] = []
    for _, row in frame.iterrows():
        keys: set[StratumKey] = {("cell", c) for c in row["cells"]}
        keys.add(("lane", str(row["dataset"])))
        for axis in _BAND_AXES:
            if axis in frame.columns:
                band = row[axis]
                if isinstance(band, str) and band in STRATA.get(axis, ()):
                    keys.add((axis, band))
        strata.append(frozenset(keys))
    frame["_strata"] = strata
    frame["_tokens"] = [frozenset(_TOKEN.findall(str(q).lower())) for q in frame["query"]]
    return frame


# ---------------------------------------------------------- reachability ---


def _admissible_supply(
    candidates: pd.DataFrame, config: RungAConfig
) -> dict[StratumKey, int]:
    """Supply per stratum after lane budgets clip each lane's contribution."""
    supply: defaultdict[StratumKey, int] = defaultdict(int)
    per_lane_seen: defaultdict[tuple[StratumKey, str], int] = defaultdict(int)
    for _, row in candidates.iterrows():
        lane = str(row["dataset"])
        budget = config.lane_budgets.get(lane, config.planned_size_ceiling)
        for key in row["_strata"]:
            if per_lane_seen[(key, lane)] < budget:
                per_lane_seen[(key, lane)] += 1
                supply[key] += 1
    return dict(supply)


def _split_debt(
    floors: dict[StratumKey, int], supply: dict[StratumKey, int]
) -> tuple[list[DebtLine], dict[StratumKey, int]]:
    debt: list[DebtLine] = []
    reachable: dict[StratumKey, int] = {}
    for key, floor in floors.items():
        s = supply.get(key, 0)
        if s < floor:
            debt.append(DebtLine(
                debt_id=f"{key[0]}:{key[1]}",
                axis=key[0], name=key[1], floor=int(floor), admissible_supply=int(s),
                missing=int(floor - s), reason="unreachable_supply",
            ))
        else:
            reachable[key] = int(floor)
    return debt, reachable


# ------------------------------------------------------------- selection ---


def _weakest_active(
    floors: dict[StratumKey, int],
    n: dict[StratumKey, int],
    stalled: set[StratumKey],
    member: dict[StratumKey, dict[str, set[int]]],
    lane_used: dict[str, int],
    lane_budgets: dict[str, int],
) -> StratumKey | None:
    """Weakest reachable stratum with at least one lane whose budget still
    admits its members. A stratum whose lanes are all exhausted returns None
    at pick time and joins `stalled`."""
    open_keys = [k for k in floors if k not in stalled and n[k] < floors[k]]
    if not open_keys:
        return None
    open_keys.sort(key=lambda k: (n[k] / floors[k], k))
    for key in open_keys:
        for lane, rows in member[key].items():
            if rows and lane_used.get(lane, 0) < lane_budgets.get(lane, float("inf")):
                return key
    return None


def _pick_for(
    stratum: StratumKey,
    member: dict[StratumKey, dict[str, set[int]]],
    lane_used: dict[str, int],
    family_used: dict[str, int],
    *,
    state: _SelectionState,
    selected_count: int,
    config: RungAConfig,
) -> tuple[int, float, float, float, float, str] | None:
    """Score once, then relax the novelty threshold without rescoring.

    Gain, novelty, and provenance state are unchanged until the next pick, so
    each candidate has one stable score throughout the relaxation sequence.
    """
    eligible: list[int] = [
        i for lane, rows in member[stratum].items()
        if lane_used.get(lane, 0) < config.lane_budgets.get(lane, float("inf"))
        for i in sorted(rows)
    ]
    if not eligible:
        return None

    scored = [
        (
            i,
            state.gain(i, config),
            state.novelty(i),
            _provenance_score_for_family(state.families[i], family_used, selected_count),
        )
        for i in eligible
    ]
    theta = config.theta0
    max_novelty = max(novelty for _, _, novelty, _ in scored)
    while theta > 0.0 and max_novelty < theta:
        theta = max(0.0, theta - config.theta_step)

    best_i, gain, novelty, provenance_score = min(
        (
            (i, gain, novelty, provenance_score)
            for i, gain, novelty, provenance_score in scored
            if theta == 0.0 or novelty >= theta
        ),
        key=lambda score: (-score[1], -score[2], -score[3], state.row_ids[score[0]]),
    )
    return best_i, float(gain), float(novelty), float(provenance_score), float(theta), "coverage"


def _structural_gain(
    row: pd.Series, chosen: pd.DataFrame | None, config: RungAConfig
) -> float:
    """G(q|D) = max over stratum(q) of max(0, 1 - n_s(D)/f_s) (spec:206-208)."""
    strata = row["_strata"]
    if not strata:
        return 0.0
    if chosen is None or len(chosen) == 0:
        counts: dict[StratumKey, int] = {}
    else:
        counts = defaultdict(int)
        for other_strata in chosen["_strata"]:
            for key in other_strata:
                counts[key] += 1
    best = 0.0
    for key in strata:
        floor = config.floors.get(key)
        if floor is None or floor == 0:
            continue
        deficit = max(0.0, 1.0 - counts.get(key, 0) / floor)
        if deficit > best:
            best = deficit
    return best


def _structural_gain_from_counts(
    strata: frozenset[StratumKey], counts: dict[StratumKey, int], config: RungAConfig
) -> float:
    """Exact structural gain using the incremental selected-strata ledger."""
    best = 0.0
    for key in strata:
        floor = config.floors.get(key)
        if floor is None or floor == 0:
            continue
        deficit = max(0.0, 1.0 - counts.get(key, 0) / floor)
        if deficit > best:
            best = deficit
    return best


def _novelty(row: pd.Series, candidates: pd.DataFrame, chosen_set: set[int]) -> float:
    """N(q|D) = 1 - max Jaccard(q, x) over same-lane x in D. Empty context -> 1."""
    if not chosen_set:
        return 1.0
    lane = row["dataset"]
    tokens = row["_tokens"]
    if not tokens:
        return 1.0
    peer_tokens = [
        candidates.iloc[j]["_tokens"]
        for j in chosen_set
        if candidates.iloc[j]["dataset"] == lane
    ]
    if not peer_tokens:
        return 1.0
    max_j = 0.0
    for other in peer_tokens:
        if not other:
            continue
        j = len(tokens & other) / len(tokens | other)
        if j > max_j:
            max_j = j
    return 1.0 - max_j


def _provenance_score(row: pd.Series, family_used: dict[str, int], total: int) -> float:
    """Π(q|D) = 1 - count(fam(q), D) / max(1, |D|) (spec:233-235)."""
    return _provenance_score_for_family(str(row["family"]), family_used, total)


def _provenance_score_for_family(family: str, family_used: dict[str, int], total: int) -> float:
    denom = max(1, total)
    return 1.0 - family_used.get(family, 0) / denom


def _residual_fill(
    candidates: pd.DataFrame,
    state: _SelectionState,
    chosen: list[int],
    trace: list[dict[str, object]],
    config: RungAConfig,
    lane_used: dict[str, int],
    family_used: dict[str, int],
    prov_used: dict[str, int],
) -> None:
    """Fill remaining capacity by novelty > underrepresented provenance > lane
    need > row_id (spec:257-259). Uses no label-derived quantity."""
    chosen_set = set(chosen)
    remaining = [i for i in range(len(candidates)) if i not in chosen_set]
    for i in sorted(remaining, key=lambda k: state.row_ids[k]):
        if len(chosen) >= config.planned_size_ceiling:
            break
        lane = state.lanes[i]
        if lane_used.get(lane, 0) >= config.lane_budgets.get(lane, float("inf")):
            continue
        novelty = state.novelty(i)
        provenance_score = _provenance_score_for_family(state.families[i], family_used, len(chosen))
        chosen.append(i)
        chosen_set.add(i)
        lane_used[lane] += 1
        family_used[state.families[i]] += 1
        prov_used[state.provenances[i]] += 1
        state.record_pick(i)
        trace.append({
            "row_id": state.row_ids[i],
            "stratum_axis": "residual",
            "stratum_name": "residual",
            "gain": 0.0,
            "novelty": float(novelty),
            "provenance_score": float(provenance_score),
            "theta_effective": 0.0,
            "lane": lane,
            "family": state.families[i],
            "reason": "residual_fill",
            "rank": len(chosen),
        })


# --------------------------------------------------------------- reports ---


def _coverage_report(
    candidates: pd.DataFrame,
    chosen: list[int],
    config: RungAConfig,
    supply: dict[StratumKey, int],
    n: dict[StratumKey, int],
    lane_used: dict[str, int],
    prov_used: dict[str, int],
    family_used: dict[str, int],
) -> CoverageReport:
    pool_size = defaultdict(int)
    for _, row in candidates.iterrows():
        for key in row["_strata"]:
            pool_size[key] += 1
    lines = sorted(
        (
            CoverageLine(
                axis=key[0],
                name=key[1],
                floor=int(config.floors[key]),
                pool=int(pool_size.get(key, 0)),
                achievable=int(supply.get(key, 0)),
                selected=int(n.get(key, 0)),
            )
            for key in config.floors
        ),
        key=lambda line: (line.axis, line.name),
    )
    return CoverageReport(
        planned_size=len(chosen),
        ceiling=config.planned_size_ceiling,
        lanes_used=sum(1 for v in lane_used.values() if v > 0),
        families_used=sum(1 for v in family_used.values() if v > 0),
        provenance_mix={k: int(v) for k, v in prov_used.items()},
        strata=lines,
    )


def _debt_frame(debt: list[DebtLine]) -> pd.DataFrame:
    if not debt:
        return pd.DataFrame(
            columns=[
                "debt_id", "axis", "name", "floor",
                "admissible_supply", "missing", "reason",
            ]
        )
    return pd.DataFrame([asdict(line) for line in debt])


def _config_fp(config: RungAConfig) -> str:
    material = json.dumps({
        "planned_size_ceiling": config.planned_size_ceiling,
        "floors": sorted((f"{k[0]}::{k[1]}", int(v)) for k, v in config.floors.items()),
        "lane_budgets": sorted(config.lane_budgets.items()),
        "theta0": config.theta0,
        "theta_step": config.theta_step,
        "residual_fill": config.residual_fill,
        "seed": config.seed,
    }, sort_keys=True)
    return hashlib.sha256(material.encode()).hexdigest()


def _atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    try:
        frame.to_parquet(tmp, index=False)
        tmp.replace(path)
    finally:
        if tmp.exists():
            tmp.unlink()


def _atomic_text(path: Path, text: str) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    try:
        tmp.write_text(text)
        tmp.replace(path)
    finally:
        if tmp.exists():
            tmp.unlink()
