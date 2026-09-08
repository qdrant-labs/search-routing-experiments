"""v4 composition: a greedy pre-label coverage selector over the all-query catalog.

Selects on diversity coverage alone — cells, lanes, corruption and corpus strata
— with no route labels, class quotas, or waste classes in the decision.
Superseded by `rungs.rung_a`, whose candidate universe is the registry rather
than the already-materialized lane snapshots this one draws from; kept as the
pre-rungs control arm to measure a rungs draw against, never to build from.
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass

import pandas as pd

from composition.pool_v3 import STRATA

DEFAULT_RHO = 0.5
DEFAULT_STRATUM_FLOOR = 25
# axes small enough that an equal target/len slice would exceed real supply keep
# a flat floor; larger axes scale their per-band floor down with member count.
_SCALE_MIN_MEMBERS = 10


@dataclass(frozen=True)
class RequirementCoverage:
    """One requirement's pre-label coverage: how much supply exists and how much
    the composition drew. `missing` is genuine composition/generation debt."""

    axis: str
    name: str
    target: int
    available: int
    selected: int

    @property
    def missing(self) -> int:
        return max(0, self.target - self.selected)


@dataclass(frozen=True)
class CoverageReport:
    """Measured outcome of one deterministic coverage composition — pre-label."""

    size: int
    target: int
    lanes_seen: int
    families_used: int
    provenance_mix: dict[str, int]
    requirements: list[RequirementCoverage]
    lane_capped: bool

    @property
    def shortages(self) -> list[RequirementCoverage]:
        return [r for r in self.requirements if r.missing > 0]


class ComposerV4:
    """Deterministic greedy coverage over one pre-label candidate catalog."""

    def compose(
        self,
        catalog: pd.DataFrame,
        target: int,
        *,
        rho: float = DEFAULT_RHO,
        kappa: float,
        stratum_floor: int = DEFAULT_STRATUM_FLOOR,
        on_progress: Callable[[int, int], None] | None = None,
    ) -> tuple[pd.DataFrame, CoverageReport]:
        """Select up to `target` rows maximizing even coverage of every
        requirement; return the selection and its coverage report."""
        assert 0 < rho <= 0.5, f"rho must be in (0, 0.5]; got {rho}"
        if target < 0 or not 0 < kappa <= 1:
            raise ValueError("target and kappa are outside their domains")

        source = catalog.reset_index(drop=True)
        row_id = self._row_ids(source).to_dict()
        lane_of = source["dataset"].astype(str).to_dict()
        family_of = self._families(source).to_dict()
        prov_of = source.get(
            "provenance", pd.Series("natural", index=source.index)
        ).astype(str).to_dict()

        strata, floors, axis_of = self._requirements(source, target, rho, stratum_floor)
        # member[req][lane] = unselected rows in that lane covering req.
        member: dict[tuple, dict[str, set[int]]] = {
            req: defaultdict(set) for req in floors
        }
        available: dict[tuple, int] = defaultdict(int)
        for i, reqs in strata.items():
            lane = lane_of[i]
            for req in reqs:
                member[req][lane].add(i)
                available[req] += 1

        cap = math.ceil(kappa * target)
        chosen: list[int] = []
        lane_count: defaultdict[str, int] = defaultdict(int)
        family_count: defaultdict[str, int] = defaultdict(int)
        prov_count: defaultdict[str, int] = defaultdict(int)
        n: defaultdict[tuple, int] = defaultdict(int)
        stalled: set[tuple] = set()
        # deficit[i]: how many still-under-floor requirements row i covers.
        # Recomputing it per candidate per pick is the greedy's hot path, so it
        # is cached and decremented for a requirement's remaining members the
        # instant it crosses its floor — floors only ever fill, so each
        # requirement triggers that bulk decrement at most once.
        deficit: dict[int, int] = {i: len(reqs) for i, reqs in strata.items()}
        log_every = max(1, target // 50)

        while len(chosen) < target and len(stalled) < len(floors):
            # most under-covered requirement with an admissible candidate; a
            # requirement whose supply is spent or wholly lane-capped stalls for
            # good (chosen rows never return, lane counts never fall).
            need = None
            for req in sorted(
                (r for r in floors if r not in stalled),
                key=lambda r: (n[r] / floors[r], r),
            ):
                lanes_open = [
                    lane for lane, rows in member[req].items()
                    if rows and lane_count[lane] < cap
                ]
                if lanes_open:
                    need = req
                    break
                stalled.add(req)
            if need is None:
                break

            # coverage first: the lane whose best row advances the most other
            # under-floor requirements; ties broken by lane balance. Novelty and
            # provenance balance decide the row inside the winning lane, so a
            # diversity floor is never sacrificed to a prettier distribution.
            best_in_lane = {
                lane: min(
                    member[need][lane],
                    key=lambda i: (
                        -deficit[i],
                        family_count[family_of[i]],
                        prov_count[prov_of[i]],
                        row_id[i],
                    ),
                )
                for lane in lanes_open
            }
            lane = min(
                lanes_open,
                key=lambda ln: (-deficit[best_in_lane[ln]], lane_count[ln], ln),
            )
            i = best_in_lane[lane]

            chosen.append(i)
            lane_count[lane_of[i]] += 1
            family_count[family_of[i]] += 1
            prov_count[prov_of[i]] += 1
            for req in strata[i]:
                bucket = member[req]
                bucket[lane_of[i]].discard(i)
                was_under = n[req] < floors[req]
                n[req] += 1
                if was_under and n[req] >= floors[req]:
                    # req just met — it no longer deepens any candidate's
                    # deficit; drop it from every remaining member once.
                    for lane_rows in bucket.values():
                        for j in lane_rows:
                            deficit[j] -= 1
            if on_progress is not None and len(chosen) % log_every == 0:
                on_progress(len(chosen), target)

        selected = source.loc[chosen].copy().reset_index(drop=True)
        requirements = sorted(
            (
                RequirementCoverage(
                    axis=axis_of[req],
                    name=req[1],
                    target=int(floors[req]),
                    available=available[req],
                    selected=n[req],
                )
                for req in floors
            ),
            key=lambda r: (r.axis, r.name),
        )
        lane_capped = len(chosen) < target and any(
            lane_count[lane] >= cap for lane in lane_count
        )
        report = CoverageReport(
            size=len(selected),
            target=target,
            lanes_seen=selected["dataset"].nunique() if len(selected) else 0,
            families_used=len({family_of[i] for i in chosen}),
            provenance_mix={k: int(v) for k, v in prov_count.items()},
            requirements=requirements,
            lane_capped=lane_capped,
        )
        return selected, report

    @staticmethod
    def _row_ids(frame: pd.DataFrame) -> pd.Series:
        if "row_id" in frame:
            return frame["row_id"].astype(str)
        return frame["dataset"].astype(str) + ":" + frame["query_id"].astype(str)

    @staticmethod
    def _families(frame: pd.DataFrame) -> pd.Series:
        """Generation family for anti-clumping; a natural row is its own family
        (null/blank lineage) so natural supply is never throttled by family
        balance."""
        own = frame["dataset"].astype(str) + ":" + frame["query_id"].astype(str)
        if "family" not in frame:
            return own
        fam = frame["family"]
        fam_str = fam.astype(str)
        valid = fam.notna() & ~fam_str.isin(("", "nan", "None"))
        return fam_str.where(valid, own)

    @staticmethod
    def _requirements(
        frame: pd.DataFrame, target: int, rho: float, stratum_floor: int
    ) -> tuple[dict[int, frozenset[tuple]], dict[tuple, float], dict[tuple, str]]:
        """Per-row requirement sets, per-requirement floors, and each
        requirement's axis. Axes: cell (multi), lane (dataset), and the STRATA
        bands. `unknown` bands are coverage misses — counted on no requirement,
        never floored (`pool_v3.STRATA` omits them by construction)."""
        cell_names = sorted({c for cells in frame["cells"] for c in cells})
        lane_names = sorted(frame["dataset"].astype(str).unique())
        axes: dict[str, tuple[str, ...]] = {
            "cell": tuple(cell_names),
            "lane": tuple(lane_names),
            **STRATA,
        }
        floors: dict[tuple, float] = {}
        axis_of: dict[tuple, str] = {}
        for axis, names in axes.items():
            count = len(names)
            # Only the cell axis scales its per-band floor with the target — it
            # is the fill's diversity driver. Lane and the corpus/corruption
            # strata keep a flat floor: a lane floor guarantees new-lane
            # representation without forcing every lane to the same large size,
            # which would also make its whole-lane candidate scan quadratic.
            scaled = axis == "cell" and count >= _SCALE_MIN_MEMBERS
            value = max(float(stratum_floor), rho * target / count) if scaled else float(stratum_floor)
            for name in names:
                floors[(axis, name)] = value
                axis_of[(axis, name)] = axis

        out: dict[int, frozenset[tuple]] = {}
        for i, row in frame.iterrows():
            reqs = {("cell", c) for c in row["cells"]}
            reqs.add(("lane", str(row["dataset"])))
            for axis in STRATA:
                band = str(row[axis])
                if band in STRATA[axis]:
                    reqs.add((axis, band))
            out[i] = frozenset(reqs)
        return out, floors, axis_of


if __name__ == "__main__":
    from pathlib import Path

    from composition.recipe import Recipe

    catalog_path = (
        Path(__file__).resolve().parents[1] / "data" / "v4" / "catalog_v4.parquet"
    )
    catalog = pd.read_parquet(catalog_path)
    recipe = Recipe()
    _, result = ComposerV4().compose(
        catalog,
        target=5000,
        kappa=recipe.target_lane_share,
        stratum_floor=recipe.stratum_floor,
    )
    print(
        f"|D|={result.size:,} lanes={result.lanes_seen} "
        f"shortages={len(result.shortages)} lane_capped={result.lane_capped}"
    )
