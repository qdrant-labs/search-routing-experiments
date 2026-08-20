"""The campaign — d42(a)'s outer while over the order sheet.

One turn of the crank: walk every hungry floor most-hungry-first and
dispatch each servable one through the AugmentationLoop. Gate-free floors
produce to the floor's need; gated floors stage only a `pilot_n` audit
sample (more would be paid feature-stock), and floors whose pilot is
already in the pool are skipped — re-running a campaign never re-pays.

The crank deliberately stops before admission and mints nothing from
scratch: admission belongs to whichever composition owns the sheet being
served, the synthetic rung belongs to `run_v3_generation`'s stage 5, and the
gated floors' credit waits on the human audits between cranks.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field

import pandas as pd
from tqdm.auto import tqdm

from augmentation.config import FAULT_STREAK, MAX_CHANCES
from augmentation.core import CreditGate
from augmentation.judge import CoherenceJudge
from augmentation.loop import AugmentationLoop, NeedsSelection
from composition.cells import CELLS_BY_NAME

PRODUCE = "produce"
SKIP_STAGED = "skip: pilot staged"
SKIP_NO_OPERATOR = "skip: no operator"
SKIP_UNSERVABLE = "skip: not servable"
SKIP_NEEDS_SELECTION = "skip: needs selection"
NEEDS_SYNTHESIS = "needs: synthetic rung"
DROPPED_EXHAUSTED_CHANCES = "dropped: exhausted chances"

__all__ = [
    "DROPPED_EXHAUSTED_CHANCES", "FAULT_STREAK", "MAX_CHANCES",
    "NEEDS_SYNTHESIS", "PRODUCE", "SKIP_NEEDS_SELECTION", "SKIP_NO_OPERATOR",
    "SKIP_STAGED", "SKIP_UNSERVABLE",
    "AugmentationCampaign", "RowBudget",
]


def _dead_end(line, action: str, synthetic: int) -> dict[str, object]:
    """A floor rung 1 cannot serve: no operator, no gate, and no parents to
    name a distractor lane with."""
    return {
        "floor": line.floor, "missing": line.missing,
        "operator": None, "gate": None, "gate_state": None,
        "action": action, "target_rows": 0,
        "synthetic_rows": synthetic, "source_dataset": None,
    }


@dataclass
class RowBudget:
    """One round's row ceiling, spent by every stage that produces: stage 4
    draws per floor, stage 5 inherits whatever is left. `limit=None` is the
    uncapped round the sheet's own numbers describe."""

    limit: int | None = None
    spent: int = 0

    def left(self) -> int | None:
        return None if self.limit is None else max(0, self.limit - self.spent)

    def take(self, want: int) -> int:
        """The largest ask still allowed — `want` itself while uncapped."""
        left = self.left()
        return want if left is None else min(want, left)

    def add(self, rows: int) -> None:
        self.spent += rows

    def exhausted(self) -> bool:
        return self.left() == 0


@dataclass
class _FloorTurn:
    """One floor's standing in the chances scheduler (d59): rows still
    needed, chances left, and every parent id already spent this run
    (accepted or faulted) so a repeat chance never retries the same one."""

    gate: str
    remaining: int
    chances_left: int = MAX_CHANCES
    tried: set[str] = field(default_factory=set)
    accepted: int = 0

    def spend_chance(self) -> bool:
        """Burn one chance for a fault-streak turn; True means none are left
        and the floor is done for this run."""
        self.chances_left -= 1
        return self.chances_left <= 0


class AugmentationCampaign:
    """Plans and runs one pass over every hungry floor."""

    def __init__(
        self,
        loop: AugmentationLoop,
        *,
        pilot_n: int | None = None,
        judge: CoherenceJudge | None = None,
        audit_cleared: set[str] | None = None,
    ) -> None:
        self.loop = loop
        self.pilot_n = loop.config.pilot_n if pilot_n is None else pilot_n
        """Audit-sample size for gated floors — a named policy value from
        the config, sized by the human who will read the sample."""
        self.judge = judge
        """Given one, coherence gates open on ITS verdicts; without one the
        gates wait on the human audit, which is the default."""
        self.audit_cleared = audit_cleared
        """The human's declaration-audit verdict (cleared query_ids): a
        declaration floor whose staged pilot passes at the same bar the
        judge's floors open at is unclamped; None keeps every one held."""

    def _audit_opened(self, pool: pd.DataFrame) -> set[str]:
        """Declaration floors whose whole staged pilot the human cleared at
        the bar coherence floors open at — an audited pilot is a paid-for
        verdict, not feature-stock."""
        if not self.audit_cleared or pool.empty:
            return set()
        gated = pool[
            pool["credit_gate"].fillna("none")
            == str(CreditGate.DECLARATION_AUDIT)
        ]
        if gated.empty:
            return set()
        passed = gated["query_id"].astype(str).isin(self.audit_cleared)
        rate = passed.groupby(gated["floor"]).mean()
        return set(rate[rate >= self.loop.config.coherence_pass_rate].index)

    def plan(self) -> pd.DataFrame:
        """The spend, before any call: one row per hungry floor with the
        action the campaign would take and the target row count. Routes
        through `loop.demand()` — the SAME cell-aware dispatch `run()` uses —
        rather than `operator_for()` directly, which only ever understood
        bare floor labels ("id:tech") and reports every cell name as unservable
        (2026-08 follow-up)."""
        pool = self.loop.pool.load()
        staged = pool["floor"].value_counts()
        opened = (
            self.judge.open_floors(pool, self.loop.config.coherence_pass_rate)
            if self.judge is not None
            else set()
        )
        opened |= self._audit_opened(pool)
        rows: list[dict[str, object]] = []
        for line in self.loop.order_sheet().itertuples(index=False):
            try:
                result, available = self.loop.demand(line.floor)
            except NeedsSelection:
                # its own parents already measure into it: minting rows from
                # nothing would buy supply the pool already holds, so this
                # line is the selection layer's, never generation's
                rows.append(_dead_end(line, SKIP_NEEDS_SELECTION, 0))
                continue
            except ValueError:
                # a cell whose stages leave no servable parent (no lane to
                # borrow distractors from either, so a human names one), or a
                # bare floor nothing registers for
                cell = line.floor in CELLS_BY_NAME
                rows.append(_dead_end(
                    line,
                    NEEDS_SYNTHESIS if cell else SKIP_NO_OPERATOR,
                    math.ceil(float(line.missing)) if cell else 0,
                ))
                continue
            planned = self.loop.planned(line.floor, result)
            gate = self.loop.owner(planned).declaration.credit_gate
            want = math.ceil(float(line.missing))
            gate_state = None
            if gate is not CreditGate.NONE:
                gate_state = "open" if line.floor in opened else "held"
                if gate_state == "held":
                    want = self.pilot_n - int(staged.get(line.floor, 0))
            if want <= 0:
                action, target, synthetic = SKIP_STAGED, 0, 0
            else:
                # what rung 1 can actually REACH, not how many parents exist:
                # a cell with a band nobody serves has 400,000 parents and
                # reaches it zero times. A bare floor has no cell plan and no
                # bands to leave unserved, so its parents are its reach.
                reachable = result.reachable if result else len(available)
                target = min(want, reachable)
                # only a CELL can fall through to synthesis — a bare floor
                # short of parents is exhausted supply, not a generation target
                synthetic = (want - target) if line.floor in CELLS_BY_NAME else 0
                action = PRODUCE if target else (
                    NEEDS_SYNTHESIS if synthetic else SKIP_UNSERVABLE
                )
            rows.append({
                "floor": line.floor,
                "missing": line.missing,
                "operator": ", ".join(op.declaration.operator for op in planned),
                "gate": str(gate),
                "gate_state": gate_state,
                "action": action,
                "target_rows": target,
                "synthetic_rows": synthetic,
                # the lane this cell's own rows live in, so the constructed
                # collection borrows distractors from plausible neighbours
                # rather than from whichever corpus happened to be first
                "source_dataset": (
                    available["dataset"].mode().iat[0]
                    if "dataset" in available.columns and not available.empty
                    else None
                ),
            })
        plan = pd.DataFrame(rows)
        total = int(plan["target_rows"].sum())
        # rows, not calls: a floor served entirely by deterministic operators
        # buys no completion at all, so the old ">= N LLM calls" read as a
        # spend estimate that is now wrong by most of its magnitude
        print(
            f"campaign plan — {len(plan)} hungry floors, {total:,} target rows "
            f"from parents, {int(plan['synthetic_rows'].sum()):,} needing the "
            "synthetic rung:"
        )
        print(plan.to_string(index=False))
        return plan

    def _schedule(
        self, produce_rows: pd.DataFrame, budget: RowBudget
    ) -> tuple[dict[str, _FloorTurn], set[str]]:
        """Run the chances scheduler (d59) to completion: a floor holds the
        front of the queue until it finishes its turn ordinarily (need met,
        or parents exhausted — free) or racks up FAULT_STREAK consecutive
        faults (spends one of its MAX_CHANCES, then goes to the literal back
        of the queue). Exhausting every chance drops the floor for the rest
        of this run, however hungry it still is. Returns each floor's final
        `_FloorTurn` plus the set dropped that way."""
        turns = {
            row.floor: _FloorTurn(gate=row.gate, remaining=int(row.target_rows))
            for row in produce_rows.itertuples(index=False)
        }
        queue: deque[str] = deque(turns)
        dropped: set[str] = set()

        bar = tqdm(total=len(queue), desc="campaign", unit="floor")
        while queue:
            if budget.exhausted():
                tqdm.write(
                    f"~ row budget spent ({budget.spent}) — {len(queue)} floor(s) "
                    "left untouched this round"
                )
                break
            floor = queue.popleft()
            turn = turns[floor]
            bar.set_postfix(floor=floor, chances_left=turn.chances_left)
            produced = self.loop.run(
                floor,
                n=budget.take(turn.remaining),
                exclude=frozenset(turn.tried),
                max_consecutive_faults=FAULT_STREAK,
            )
            turn.accepted += len(produced)
            turn.remaining -= len(produced)
            turn.tried.update(produced.attrs.get("attempted_ids", ()))
            budget.add(len(produced))

            if not produced.attrs.get("stopped_early", False):
                bar.update(1)   # finished ordinarily: need met, or parents exhausted
                continue

            if turn.spend_chance():
                tqdm.write(
                    f"~ {floor}: exhausted all {MAX_CHANCES} chances "
                    f"({FAULT_STREAK} consecutive faults each) — skipped"
                )
                dropped.add(floor)
                bar.update(1)
            else:
                spent = MAX_CHANCES - turn.chances_left
                tqdm.write(
                    f"~ {floor}: {FAULT_STREAK} consecutive faults — chance "
                    f"{spent}/{MAX_CHANCES} spent, requeued"
                )
                queue.append(floor)
        bar.close()
        return turns, dropped

    def run(self, budget: RowBudget | None = None) -> pd.DataFrame:
        """Execute the plan via the chances scheduler (d59) under a shared
        `budget` and report the outcome — one row per hungry floor, dropped
        floors marked distinctly from an ordinary completion."""
        plan = self.plan()
        budget = budget or RowBudget()
        produce_rows = plan[plan["action"] == PRODUCE]
        skipped_rows = plan[plan["action"] != PRODUCE]
        turns, dropped = self._schedule(produce_rows, budget)

        results = [
            {
                **row._asdict(),
                "action": DROPPED_EXHAUSTED_CHANCES if row.floor in dropped else PRODUCE,
                "accepted": turns[row.floor].accepted,
            }
            for row in produce_rows.itertuples(index=False)
        ] + [
            {**row._asdict(), "accepted": 0}
            for row in skipped_rows.itertuples(index=False)
        ]
        summary = pd.DataFrame(results)
        pool = self.loop.pool.load()
        gate_free = int(
            summary.loc[summary["gate"] == str(CreditGate.NONE), "accepted"].sum()
        )
        staged = int(summary["accepted"].sum()) - gate_free
        print(
            f"\ncampaign done: {int(summary['accepted'].sum()):,} rows accepted "
            f"({gate_free:,} credit-eligible, {staged:,} gated audit samples) "
            f"| pool now {len(pool):,} rows | "
            f"{int(summary['synthetic_rows'].sum()):,} rows still owed to the "
            "synthetic rung (stage 5 mints them)"
        )
        if dropped:
            print(
                f"dropped ({MAX_CHANCES} chances exhausted, "
                f"{FAULT_STREAK} consecutive faults each): {sorted(dropped)}"
            )
        door = (
            "V3Composition().admit" if "v3" in self.loop.sheet_path.parts
            else "CellFill().admit"
        )
        print(
            f"next: {door}(GeneratedPool().load()) admits the "
            "credit-eligible rows; the gated floors wait on their audits: "
            + ", ".join(
                sorted(set(
                    summary.loc[
                        (summary["gate"].notna())
                        & (summary["gate"] != str(CreditGate.NONE)),
                        "floor",
                    ]
                )))
        )
        return summary
