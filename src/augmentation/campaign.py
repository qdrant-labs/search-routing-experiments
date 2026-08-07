"""The campaign — d42(a)'s outer while over the order sheet.

One turn of the crank: walk every hungry floor most-hungry-first and
dispatch each servable one through the AugmentationLoop. Gate-free floors
produce to the floor's need; gated floors stage only a `pilot_n` audit
sample (more would be paid feature-stock), and floors whose pilot is
already in the pool are skipped — re-running a campaign never re-pays.

The crank deliberately stops before admission: `CellFill().admit(...)` is
the only door into the composition, and the gated floors' credit waits on
the human audits between cranks.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field

import pandas as pd
from tqdm.auto import tqdm

from augmentation.core import CreditGate
from augmentation.loop import AugmentationLoop
from composition.cells import CELLS_BY_NAME

PRODUCE = "produce"
SKIP_STAGED = "skip: pilot staged"
SKIP_NO_OPERATOR = "skip: no operator"
SKIP_UNSERVABLE = "skip: not servable"
DROPPED_EXHAUSTED_CHANCES = "dropped: exhausted chances"

MAX_CHANCES = 3
"""How many separate turns a floor gets before it is dropped for the rest of
the run (d59) — not a timeout, a fairness budget: 3 chances x FAULT_STREAK
faults each is the fixed, worst-case-provable ceiling on wasted spend."""
FAULT_STREAK = 3
"""Consecutive faulty parent attempts (d59's unified signal — an engine
error, a structural rejection, or a measured-but-failed target all count
the same) that end a floor's current turn and spend one chance."""


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
        self, loop: AugmentationLoop, *, pilot_n: int | None = None
    ) -> None:
        self.loop = loop
        self.pilot_n = loop.config.pilot_n if pilot_n is None else pilot_n
        """Audit-sample size for gated floors — a named policy value from
        the config, sized by the human who will read the sample."""

    def plan(self) -> pd.DataFrame:
        """The spend, before any call: one row per hungry floor with the
        action the campaign would take and the target row count. Routes
        through `loop.demand()` — the SAME cell-aware dispatch `run()` uses —
        rather than `operator_for()` directly, which only ever understood
        bare floor labels ("id:tech") and reports every cell name as unservable
        (2026-08 follow-up)."""
        staged = self.loop.pool.load()["floor"].value_counts()
        rows: list[dict[str, object]] = []
        for line in self.loop.order_sheet().itertuples(index=False):
            try:
                result, _ = self.loop.demand(line.floor)
            except ValueError:
                # a cell whose stages leave no servable parent, or a bare
                # floor nothing registers for — demand() raises either way
                action = (
                    SKIP_NO_OPERATOR
                    if line.floor not in CELLS_BY_NAME
                    else SKIP_UNSERVABLE
                )
                rows.append({
                    "floor": line.floor, "missing": line.missing,
                    "operator": None, "gate": None,
                    "action": action, "target_rows": 0,
                })
                continue
            planned = self.loop.planned(line.floor, result)
            gate = self.loop.owner(planned).declaration.credit_gate
            if gate is CreditGate.NONE:
                action, target = PRODUCE, math.ceil(float(line.missing))
            else:
                remaining = self.pilot_n - int(staged.get(line.floor, 0))
                if remaining <= 0:
                    action, target = SKIP_STAGED, 0
                else:
                    action, target = PRODUCE, remaining
            rows.append({
                "floor": line.floor,
                "missing": line.missing,
                "operator": ", ".join(op.declaration.operator for op in planned),
                "gate": str(gate),
                "action": action,
                "target_rows": target,
            })
        plan = pd.DataFrame(rows)
        total = int(plan["target_rows"].sum())
        print(f"campaign plan — {len(plan)} hungry floors, "
              f"{total:,} target rows (>= {total:,} LLM calls):")
        print(plan.to_string(index=False))
        return plan

    def _schedule(
        self, produce_rows: pd.DataFrame
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
            floor = queue.popleft()
            turn = turns[floor]
            bar.set_postfix(floor=floor, chances_left=turn.chances_left)
            produced = self.loop.run(
                floor,
                n=turn.remaining,
                exclude=frozenset(turn.tried),
                max_consecutive_faults=FAULT_STREAK,
            )
            turn.accepted += len(produced)
            turn.remaining -= len(produced)
            turn.tried.update(produced.attrs.get("attempted_ids", ()))

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

    def run(self) -> pd.DataFrame:
        """Execute the plan via the chances scheduler (d59) and report the
        outcome — one row per hungry floor, dropped floors marked distinctly
        from an ordinary completion."""
        plan = self.plan()
        produce_rows = plan[plan["action"] == PRODUCE]
        skipped_rows = plan[plan["action"] != PRODUCE]
        turns, dropped = self._schedule(produce_rows)

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
            f"| pool now {len(pool):,} rows"
        )
        if dropped:
            print(
                f"dropped ({MAX_CHANCES} chances exhausted, "
                f"{FAULT_STREAK} consecutive faults each): {sorted(dropped)}"
            )
        print(
            "next: CellFill().admit(GeneratedPool().load()) admits the "
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
