"""The campaign — d42(a)'s outer while over the order sheet.

One turn of the crank: walk every hungry floor most-hungry-first and
dispatch each servable one through the AugmentationLoop. Gate-free floors
produce to the floor's need; gated floors stage only a `pilot_n` audit
sample (more would be paid feature-stock), and floors whose pilot is
already in the pool are skipped — re-running a campaign never re-pays.

The crank deliberately stops before admission: `MiniFill().admit(...)` is
the only door into the composition, and the gated floors' credit waits on
the human audits between cranks.
"""

from __future__ import annotations

import math

import pandas as pd
from tqdm.auto import tqdm

from augmentation.core import CreditGate
from augmentation.loop import AugmentationLoop

PRODUCE = "produce"
SKIP_STAGED = "skip: pilot staged"
SKIP_NO_OPERATOR = "skip: no operator"


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
        action the campaign would take and the target row count."""
        staged = self.loop.pool.load()["floor"].value_counts()
        rows: list[dict[str, object]] = []
        for line in self.loop.order_sheet().itertuples(index=False):
            operator = self.loop.operator_for(line.floor)
            if operator is None:
                action, target = SKIP_NO_OPERATOR, 0
                gate = None
            else:
                gate = str(operator.declaration.credit_gate)
                if operator.declaration.credit_gate is CreditGate.NONE:
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
                "operator": operator.declaration.operator if operator else None,
                "gate": gate,
                "action": action,
                "target_rows": target,
            })
        plan = pd.DataFrame(rows)
        total = int(plan["target_rows"].sum())
        print(f"campaign plan — {len(plan)} hungry floors, "
              f"{total:,} target rows (>= {total:,} LLM calls):")
        print(plan.to_string(index=False))
        return plan

    def run(self) -> pd.DataFrame:
        """Execute the plan. Outer bar tracks floors; each floor's inner
        bar (accepted vs need, before/after lines) comes from loop.run."""
        plan = self.plan()
        results: list[dict[str, object]] = []
        bar = tqdm(
            plan.itertuples(index=False),
            total=len(plan),
            desc="campaign",
            unit="floor",
        )
        for entry in bar:
            bar.set_postfix(floor=entry.floor)
            if entry.action != PRODUCE:
                tqdm.write(f"~ {entry.floor}: {entry.action}")
                results.append({**entry._asdict(), "accepted": 0})
                continue
            produced = self.loop.run(entry.floor, n=int(entry.target_rows))
            results.append({**entry._asdict(), "accepted": len(produced)})
        bar.close()

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
        print(
            "next: MiniFill().admit(GeneratedPool().load()) admits the "
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
