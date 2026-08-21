"""The coherence gate's judge: one LLM verdict per COHERENCE_GATE row, banked
as its own artifact.

The gate asks two things of a generated row — that a real user could plausibly
issue the query, and that the document paired with it actually answers it. The
verdict never touches the pool, whose `credit_gate` is provenance, so admission
and the campaign's clamp read this artifact instead of a rewritten row.
"""

from __future__ import annotations

import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Final

import pandas as pd
from tqdm.auto import tqdm

from augmentation.config import AugmentationConfig
from augmentation.constructed import ConstructedDocs
from augmentation.core import CreditGate
from augmentation.engine import Augmenter, Spend, windowed_map
from augmentation.parents import ParentPool

_COLUMNS: Final[tuple[str, ...]] = (
    "query_id", "floor", "verdict", "reason", "model", "judged_at",
)

INSTRUCTION = (
    "You audit generated search queries before they earn dataset credit. "
    "Both of these must hold:\n"
    "1. a real user could plausibly type this query into a search box;\n"
    "2. the document below actually answers that query.\n"
    "Start your reply with 'yes' if both hold, or 'no' if either fails, then "
    "a dash and one clause of at most 12 words saying why. Nothing else."
)


def _parse(reply: str) -> tuple[bool, str] | None:
    """The (verdict, reason) a reply opening with yes/no carries, else None —
    an answer nothing can read is never a silent pass."""
    match = re.match(
        r"\W*(yes|no)\b[\s\W]*(.*)", reply.strip(), re.IGNORECASE | re.DOTALL
    )
    if match is None:
        return None
    return match.group(1).lower() == "yes", " ".join(match.group(2).split())[:200]


class CoherenceJudge:
    """Owns `paths.coherence_audit` — idempotent on query_id, so a rerun
    spends only on rows still unjudged."""

    def __init__(
        self,
        engine: Augmenter | None = None,
        *,
        config: AugmentationConfig | None = None,
        docs: ConstructedDocs | None = None,
        parents: ParentPool | None = None,
    ) -> None:
        self.config = config or AugmentationConfig()
        self.docs = docs or ConstructedDocs(self.config.paths)
        self.parents = parents
        """Inject's evidence is a lane document, so a row grounded in one is
        unjudgeable without the pool that can read it."""
        self._engine = engine

    @property
    def path(self) -> Path:
        return self.config.paths.coherence_audit

    def load(self) -> pd.DataFrame:
        if not self.path.exists():
            return pd.DataFrame(columns=_COLUMNS)
        return pd.read_parquet(self.path)

    def passed(self) -> set[str]:
        """Query ids the gate has cleared — what admission reads."""
        verdicts = self.load()
        if verdicts.empty:
            return set()
        return set(
            verdicts.loc[verdicts["verdict"].astype(bool), "query_id"].astype(str)
        )

    @staticmethod
    def candidates(pool: pd.DataFrame) -> pd.DataFrame:
        """Every pool row this gate holds back."""
        if "credit_gate" not in pool.columns:
            return pool.iloc[0:0]
        gate = pool["credit_gate"].fillna(str(CreditGate.NONE))
        return pool[gate == CreditGate.COHERENCE_GATE]

    def open_floors(self, pool: pd.DataFrame, min_pass_rate: float) -> set[str]:
        """Floors whose whole staged pilot is judged and passed at the dial —
        every other gated floor stays clamped to the pilot."""
        staged = self.candidates(pool)
        if staged.empty:
            return set()
        verdicts = self.load()
        judged = staged["query_id"].astype(str).map(
            dict(zip(
                verdicts["query_id"].astype(str),
                verdicts["verdict"].astype(bool),
                strict=True,
            ))
        )
        return {
            floor
            for floor, values in judged.groupby(staged["floor"])
            if values.notna().all() and values.astype(bool).mean() >= min_pass_rate
        }

    def run(self, pool: pd.DataFrame) -> dict[str, int]:
        """Judge every unjudged gated row — `llm_workers` calls in flight,
        verdicts consumed in submission order and banked in 100-row chunks
        (per-row full-file rewrites were quadratic; a crash re-pays <=100)."""
        staged = self.candidates(pool)
        already = set(self.load()["query_id"].astype(str))
        todo = staged[~staged["query_id"].astype(str).isin(already)]
        counts = dict(
            candidates=len(staged), skipped=len(staged) - len(todo),
            judged=0, passed=0, failed=0, unparsed=0, no_evidence=0,
        )
        engine = self._engine or Augmenter(
            self.config.engine, seed=self.config.seed
        )
        constructed = self.docs.load()
        spend = Spend()

        def attempt(row):
            evidence = self._evidence(row, constructed)
            if not evidence:
                return "no_evidence", None, Spend()
            reply, hop = engine.ask(
                INSTRUCTION, f"Query: {row.query}\n\nDocument:\n{evidence}"
            )
            parsed = _parse(reply or "")
            if parsed is None:
                return "unparsed", None, hop
            return "judged", parsed, hop

        buffer: list[dict] = []
        bar = tqdm(total=len(todo), desc="coherence", unit="row")
        for row, (status, parsed, hop) in windowed_map(
            attempt, todo.itertuples(index=False), self.config.llm_workers
        ):
            bar.update(1)
            spend.add(hop)
            if status != "judged":
                counts[status] += 1
                continue
            verdict, reason = parsed
            counts["judged"] += 1
            counts["passed" if verdict else "failed"] += 1
            buffer.append({
                "query_id": str(row.query_id),
                "floor": str(row.floor),
                "verdict": verdict,
                "reason": reason,
                "model": engine.model,
                "judged_at": datetime.now(UTC).isoformat(timespec="seconds"),
            })
            if len(buffer) >= 100:
                self._append(buffer)
                buffer = []
        if buffer:
            self._append(buffer)
        bar.close()
        print(f"  {spend.summary(counts['judged'])}")
        return counts

    def _evidence(self, row, constructed: pd.DataFrame) -> str:
        """The document the verdict reads: the row's constructed answer docs,
        else the lane document it was grounded in."""
        mine = constructed[
            constructed["for_query"].astype(str) == str(row.query_id)
        ]
        if not mine.empty:
            return "\n\n".join(mine["text"].astype(str))
        if self.parents is None:
            return ""
        return self.parents.gold_text(
            pd.Series({**row._asdict(), "dataset": row.home_lane})
        )

    def _append(self, verdicts: list[dict[str, object]]) -> None:
        fresh = pd.DataFrame(verdicts, columns=_COLUMNS)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        pd.concat([self.load(), fresh], ignore_index=True).to_parquet(
            self.path, index=False
        )
