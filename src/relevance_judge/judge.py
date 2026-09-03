"""The relevance-atom judge: one binary verdict per (query, doc), banked as its
own artifact and merged with human qrels only at scoring time.

The judge never sees route names, lists to compare, or scores — one call asks
"is this document relevant to this query?" and nothing else. That shape is the
whole point: it generates truth atoms checkable against human qrels, never route
opinions (doc §2).
"""

from __future__ import annotations

import hashlib
import re
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Final

import pandas as pd
from litellm import completion
from tqdm.auto import tqdm

from augmentation.engine import (
    TRANSIENT_PROVIDER_ERRORS,
    Budget,
    Spend,
    windowed_map,
)
from relevance_judge.config import RelevanceJudgeConfig

INSTRUCTION: Final[str] = (
    "You judge whether a document contains the answer a search query needs — not "
    "whether it shares the query's topic.\n"
    "Say yes only if the document contains the specific fact, entity, value, "
    "procedure, fix, ruling, argument, or explanation the query asks for — either "
    "fully, or as a substantial part that genuinely helps answer it. A fact "
    "equivalent in substance counts (an annulment answers whether a couple "
    "divorced). Say no if that answer content is absent, even when the document is "
    "closely related: same subject, same product or API, a different aspect, "
    "background, a setup or framing that never answers, or the query's terms "
    "repeated. Shared words and topical closeness are not evidence of relevance. A "
    "blank, boilerplate, or title-only document is never relevant. When you cannot "
    "confirm the answer content is actually present, say no — a wrong yes becomes "
    "permanent gold that corrupts every score built on it, while a wrong no costs "
    "only one missed pair.\n"
    "Your reason must name the specific answer content you found, or the specific "
    "gap. Never answer yes with a reason that admits the document does not answer "
    "the query.\n"
    "Reply with a lowercase yes or no as the very first word — no quotes, capital "
    "letters, or punctuation before it — then a dash, then that reason in at most "
    "12 words. Nothing before the verdict, nothing after the reason."
)

ATOM_COLUMNS: Final[tuple[str, ...]] = (
    "dataset", "query_id", "doc_id", "relevance", "source",
    "reason", "prompt_hash", "judged_at", "judge_run_id",
)
RUN_COLUMNS: Final[tuple[str, ...]] = (
    "judge_run_id", "model", "opened_at", "passed",
    "precision_relevant", "agreement_overall", "n_validation",
)


def _parse(reply: str) -> tuple[bool, str] | None:
    """The (relevant, reason) a yes/no reply carries, else None — an answer
    nothing can read is never a silent relevant."""
    match = re.match(r"\W*(yes|no)\b[\s\W]*(.*)", reply.strip(), re.IGNORECASE | re.DOTALL)
    if match is None:
        return None
    return match.group(1).lower() == "yes", " ".join(match.group(2).split())[:200]


def _prompt_hash(query: str, doc_text: str) -> str:
    h = hashlib.sha256()
    h.update(INSTRUCTION.encode())
    h.update(b"\n--q--\n")
    h.update(query.encode())
    h.update(b"\n--d--\n")
    h.update(doc_text.encode())
    return h.hexdigest()[:16]


class RelevanceJudge:
    """One binary verdict per (query, doc). `judged_qrels.parquet` is idempotent
    on (dataset, query_id, doc_id) — a rerun spends only on unjudged pairs."""

    def __init__(self, config: RelevanceJudgeConfig | None = None) -> None:
        self.config = config or RelevanceJudgeConfig()
        self.model = self.config.engine.model

    @property
    def path(self) -> Path:
        return self.config.judged_qrels

    def load(self) -> pd.DataFrame:
        if not self.path.exists():
            return pd.DataFrame(columns=ATOM_COLUMNS)
        return pd.read_parquet(self.path)

    def judged_keys(self) -> set[tuple[str, str, str]]:
        atoms = self.load()
        if atoms.empty:
            return set()
        return set(
            zip(
                atoms["dataset"].astype(str),
                atoms["query_id"].astype(str),
                atoms["doc_id"].astype(str),
                strict=True,
            )
        )

    def judge_one(
        self, query: str, doc_text: str, *, budget: Budget | None = None
    ) -> tuple[bool | None, str, str, Spend]:
        """The atomic call, shared by the harness (compare to human) and the
        banking path. Returns (relevant | None, reason, prompt_hash, spend);
        None relevance means the provider blipped or the reply was unreadable."""
        prompt_hash = _prompt_hash(query, doc_text)
        spend = Spend()
        if not query.strip() or not doc_text.strip():
            # an empty doc cannot answer anything — never spend a call to "judge" it,
            # and never let a blank prompt drift to a relevant verdict.
            return False, "empty query or document", prompt_hash, spend
        messages = [
            {"role": "system", "content": INSTRUCTION},
            {"role": "user", "content": f"Query: {query}\n\nDocument:\n{doc_text}"},
        ]
        # ponytail: max_tokens=128 fits "yes - <clause>"; if luna reasons hidden
        # tokens, raise it or set reasoning off per its card — one knob.
        if budget is not None:
            budget.reserve(messages, 128)
        # OpenRouter-native reasoning control (litellm forwards extra_body verbatim);
        # 'none' stops luna emitting a reasoning preamble that would fail the parser.
        extra = (
            {"extra_body": {"reasoning": {"effort": self.config.reasoning_effort}}}
            if self.config.reasoning_effort else {}
        )
        try:
            start = time.monotonic()
            response = completion(
                model=self.model, messages=messages, max_tokens=128,
                temperature=0.0, **extra,
            )
        except TRANSIENT_PROVIDER_ERRORS:
            return None, "", prompt_hash, spend
        elapsed = time.monotonic() - start
        usage = getattr(response, "usage", None)
        prompt_tokens = getattr(usage, "prompt_tokens", 0) or 0
        answer_tokens = getattr(usage, "completion_tokens", 0) or 0
        if budget is not None:
            budget.charge(prompt_tokens, answer_tokens)
        spend.add_hop(elapsed, prompt_tokens + answer_tokens)
        parsed = _parse(response.choices[0].message.content or "")
        if parsed is None:
            return None, "unparsed", prompt_hash, spend
        return parsed[0], parsed[1], prompt_hash, spend

    def judge_pairs(
        self,
        pairs: pd.DataFrame,
        *,
        run_id: str,
        budget: Budget | None = None,
    ) -> dict[str, int]:
        """Bank a verdict for every unjudged pair. `pairs` needs
        [dataset, query_id, doc_id, query, doc_text]; verdicts land in 100-row
        chunks (a crash re-pays <=100), stamped with `run_id`."""
        already = self.judged_keys()
        if pairs.empty:
            todo = pairs
        else:
            keys = pd.MultiIndex.from_frame(
                pairs[["dataset", "query_id", "doc_id"]].astype(str)
            )
            todo = pairs[~keys.isin(already)]
        counts = dict(
            candidates=len(pairs), skipped=len(pairs) - len(todo),
            judged=0, relevant=0, unreadable=0,
        )
        spend = Spend()

        def attempt(row):
            relevant, reason, prompt_hash, hop = self.judge_one(
                str(row.query), str(row.doc_text), budget=budget
            )
            return relevant, reason, prompt_hash, hop

        buffer: list[dict] = []
        bar = tqdm(total=len(todo), desc="relevance", unit="pair")
        for row, (relevant, reason, prompt_hash, hop) in windowed_map(
            attempt, todo.itertuples(index=False), self.config.llm_workers
        ):
            bar.update(1)
            spend.add(hop)
            if relevant is None:
                counts["unreadable"] += 1
                continue
            counts["judged"] += 1
            counts["relevant"] += int(relevant)
            buffer.append({
                "dataset": str(row.dataset),
                "query_id": str(row.query_id),
                "doc_id": str(row.doc_id),
                "relevance": int(relevant),
                "source": "llm",
                "reason": reason,
                "prompt_hash": prompt_hash,
                "judged_at": datetime.now(UTC).isoformat(timespec="seconds"),
                "judge_run_id": run_id,
            })
            if len(buffer) >= 100:
                self._append(buffer)
                buffer = []
        if buffer:
            self._append(buffer)
        bar.close()
        print(f"  {spend.summary(counts['judged'])}")
        return counts

    def as_qrelstore(self):
        """Project the judged atoms to the `QrelStore` schema (source='llm') for
        the scoring-time merge. Imported lazily so the judge has no scoring dep."""
        from hybrid_search_rrf_dataset.qrels import QrelStore

        atoms = self.load()
        frame = atoms.assign(source="llm")[
            ["dataset", "query_id", "doc_id", "relevance", "source"]
        ] if not atoms.empty else pd.DataFrame(columns=QrelStore.COLUMNS)
        return QrelStore(frame)

    def _append(self, atoms: list[dict[str, object]]) -> None:
        # ponytail: read-modify-write of the whole file per flush, so total cost
        # is quadratic in banked atoms. Fine to ~10K; above that write per-run
        # part files and glob them in `load()`.
        fresh = pd.DataFrame(atoms, columns=ATOM_COLUMNS)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        pd.concat([self.load(), fresh], ignore_index=True).to_parquet(
            self.path, index=False
        )


class JudgeRunLog:
    """`judge_runs.parquet` — one row per judging run, holding the model and the
    validation metrics measured at judging time. Atoms reference a run by id
    rather than duplicating these run-level facts on every pair."""

    def __init__(self, config: RelevanceJudgeConfig | None = None) -> None:
        self.config = config or RelevanceJudgeConfig()

    @property
    def path(self) -> Path:
        return self.config.judge_runs

    def load(self) -> pd.DataFrame:
        if not self.path.exists():
            return pd.DataFrame(columns=RUN_COLUMNS)
        return pd.read_parquet(self.path)

    def open(self, model: str, metrics: dict[str, object]) -> str:
        """Register a run from its validation metrics, returning its id. The id
        is timestamp-based so a rerun is a distinct, auditable run."""
        opened_at = datetime.now(UTC).isoformat(timespec="seconds")
        run_id = f"{opened_at}::{model}"
        row = {
            "judge_run_id": run_id, "model": model, "opened_at": opened_at,
            "passed": bool(metrics.get("passed", False)),
            "precision_relevant": float(metrics.get("precision_relevant", float("nan"))),
            "agreement_overall": float(metrics.get("agreement_overall", float("nan"))),
            "n_validation": int(metrics.get("n_validation", 0)),
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        pd.concat(
            [self.load(), pd.DataFrame([row], columns=RUN_COLUMNS)], ignore_index=True
        ).to_parquet(self.path, index=False)
        return run_id


def _self_check() -> None:
    assert _parse("yes - answers the query") == (True, "answers the query")
    assert _parse("NO — off topic")[0] is False
    assert _parse("  Yes.") == (True, "")
    assert _parse("maybe") is None
    assert _parse("") is None
    a = _prompt_hash("q", "d")
    assert a == _prompt_hash("q", "d") and a != _prompt_hash("q", "d2")
    print("judge self-check ok")


if __name__ == "__main__":
    _self_check()
