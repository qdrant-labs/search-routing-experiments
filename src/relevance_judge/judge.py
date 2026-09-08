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
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Final, NamedTuple

import litellm
import pandas as pd
from litellm import completion
from tqdm.auto import tqdm

from augmentation.engine import (
    Budget,
    BudgetExceeded,
    Spend,
    windowed_map,
)
from relevance_judge.config import RelevanceJudgeConfig
from relevance_judge.lane_context import context_for

INSTRUCTION: Final[str] = (
    "You judge whether a document contains the answer a search query needs — not "
    "whether it shares the query's topic, and not whether it would help someone "
    "working on the problem. A document can be genuinely useful — right API, near "
    "example, good background — and still never state the answer; that document "
    "is a no.\n"
    "First name the need. A factual query needs the exact fact, entity, value, "
    "ruling, argument, or explanation it asks for; a product or item search needs "
    "that item with the asked attributes; a how-to needs the procedure that "
    "accomplishes the asked task; an error message, failing snippet, or bug "
    "report needs the fix — the change, setting, call, or version that resolves "
    "that specific failure. For an error, a document that shows the same API used "
    "correctly, an analogous example, or the general mechanism is not the fix "
    "unless its text contains the exact element whose absence or misuse causes "
    "the error.\n"
    "Then find the answer in the document's visible words. One line of a long "
    "document is enough, and wording need not match — a fact equivalent in "
    "substance counts (an annulment answers whether a couple divorced). But if "
    "you can only describe what the document demonstrates, explains, or sets up, "
    "and cannot copy the words that state the answer, the answer is absent. "
    "Documents are often truncated fragments: judge only the text shown — what "
    "the full page probably contains does not exist here. A blank, boilerplate, "
    "or title-only document is never relevant.\n"
    "Answer in exactly four lines:\n"
    "ASKED: what the query needs — for an error, the fix. If the query offers "
    "alternatives (A or B or C, or a comma-separated list), name the single one "
    "this document could satisfy, not all of them. At most 12 words.\n"
    "EVIDENCE: the document's own words that state it, copied, shortened with "
    "... — or NOTHING. At most 15 words.\n"
    "MISSING: whatever ASKED names that the document never states — or NOTHING. "
    "Judge against ASKED, not against the whole query: an alternative ASKED did "
    "not name is not missing. Substance counts, wording does not; name only what "
    "was asked, never nice-to-have extras. At most 10 words.\n"
    "VERDICT: yes or no, lowercase, then a dash, then your reason in at most 10 "
    "words.\n"
    "The first three lines decide the fourth: VERDICT is yes only when EVIDENCE "
    "quotes the document and MISSING is NOTHING. EVIDENCE that only names the "
    "document's topic or genre ('discusses X', 'is about Y', 'covers Z') counts "
    "as NOTHING — but a copied span that carries the answer IS evidence even if "
    "its wording differs from the query's. If EVIDENCE is NOTHING or MISSING "
    "names anything, VERDICT is no. Hedging in YOUR OWN assessment — a gap you "
    "would concede with 'but not', 'does not state', 'though', or 'appears "
    "incorrect' — belongs in MISSING and makes the verdict no; those same words "
    "appearing inside the document you quote mean nothing. When you cannot "
    "confirm the answer is present, say no — a wrong yes becomes permanent gold "
    "that corrupts every score built on it; a wrong no costs only one missed pair."
)

RATIONALE_FIELDS: Final[tuple[str, ...]] = ("asked", "evidence", "missing")
"""The judge's pre-verdict reasoning fields. Stored per atom because a gold
label is permanent: EVIDENCE is the quote that justifies it, MISSING is why a
no was a no. Discarding them made every audit start from scratch."""


ATOM_COLUMNS: Final[tuple[str, ...]] = (
    "dataset", "query_id", "doc_id", "relevance", "source",
    "reason", *RATIONALE_FIELDS, "prompt_hash", "judged_at", "judge_run_id",
)
RUN_COLUMNS: Final[tuple[str, ...]] = (
    "judge_run_id", "model", "opened_at", "passed",
    "precision_relevant", "agreement_overall", "n_validation",
)


class Verdict(NamedTuple):
    """One judged pair. `fields` carries the labelled lines the prompt asks for
    before the verdict; empty when the reply was unreadable."""

    relevant: bool | None
    reason: str
    prompt_hash: str
    spend: Spend
    fields: dict[str, str] = {}


def parse_fields(reply: str) -> dict[str, str]:
    """The labelled pre-verdict lines, lowercased keys. A field the model omits
    is simply absent — never invented."""
    out: dict[str, str] = {}
    for name in RATIONALE_FIELDS:
        m = re.search(rf"^{name}\s*:\s*(.*)$", reply, re.IGNORECASE | re.MULTILINE)
        if m:
            out[name] = " ".join(m.group(1).split())[:300]
    return out


_VERDICT = re.compile(r"VERDICT\s*:\s*\W*(yes|no)\b[\s\W]*(.*)", re.IGNORECASE | re.DOTALL)
_BARE = re.compile(r"\W*(yes|no)\b[\s\W]*(.*)", re.IGNORECASE | re.DOTALL)


def _parse(reply: str) -> tuple[bool, str] | None:
    """The (relevant, reason) a reply carries, else None — an answer nothing can
    read is never a silent relevant. Prefers the labelled VERDICT line so the
    ASKED/EVIDENCE/MISSING lines above it cannot be mistaken for the verdict;
    the bare verdict-first form still parses, so banked runs stay reproducible."""
    reply = reply.strip()
    match = _VERDICT.search(reply) or _BARE.match(reply)
    if match is None:
        return None
    return match.group(1).lower() == "yes", " ".join(match.group(2).split())[:200]


_CONNECTION_FAULTS: Final[tuple[str, ...]] = (
    "connection refused", "connection reset", "connection aborted",
    "connection error", "server disconnected", "temporarily unavailable",
    "broken pipe", "timed out",
)
"""Substrings of a transport failure that completed NO call — nothing was
charged and the work is simply lost, so a retry is free to attempt. Matched on
the message because litellm wraps the socket error in a provider exception
(`OpenrouterException - [Errno 61] Connection refused`), erasing the type."""


def _is_connection_fault(error: Exception) -> bool:
    text = str(error).lower()
    return any(mark in text for mark in _CONNECTION_FAULTS)


def _prompt_hash(query: str, doc_text: str, lane_context: str = "") -> str:
    h = hashlib.sha256()
    h.update(INSTRUCTION.encode())
    h.update(b"\n--ctx--\n")
    h.update(lane_context.encode())
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
        self.dropped: Counter[str] = Counter()
        """Provider errors swallowed per exception name. Silence here is what made
        a 60s-per-call provider look like a hung run."""
        self.dropped_detail: dict[str, str] = {}
        """First message seen per error name — the counter says how often, this
        says what actually went wrong."""
        self.unreadable: list[str] = []
        """Raw replies the parser could not read, capped by config — the evidence
        needed to tell a format regression from a provider fault."""
        # The MODULE global is the binding one: litellm ships request_timeout=6000
        # and the per-call `timeout=` kwarg did not override it on the OpenRouter
        # path (a timeout=30 call was measured completing at 60.9s).
        litellm.request_timeout = self.config.request_timeout_s
        # litellm prints a "Give Feedback / Get Help" banner to stderr for EVERY
        # provider error, before the exception reaches our handler. At 32 workers
        # that buries the run's own output in thousands of lines while telling us
        # nothing — `dropped` already counts the errors and keeps their messages.
        litellm.suppress_debug_info = True

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
        self, query: str, doc_text: str, *, dataset: str = "",
        budget: Budget | None = None,
    ) -> Verdict:
        """The atomic call, shared by the harness (compare to human) and the
        banking path. Returns (relevant | None, reason, prompt_hash, spend);
        None relevance means the provider blipped or the reply was unreadable."""
        lane_context = context_for(dataset)
        prompt_hash = _prompt_hash(query, doc_text, lane_context)
        spend = Spend()
        if not query.strip() or not doc_text.strip():
            # an empty doc cannot answer anything — never spend a call to "judge" it,
            # and never let a blank prompt drift to a relevant verdict.
            return Verdict(False, "empty query or document", prompt_hash, spend)
        # The lane card goes in the SYSTEM turn, after the universal rules:
        # it narrows what counts as relevant for this collection, never loosens
        # the standard of evidence the instruction demands.
        system = f"{INSTRUCTION}\n\n{lane_context}" if lane_context else INSTRUCTION
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": f"Query: {query}\n\nDocument:\n{doc_text}"},
        ]
        # The cap must clear the two-line reply even when the model overruns the
        # word limits: a truncated VERDICT line is an unreadable, PAID pair.
        if budget is not None:
            budget.reserve(messages, self.config.max_answer_tokens)
        # luna is a REASONING model, and hidden reasoning is what makes a yes/no
        # take ~60s. OpenRouter advertises two spellings for the control
        # (`reasoning` and `reasoning_effort`); the nested extra_body form alone
        # left latency at 60s, so send the top-level param litellm maps natively
        # AND keep the nested one as the fallback for providers that read it.
        extra = (
            {
                "reasoning_effort": self.config.reasoning_effort,
                "extra_body": {"reasoning": {"effort": self.config.reasoning_effort}},
            }
            if self.config.reasoning_effort else {}
        )
        # A reply cut off at the token cap loses its VERDICT line and parses to
        # nothing — a PAID pair thrown away (249 of them, once). The provider
        # says so via finish_reason, so detect it and retry with room rather
        # than raising the cap and hoping the next corpus is not longer.
        cap = self.config.max_answer_tokens
        attempt, tries = 0, 0
        while attempt < 2:
            try:
                start = time.monotonic()
                response = completion(
                    model=self.model, messages=messages, max_tokens=cap,
                    temperature=0.0, **extra,
                )
            except BudgetExceeded:
                # the ceiling is the ONE error that must stop the run
                raise
            except Exception as error:  # noqa: BLE001 - one bad pair cannot end a paid run
                # Deliberately broad: a non-transient provider error (a rejected
                # param, a filtered document, a malformed row) used to propagate
                # out of windowed_map and abandon every remaining pair.
                name = type(error).__name__
                # A refused/reset connection completed no call, so it cost nothing
                # and the work is simply lost — measured at 2,115 of 3,600 pairs in
                # one run ("[Errno 61] Connection refused" under sustained load).
                # Retrying with backoff recovers them; a provider that rejects the
                # request itself will reject it again, so only connection-level
                # faults are worth a second attempt.
                if _is_connection_fault(error) and tries < self.config.connect_retries:
                    tries += 1
                    self.dropped["connection_retried"] += 1
                    time.sleep(self.config.connect_backoff_s * tries)
                    continue
                self.dropped[name] += 1
                self.dropped_detail.setdefault(name, str(error)[:300])
                return Verdict(None, "", prompt_hash, spend)

            usage = getattr(response, "usage", None)
            prompt_tokens = getattr(usage, "prompt_tokens", 0) or 0
            answer_tokens = getattr(usage, "completion_tokens", 0) or 0
            if budget is not None:
                budget.charge(prompt_tokens, answer_tokens)
            spend.add_hop(time.monotonic() - start, prompt_tokens + answer_tokens)

            reply = response.choices[0].message.content or ""
            parsed = _parse(reply)
            if parsed is not None:
                return Verdict(parsed[0], parsed[1], prompt_hash, spend,
                               parse_fields(reply))

            truncated = getattr(response.choices[0], "finish_reason", None) == "length"
            if truncated and attempt == 0:
                self.dropped["truncated_retried"] += 1
                cap *= 2
                attempt += 1
                continue
            self.dropped["truncated" if truncated else "unparsed"] += 1
            if len(self.unreadable) < self.config.unreadable_samples:
                self.unreadable.append(f"[finish={'length' if truncated else 'other'}] {reply[:360]}")
            return Verdict(None, "unparsed", prompt_hash, spend)
        return Verdict(None, "unparsed", prompt_hash, spend)

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
            return self.judge_one(str(row.query), str(row.doc_text),
                                  dataset=str(row.dataset), budget=budget)

        buffer: list[dict] = []
        bar = tqdm(total=len(todo), desc="relevance", unit="pair")
        # `finally`, because a budget stop (or any error) reaches us through
        # windowed_map's future.result(): without it the partial buffer of
        # already-PAID verdicts would be dropped on the way out.
        try:
            for row, verdict in windowed_map(
                attempt, todo.itertuples(index=False), self.config.llm_workers
            ):
                bar.update(1)
                relevant, reason = verdict.relevant, verdict.reason
                spend.add(verdict.spend)
                if self.dropped:
                    bar.set_postfix_str(f"drop={sum(self.dropped.values())}")
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
                    **{f: verdict.fields.get(f, "") for f in RATIONALE_FIELDS},
                    "prompt_hash": verdict.prompt_hash,
                    "judged_at": datetime.now(UTC).isoformat(timespec="seconds"),
                    "judge_run_id": run_id,
                })
                if len(buffer) >= 100:
                    self._append(buffer)
                    buffer = []
        finally:
            if buffer:
                self._append(buffer)
            bar.close()
        print(f"  {spend.summary(counts['judged'])}")
        if self.dropped:
            print(f"  dropped: {dict(self.dropped)}")
            for name, message in self.dropped_detail.items():
                print(f"    {name}: {message}")
        for reply in self.unreadable:
            print(f"    unparsed reply: {reply!r}")
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
