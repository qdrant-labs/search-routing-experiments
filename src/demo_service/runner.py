"""The demo's stage machine: one document, one candidate query at a time,
budgeted and deadlined generation, three separately-tracked checks (shape,
guard, source-answerability), retrieval, and export.

Everything that can go wrong — no pipeline-service, no model key, a slow
model, an exhausted budget, a Qdrant miss — raises `DemoUnavailable` (or a
subclass). `demo_service.api` is the only place that catches it and swaps in
a `ReplayBundle`; this module never renders anything and never falls back
on its own, so a late background call can update `Budget.spent_usd` (money
really was spent) but can never touch the run state a caller already read.
"""

from __future__ import annotations

import os
import time
import uuid
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor, TimeoutError as FutureTimeoutError, wait
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
import requests
from litellm import completion
from pydantic import BaseModel, ConfigDict, Field
from qdrant_client import QdrantClient

from augmentation.engine import Budget, BudgetExceeded
from augmentation.lane_synthetic import LaneSyntheticOperator, normalized
from demo_service.config import (
    BEHAVIORS,
    KNOWN_PRICING,
    Behavior,
    DemoConfig,
    SourceDoc,
)
from hybrid_search_rrf_dataset.fusion import (
    DenseOnlyStrategy,
    FusionStrategy,
    PureRRFStrategy,
    SparseOnlyStrategy,
)

from hybrid_search_rrf_dataset.retrieval.wave3 import (
    HOME_DEPOT_DENSE,
    HOME_DEPOT_SPARSE,
)

_POOL = ThreadPoolExecutor(max_workers=4, thread_name_prefix="demo-service")
"""Shared across the process. A late call from an expired deadline keeps
running here — it is never awaited again, so it cannot mutate a run's
already-read state; see the module docstring."""

_PIPELINE_TIMEOUT_S = 5.0

_FULL_EXTRACTOR: Any = None


def _full_extractor():
    """One all-engines extractor (spaCy + wordfreq + langid) for the process,
    lazy: typo spans and language sets are invisible to the regex default, and
    only the real-traffic behavior pays the model-load cost."""
    global _FULL_EXTRACTOR
    if _FULL_EXTRACTOR is None:
        from query_taxonomy.features import FeatureExtractor

        try:
            _FULL_EXTRACTOR = FeatureExtractor(engines=None)
        except Exception as exc:  # model not downloaded, engine missing
            raise DemoUnavailable(f"full extractor unavailable: {exc}") from exc
    return _FULL_EXTRACTOR


class DemoUnavailable(RuntimeError):
    """Any condition that should fall back to a `ReplayBundle`."""


class DeadlineExceeded(DemoUnavailable):
    pass


class CallLimitExceeded(DemoUnavailable):
    pass


class PipelineUnavailable(DemoUnavailable):
    pass


class NotEligibleToSave(RuntimeError):
    """Raised by `save()` for a candidate that has not cleared shape, guard,
    and answerability. Not a `DemoUnavailable` — this is a normal 409, not a
    reason to fall back to replay."""


class DeadlineGate:
    """A monotonic wall-clock budget for one stage."""

    def __init__(self, seconds: float) -> None:
        self._deadline = time.monotonic() + seconds
        self._seconds = seconds

    def remaining(self) -> float:
        return self._deadline - time.monotonic()

    def check(self) -> None:
        if self.remaining() <= 0:
            raise DeadlineExceeded(f"{self._seconds:.0f}s deadline exceeded")

    def run(self, fn: Callable[[], Any]) -> Any:
        """Run `fn` on the shared pool; raise `DeadlineExceeded` if it has
        not completed in time. `fn` keeps running in its worker thread if it
        does — the pool cannot preempt a blocking network call — but nothing
        here ever reads that late result."""
        remaining = self.remaining()
        if remaining <= 0:
            raise DeadlineExceeded(f"{self._seconds:.0f}s deadline already expired")
        future: Future = _POOL.submit(fn)
        try:
            return future.result(timeout=remaining)
        except FutureTimeoutError as exc:
            raise DeadlineExceeded(
                f"{self._seconds:.0f}s deadline exceeded"
            ) from exc


class CheckResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    shape_passed: bool
    shape_checks: tuple[dict, ...]
    guard_rejected: bool
    guard_reason: str | None
    answerable_status: str | None = None
    answerable_method: str = "presenter_inspection"

    @property
    def accepted(self) -> bool:
        return (
            self.shape_passed
            and not self.guard_rejected
            and self.answerable_status == "answers"
        )


class StrategyResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    strategy: str
    ranked_doc_ids: tuple[str, ...]
    ranked_titles: tuple[str, ...] = ()
    """Display titles for the top of the ranking; default keeps replay
    bundles recorded before this field loadable."""
    source_rank: int | None
    timed_out: bool = False


class RunRecord(BaseModel):
    """One test: live or replayed, complete or not. Saved to `export_dir` as
    one JSON document — the atomicity the design asks for ("a replay swaps
    query+checks+rankings together") falls out of this being one object with
    no field ever read independently of the rest."""

    model_config = ConfigDict(frozen=True)

    mode: str  # "live" | "replay"
    behavior_id: str
    collection: str
    doc_id: str
    query_id: str
    query_text: str
    targets: dict
    checks: CheckResult
    provenance: dict = Field(default_factory=dict)
    retrieval: tuple[StrategyResult, ...] | None = None
    fetch_limit: int
    saved_at: str | None = None

    @property
    def accepted(self) -> bool:
        return self.checks.accepted


ReplayBundle = RunRecord
"""A prepared replay is just a `RunRecord` with mode='replay', loaded whole
from `replays/<behavior_id>.json`. One schema for both — nothing to keep in
sync between a 'live shape' and a 'replay shape'."""


def _identifier_targets(entry: SourceDoc) -> dict:
    return {"spans": [{"feature": entry.identifier_feature, "min_count": 1}], "stats": []}


STRUCTURED_SHAPES: dict[str, dict] = {
    "negation": {
        "label": "Exclusion — “without X”",
        "prompt": (
            "\n\nThe query MUST express an exclusion with a negation word "
            "(no / without / not), e.g. \"security bar no drilling\"."
        ),
        "feature": "negation",
    },
    "operator_syntax": {
        "label": "Operator syntax — “A AND B”",
        "prompt": (
            "\n\nThe query MUST use explicit search-operator syntax: an "
            "uppercase AND or OR between two terms."
        ),
        "feature": "operator_syntax",
    },
}
"""Shape variants for the structured behavior: form, not content, so they are
legal on any document — an exclusion cannot manufacture relevance the way an
injected identifier would. Both banks fire on the regex default (measured)."""

_QWERTY_NEIGHBOURS = {
    "q": "w", "w": "e", "e": "r", "r": "t", "t": "y", "y": "u", "u": "i",
    "i": "o", "o": "p", "p": "o", "a": "s", "s": "d", "d": "f", "f": "g",
    "g": "h", "h": "j", "j": "k", "k": "l", "l": "k", "z": "x", "x": "c",
    "c": "v", "v": "b", "b": "n", "n": "m", "m": "n",
    "0": "9", "1": "2", "2": "3", "3": "4", "4": "5", "5": "6", "6": "7",
    "7": "8", "8": "9", "9": "8",
}


def mistype(surface: str, seed: int) -> str:
    """A deterministic near-miss of an identifier — one seeded QWERTY-neighbour
    substitution, case preserved. The mistake a real customer makes, made
    reproducibly."""
    from random import Random

    rng = Random(seed)
    positions = [i for i, ch in enumerate(surface) if ch.lower() in _QWERTY_NEIGHBOURS]
    if not positions:
        return surface + "1"   # nothing substitutable: append instead
    i = rng.choice(positions)
    replacement = _QWERTY_NEIGHBOURS[surface[i].lower()]
    if surface[i].isupper():
        replacement = replacement.upper()
    return surface[:i] + replacement + surface[i + 1:]


def _cell_targets(plan: dict) -> dict:
    """Union of every requirement's target, actionable or not — an unserved
    SELECT/REBUILD step becomes a target built from its own band rather than
    being silently dropped (design: 'do not silently treat them as fulfilled')."""
    spans: list[dict] = []
    stats: list[dict] = []
    for step in plan["steps"]:
        if step["operator"]:
            targets = step.get("targets") or {}
            spans.extend(targets.get("spans", []))
            stats.extend(targets.get("stats", []))
            continue
        for band in step["requirement"]:
            if band["is_span"]:
                spans.append({"feature": band["member"], "min_count": 1})
            else:
                stats.append({
                    "stat": band["member"],
                    "min_value": band["at_least"],
                    "max_value": band["at_most"],
                })
    return {"spans": spans, "stats": stats}


def contains_verbatim(text: str, surface: str) -> bool:
    return surface.lower() in text.lower()


class SearchTestDemo:
    def __init__(
        self,
        config: DemoConfig,
        *,
        qdrant_factory: Callable[[], QdrantClient] | None = None,
    ) -> None:
        self.config = config
        self._qdrant_factory = qdrant_factory or self._default_qdrant_factory
        self._op = LaneSyntheticOperator()
        self.mode = "unknown"
        self.replay_reason: str | None = None
        self._client: QdrantClient | None = None
        self._strategies: dict[str, FusionStrategy] | None = None
        self._doc_cache: dict[str, dict[str, str]] = {}
        self._taken_cache: set[str] | None = None
        self.reset()

    # ---- session -------------------------------------------------------

    def reset(self) -> None:
        """A fresh demo/rehearsal session: new budget, new call count, no
        in-progress candidate. Does not re-run preflight."""
        self._budget = Budget(
            self.config.engine.max_spend_usd,
            usd_per_mtok_in=self.config.engine.usd_per_mtok_in,
            usd_per_mtok_out=self.config.engine.usd_per_mtok_out,
        )
        self._llm_calls = 0
        self._current: dict[str, Any] = {}

    def _default_qdrant_factory(self) -> QdrantClient:
        return QdrantClient(
            url=self.config.qdrant_url,
            api_key=self.config.qdrant_api_key,
            cloud_inference=True,
            timeout=30,
        )

    def preflight(self) -> dict:
        """Every check the design's 'run before the timed presentation'
        preflight names. Never raises — a failure flips `self.mode` to
        'replay' with a reason, since /health must render, not 500."""
        reasons: list[str] = []

        try:
            r = requests.get(
                f"{self.config.pipeline_service_url}/health", timeout=_PIPELINE_TIMEOUT_S
            )
            r.raise_for_status()
        except requests.RequestException as exc:
            reasons.append(f"pipeline-service unreachable: {exc}")

        if not self.config.qdrant_url or not self.config.qdrant_api_key:
            reasons.append("QDRANT_CLOUD_URL/QDRANT_CLOUD_API_KEY not set")
        else:
            try:
                client = self._qdrant_factory()
                client.get_collection(self.config.collection)
                self._client = client
            except Exception as exc:
                reasons.append(f"Qdrant collection check failed: {exc}")

        if self.config.engine.model.startswith("anthropic/") and not os.environ.get(
            "ANTHROPIC_API_KEY"
        ):
            reasons.append("ANTHROPIC_API_KEY not set for the configured model")

        pricing = KNOWN_PRICING.get(self.config.engine.model)
        configured = (self.config.engine.usd_per_mtok_in, self.config.engine.usd_per_mtok_out)
        if pricing is None:
            reasons.append(f"no verified pricing on file for {self.config.engine.model!r}")
        elif pricing != configured:
            reasons.append(
                f"configured pricing {configured} does not match the verified "
                f"{pricing} for {self.config.engine.model!r}"
            )

        self.mode = "replay" if reasons else "live"
        self.replay_reason = "; ".join(reasons) or None
        return {"mode": self.mode, "reasons": reasons}

    def current_state(self) -> dict:
        """Whatever the current run has so far, tolerant of missing stages —
        the one shape every stage endpoint renders, live or replay."""
        c = self._current
        checks = c.get("checks")
        retrieval = c.get("retrieval")
        return {
            "behavior_id": c.get("behavior_id"),
            "features": list(c.get("features", ())),
            "query_id": c.get("query_id"),
            "query_text": c.get("query_text"),
            "guard_reason": c.get("guard_reason"),
            "checks": checks.model_dump() if checks is not None else None,
            "retrieval": [s.model_dump() for s in retrieval] if retrieval is not None else None,
            "spent_usd": round(self._budget.spent_usd, 4),
            "calls_used": self._llm_calls,
        }

    # ---- source ----------------------------------------------------------

    def _doc(self, doc_id: str) -> dict[str, str]:
        if doc_id not in self._doc_cache:
            frame = pd.read_parquet(
                self.config.data_dir / "corpus.parquet",
                filters=[("doc_id", "==", doc_id)],
            )
            if frame.empty:
                raise DemoUnavailable(f"pinned doc {doc_id!r} not in corpus")
            row = frame.iloc[0]
            self._doc_cache[doc_id] = {
                "doc_id": str(row.doc_id), "title": str(row.title), "text": str(row.text),
            }
        return self._doc_cache[doc_id]

    def _taken(self) -> set[str]:
        if self._taken_cache is None:
            real = pd.read_parquet(self.config.data_dir / "queries.parquet")["text"]
            minted = pd.read_parquet(self.config.data_dir / "minted_queries.parquet")["text"]
            self._taken_cache = {normalized(t) for t in pd.concat([real, minted]).astype(str)}
        return self._taken_cache

    def structured_options(self, entry: SourceDoc) -> list[dict]:
        """The structured-behavior variants this document supports: its own
        identifier (exact or mistyped) when it carries one, shape features
        everywhere."""
        options = []
        if entry.identifier_surface:
            wrong = mistype(entry.identifier_surface, self.config.corruption_seed)
            options.append({
                "id": "exact_id",
                "label": f"exact identifier — {entry.identifier_surface}",
            })
            options.append({
                "id": "wrong_id",
                "label": f"mistyped identifier — {wrong}",
            })
        options.extend(
            {"id": shape_id, "label": shape["label"]}
            for shape_id, shape in STRUCTURED_SHAPES.items()
        )
        return options

    def _pick_structured(self, entry: SourceDoc) -> tuple[str, ...]:
        """Roll the structure: one or two variants, chosen by the system so
        the presenter never curates the test. exact_id and wrong_id are
        mutually exclusive; shapes combine with either. Deliberately
        unseeded — every press of Generate is a different test."""
        from random import Random

        rng = Random()
        chosen: list[str] = []
        if entry.identifier_surface and rng.random() < 0.6:
            chosen.append(rng.choice(["exact_id", "wrong_id"]))
        shape_count = rng.randint(0 if chosen else 1, 1)
        if shape_count:
            chosen.append(rng.choice(list(STRUCTURED_SHAPES)))
        return tuple(chosen)

    def source(self) -> dict:
        """The document shelf: every pinned doc with the identifier the
        taxonomy detected in it, and which behaviors it supports."""
        documents = []
        for entry in self.config.docs:
            doc = self._doc(entry.doc_id)
            documents.append({
                "doc_id": entry.doc_id,
                "title": doc["title"],
                "excerpt": doc["text"][:280],
                "identifier_surface": entry.identifier_surface,
                "identifier_feature": entry.identifier_feature,
                "behaviors": [b.id for b in BEHAVIORS.values()],
                "structured_options": self.structured_options(entry),
            })
        return {
            "collection": self.config.collection,
            "fetch_limit": self.config.fetch_limit,
            "default_doc_id": self.config.default_doc_id,
            "documents": documents,
            "behaviors": [
                {"id": b.id, "label": b.label, "description": b.description}
                for b in BEHAVIORS.values()
            ],
        }

    def _titles_for(self, doc_ids: list[str]) -> dict[str, str]:
        if not doc_ids:
            return {}
        frame = pd.read_parquet(
            self.config.data_dir / "corpus.parquet",
            columns=["doc_id", "title"],
            filters=[("doc_id", "in", doc_ids)],
        )
        return dict(zip(frame["doc_id"].astype(str), frame["title"].astype(str), strict=False))

    # ---- pipeline-service calls -------------------------------------------

    def _pipeline_post(self, path: str, body: dict) -> dict:
        try:
            r = requests.post(
                f"{self.config.pipeline_service_url}{path}", json=body,
                timeout=_PIPELINE_TIMEOUT_S,
            )
            r.raise_for_status()
            return r.json()
        except requests.RequestException as exc:
            raise PipelineUnavailable(f"{path}: {exc}") from exc

    # ---- generate ----------------------------------------------------------

    def _call_model(self, prompt: str, gate: DeadlineGate) -> tuple[str, int, int]:
        if self._llm_calls >= self.config.max_llm_calls:
            raise CallLimitExceeded(f"{self.config.max_llm_calls} calls already used")
        messages = [{"role": "user", "content": prompt}]
        try:
            self._budget.reserve(messages, max_tokens=64)
        except BudgetExceeded as exc:
            raise DemoUnavailable(str(exc)) from exc

        def _do() -> Any:
            return completion(
                model=self.config.engine.model, messages=messages,
                max_tokens=64, temperature=0.7,
                timeout=max(gate.remaining() - 1, 1),
            )

        response = gate.run(_do)
        usage = getattr(response, "usage", None)
        prompt_tokens = getattr(usage, "prompt_tokens", 0) or 0
        completion_tokens = getattr(usage, "completion_tokens", 0) or 0
        try:
            self._budget.charge(prompt_tokens, completion_tokens)
        except BudgetExceeded as exc:
            raise DemoUnavailable(str(exc)) from exc
        self._llm_calls += 1
        text = (response.choices[0].message.content or "").strip()
        return text, prompt_tokens, completion_tokens

    def _behavior_prompt(
        self, behavior: Behavior, doc_text: str, entry: SourceDoc,
        features: tuple[str, ...] = (),
    ) -> tuple[str, dict]:
        """The prompt, and the pre-built (not yet aggregated) plan context —
        `cell_plan` for the conversational behavior, else empty."""
        base = self._op.instruction(
            "home-depot", doc_text, self.config.style_exemplars
        )
        if behavior.kind == "identifier":
            parts = []
            for f in features:
                if f == "wrong_id":
                    wrong = mistype(entry.identifier_surface, self.config.corruption_seed)
                    parts.append(
                        "\n\nThe query MUST include this exact text, character for "
                        f"character: \"{wrong}\" — a customer's mistyped version of "
                        "the real model number. Do NOT write the correct model "
                        f"number \"{entry.identifier_surface}\" anywhere. Weave the "
                        "mistyped one into a natural search query."
                    )
                elif f == "exact_id":
                    parts.append(
                        "\n\nThe query MUST include this exact text, character for "
                        f"character: \"{entry.identifier_surface}\". It is an identifier "
                        "from the document above. Weave it into a natural search query "
                        "rather than writing it alone."
                    )
                else:
                    parts.append(STRUCTURED_SHAPES[f]["prompt"])
            return base + "".join(parts), {}
        if behavior.kind == "messy":
            suffix = (
                "\n\nIMPORTANT: write the query IN SPANISH — a "
                "Spanish-speaking shopper in a hurry. A terse fragment of 2 to "
                "6 words, no full sentence, no punctuation, no courtesy. Brand "
                "names and model numbers stay as they are; every other word "
                "must be Spanish."
            )
            return base + suffix, {}

        plan = self._pipeline_post(
            "/augment/cell",
            {
                "cell": behavior.cell, "text": doc_text,
                "parent": {
                    "query_id": "demo", "dataset": self.config.collection,
                    "query": doc_text, "floors": [], "surfaces": [], "bank": "",
                },
            },
        )
        # looks_like is the cell's generator-facing description (d55d) — it is
        # what carries the unserved requirements (e.g. the stopword floor) that
        # no operator instruction states, so the model hears them too.
        parts = [f"Write it in this style: {plan['looks_like']}"] if plan.get("looks_like") else []
        parts += [s["instruction"] for s in plan["steps"] if s["operator"] and s["instruction"]]
        suffix = "\n\n" + "\n".join(parts) if parts else ""
        return base + suffix, {"cell_plan": plan}

    def generate(
        self, behavior_id: str, doc_id: str | None = None, feature: str | None = None
    ) -> dict:
        if self.mode == "replay":
            raise DemoUnavailable(self.replay_reason or "preflight marked this run as replay")
        behavior = BEHAVIORS[behavior_id]
        entry = self.config.doc_entry(doc_id)
        features: tuple[str, ...] = ()
        if behavior.kind == "identifier":
            options = [o["id"] for o in self.structured_options(entry)]
            if feature is not None:
                if feature not in options:
                    raise ValueError(
                        f"document {entry.doc_id} does not support the "
                        f"{feature!r} variant; it offers {options}"
                    )
                features = (feature,)
            else:
                features = self._pick_structured(entry)
        doc = self._doc(entry.doc_id)
        prompt, extra = self._behavior_prompt(behavior, doc["text"], entry, features)
        gate = DeadlineGate(self.config.generate_deadline_s)
        text, prompt_tokens, completion_tokens = self._call_model(prompt, gate)
        clean_text = text
        if behavior.kind == "messy":
            # Deterministic damage AFTER generation: the model writes clean
            # terse text; QwertyTypo turns one frequent word into exactly the
            # shape TypoBank recognises. Free, seeded, reproducible.
            text = self._damage(text)

        guard_reason = self._op.rejects(text, doc["text"], self._taken())
        if behavior.kind == "identifier":
            spans = []
            for f in features:
                if f == "exact_id":
                    spans.append({"feature": entry.identifier_feature, "min_count": 1})
                elif f in STRUCTURED_SHAPES:
                    spans.append({"feature": STRUCTURED_SHAPES[f]["feature"],
                                  "min_count": 1})
                # wrong_id: string checks in check(), no span target
            targets = {"spans": spans, "stats": []}
        elif behavior.kind == "messy":
            targets = {"spans": [], "stats": [
                {"stat": "length_words", "max_value": self.config.messy_max_words},
            ]}
        else:
            targets = _cell_targets(extra["cell_plan"])

        self._current = {
            "doc_id": entry.doc_id,
            "features": features,
            "clean_text": clean_text,
            "query_id": str(uuid.uuid4()),
            "query_text": text,
            "behavior_id": behavior.id,
            "targets": targets,
            "guard_reason": guard_reason,
            "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "attempts": self._current.get("attempts", 0) + 1 if self._current.get("behavior_id") == behavior.id else 1,
        }
        return {
            "mode": "live", "query_text": text, "guard_reason": guard_reason,
            "spent_usd": round(self._budget.spent_usd, 4),
            "calls_used": self._llm_calls,
        }

    def repair(self) -> CheckResult:
        if not self._current:
            raise ValueError("nothing to repair yet — choose a behavior to generate a query first")
        behavior = BEHAVIORS[self._current["behavior_id"]]
        entry = self.config.doc_entry(self._current["doc_id"])
        doc = self._doc(entry.doc_id)
        checks = self._current.get("checks")
        failed = [c for c in (checks.shape_checks if checks else ()) if not c.get("passed", True)]
        reasons = "; ".join(c.get("target", "") for c in failed) or (self._current.get("guard_reason") or "")
        prompt, _extra = self._behavior_prompt(
            behavior, doc["text"], entry, self._current.get("features", ())
        )
        prompt += (
            f"\n\nYour previous attempt was: \"{self._current['query_text']}\". "
            f"It failed: {reasons}. Try again, fixing exactly that."
        )
        gate = DeadlineGate(self.config.generate_deadline_s)
        text, prompt_tokens, completion_tokens = self._call_model(prompt, gate)
        clean_text = text
        if behavior.kind == "messy":
            text = self._damage(text)
        guard_reason = self._op.rejects(text, doc["text"], self._taken())
        self._current.update({
            "query_text": text, "clean_text": clean_text, "guard_reason": guard_reason,
            "prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens,
            "attempts": self._current["attempts"] + 1,
        })
        return self.check()

    # ---- check -------------------------------------------------------------

    def check(self) -> CheckResult:
        if not self._current:
            raise ValueError("no candidate yet — choose a behavior to generate a query first")
        text = self._current["query_text"]
        targets = self._current["targets"]
        response = self._pipeline_post("/generate/verify", {"text": text, "targets": targets})
        checks = list(response["checks"])
        passed = bool(response["passed"])

        behavior = BEHAVIORS[self._current["behavior_id"]]
        entry = self.config.doc_entry(self._current["doc_id"])
        if behavior.kind == "messy":
            messy_rows = self._measure_messy(text, self._current.get("clean_text", text))
            checks.extend(messy_rows)
            passed = passed and all(row["passed"] for row in messy_rows)
        features = self._current.get("features", ())
        if behavior.kind == "identifier" and "exact_id" in features:
            present = contains_verbatim(text, entry.identifier_surface)
            checks.append({
                "target": f"identifier verbatim: {entry.identifier_surface}",
                "measured": 1.0 if present else 0.0, "passed": present,
            })
            passed = passed and present
        if behavior.kind == "identifier" and "wrong_id" in features:
            wrong = mistype(entry.identifier_surface, self.config.corruption_seed)
            wrong_in = contains_verbatim(text, wrong)
            right_absent = not contains_verbatim(text, entry.identifier_surface)
            checks.append({
                "target": f"mistyped identifier present: {wrong}",
                "measured": 1.0 if wrong_in else 0.0, "passed": wrong_in,
            })
            checks.append({
                "target": f"correct identifier absent: {entry.identifier_surface}",
                "measured": 0.0 if right_absent else 1.0, "passed": right_absent,
            })
            passed = passed and wrong_in and right_absent

        result = CheckResult(
            shape_passed=passed, shape_checks=tuple(checks),
            guard_rejected=self._current["guard_reason"] is not None,
            guard_reason=self._current["guard_reason"],
            answerable_status=self._current.get("answerable_status"),
        )
        self._current["checks"] = result
        return result

    def _damage(self, text: str) -> str:
        """One QwertyTypo, seeded. A few successor seeds are tried when the
        seeded pick finds no eligible frequent word; unchanged text is not an
        error here — the measured-damage check will fail visibly instead."""
        from random import Random

        from augmentation.corruption import QwertyTypo

        typo = QwertyTypo()
        for seed in range(self.config.corruption_seed, self.config.corruption_seed + 5):
            damaged = typo.apply(text, Random(seed))
            if damaged != text:
                return damaged
        return text

    def _measure_messy(self, text: str, clean_text: str) -> list[dict]:
        """Measured rows for the real-traffic gate: damage must be detectable
        on the FINAL text, and the non-English language is measured on the
        UNDAMAGED text — a typo can fool langid on a short string (measured:
        'xup' pushed an English query to 'sv'), and the gate must not pass on
        an artifact of its own damage."""
        from query_taxonomy.taxonomy import FeatureGroup

        extractor = _full_extractor()
        features = extractor.resolve(text)
        corruption = features.spans.get(FeatureGroup.CORRUPTION, {})
        typos = len(corruption.get("typo", []))
        unknown = 0.0
        for by_bank in features.stats.values():
            for bank, stats in by_bank.items():
                for stat in stats:
                    if stat.name == "unknown_token_rate":
                        unknown = float(stat.value)
        languages = extractor.resolve(clean_text).language_set or []
        # The ORDERED language, not "anything non-English": langid is noisy on
        # short brand-heavy strings (measured: clean English "makita concrete
        # planer cup wheel" reads ['en', 'sv']), so the gate verifies what the
        # prompt asked for rather than passing on classifier noise.
        ordered = [lang for lang in languages if lang in ("es", "de")]
        return [
            {
                "target": "measurable damage: typo span, or unknown_token_rate >= 0.15",
                "measured": typos if typos else round(unknown, 3),
                "passed": typos >= 1 or unknown >= 0.15,
            },
            {
                "target": "ordered language measured: es or de (langid, not asserted)",
                "measured": ", ".join(languages) if languages else 0.0,
                "passed": bool(ordered),
            },
        ]

    def answerable(self, status: str) -> CheckResult:
        if status not in ("answers", "does_not_answer"):
            raise ValueError(f"unknown answerability status {status!r}")
        if "checks" not in self._current:
            raise ValueError("the candidate has not been checked yet — its measured checks must run before your inspection verdict")
        prior = self._current["checks"]
        result = CheckResult(
            shape_passed=prior.shape_passed, shape_checks=prior.shape_checks,
            guard_rejected=prior.guard_rejected, guard_reason=prior.guard_reason,
            answerable_status=status,
        )
        self._current["checks"] = result
        self._current["answerable_status"] = status
        return result

    # ---- retrieve ------------------------------------------------------

    def _default_strategies(self) -> dict[str, FusionStrategy]:
        if self._strategies is None:
            client = self._client or self._qdrant_factory()
            self._strategies = {
                str(cls.name): cls(
                    client, self.config.collection, HOME_DEPOT_DENSE, HOME_DEPOT_SPARSE, self.config.fetch_limit
                )
                for cls in (DenseOnlyStrategy, SparseOnlyStrategy, PureRRFStrategy)
            }
        return self._strategies

    def retrieve(
        self, strategies: dict[str, Callable[[str], dict[str, float]]] | None = None
    ) -> tuple[StrategyResult, ...]:
        if not self._current or "checks" not in self._current:
            raise ValueError("no checked candidate yet — generate and check a query before running retrieval")
        if self.mode == "replay" and strategies is None:
            raise DemoUnavailable(self.replay_reason or "replay mode")
        text = self._current["query_text"]
        by_name = strategies or {
            name: strat.rank for name, strat in self._default_strategies().items()
        }
        gate = DeadlineGate(self.config.retrieve_deadline_s)
        futures = {name: _POOL.submit(fn, text) for name, fn in by_name.items()}
        done, _pending = wait(futures.values(), timeout=max(gate.remaining(), 0))

        rankings: dict[str, tuple[str, ...]] = {}
        for name, future in futures.items():
            rankings[name] = tuple(future.result().keys()) if future in done else ()
        shown = {i for ids in rankings.values() for i in ids[:10]}
        titles = self._titles_for(sorted(shown))

        results = []
        run_doc = self._current["doc_id"]
        for name, future in futures.items():
            if future not in done:
                results.append(StrategyResult(
                    strategy=name, ranked_doc_ids=(), source_rank=None, timed_out=True,
                ))
                continue
            ids = rankings[name]
            source_rank = ids.index(run_doc) + 1 if run_doc in ids else None
            results.append(StrategyResult(
                strategy=name, ranked_doc_ids=ids,
                ranked_titles=tuple(titles.get(i, "") for i in ids[:10]),
                source_rank=source_rank,
            ))

        record = tuple(results)
        self._current["retrieval"] = record
        self._current["retrieved_at"] = datetime.now(UTC).isoformat(timespec="seconds")
        return record

    # ---- retain --------------------------------------------------------

    def _record(self) -> RunRecord:
        checks: CheckResult = self._current["checks"]
        retrieval = self._current.get("retrieval")
        return RunRecord(
            mode="live" if self.mode == "live" else "replay",
            behavior_id=self._current["behavior_id"],
            collection=self.config.collection, doc_id=self._current["doc_id"],
            query_id=self._current["query_id"], query_text=self._current["query_text"],
            targets=self._current["targets"], checks=checks,
            provenance={
                "features": list(self._current.get("features", ())),
                "model": self.config.engine.model,
                "generated_at": self._current["generated_at"],
                "attempts": self._current["attempts"],
                "spent_usd": round(self._budget.spent_usd, 4),
                "prompt_tokens": self._current.get("prompt_tokens", 0),
                "completion_tokens": self._current.get("completion_tokens", 0),
            },
            retrieval=retrieval, fetch_limit=self.config.fetch_limit,
        )

    def save(self) -> RunRecord:
        if "checks" not in self._current:
            raise NotEligibleToSave("no check has been run for the current candidate")
        record = self._record()
        failures = self._unmet_gates(record.checks)
        if failures:
            raise NotEligibleToSave("; ".join(failures))
        return self._export(record, accepted=True)

    @staticmethod
    def _unmet_gates(checks: CheckResult) -> list[str]:
        """Which of the three acceptance gates are still open, in the
        presenter's language — the save refusal names exactly what to do."""
        failures: list[str] = []
        if not checks.shape_passed:
            failed = [c["target"] for c in checks.shape_checks if not c.get("passed")]
            failures.append(f"shape checks failed ({'; '.join(failed)})")
        if checks.guard_rejected:
            failures.append(f"generation guard rejected it ({checks.guard_reason})")
        if checks.answerable_status is None:
            failures.append(
                "source answerability not inspected yet — read the query against "
                "the source excerpt and click 'Source answers it'"
            )
        elif checks.answerable_status != "answers":
            failures.append("you marked the source as not answering this query")
        return failures

    def save_rejected(self) -> RunRecord:
        """Retain a labeled rejected attempt. Not wired to an HTTP endpoint —
        the timed demo never needs it — but the property is real and tested."""
        if "checks" not in self._current:
            raise NotEligibleToSave("no check has been run for the current candidate")
        return self._export(self._record(), accepted=False)

    def _export(self, record: RunRecord, *, accepted: bool) -> RunRecord:
        self.config.export_dir.mkdir(parents=True, exist_ok=True)
        prefix = "" if accepted else "rejected_"
        path = self.config.export_dir / f"{prefix}{record.query_id}.json"
        stamped = record.model_copy(
            update={"saved_at": datetime.now(UTC).isoformat(timespec="seconds")}
        )
        path.write_text(stamped.model_dump_json(indent=2))
        return stamped

    def record_replay(self) -> Path:
        """Rehearsal tool: freeze the current completed run as the prepared
        replay for its behavior. Query, checks and rankings travel as the one
        bundle the fallback path loads whole."""
        if "retrieval" not in self._current:
            raise ValueError("a replay can only be recorded from a completed run — retrieval has not run yet")
        record = self._record().model_copy(update={"mode": "replay"})
        self.config.replay_dir.mkdir(parents=True, exist_ok=True)
        path = self.config.replay_dir / f"{record.behavior_id}.json"
        path.write_text(record.model_dump_json(indent=2))
        return path

    def suite(self) -> list[RunRecord]:
        if not self.config.export_dir.exists():
            return []
        records = [
            RunRecord.model_validate_json(p.read_text())
            for p in sorted(self.config.export_dir.glob("*.json"))
        ]
        return sorted(records, key=lambda r: r.saved_at or "", reverse=True)

    def rerun(self, query_id: str) -> RunRecord:
        """Re-run retrieval only, against the saved record's exact query
        text, and overwrite its retrieval block — the 'rerun against a
        changed retrieval configuration' property."""
        matches = [
            p for p in self.config.export_dir.glob(f"*{query_id}.json")
        ]
        if not matches:
            raise ValueError(f"no saved test with query_id {query_id!r}")
        record = RunRecord.model_validate_json(matches[0].read_text())
        self._current = {
            "doc_id": record.doc_id,
            "query_id": record.query_id, "query_text": record.query_text,
            "behavior_id": record.behavior_id, "targets": record.targets,
            "guard_reason": record.checks.guard_reason,
            "generated_at": record.provenance.get("generated_at", ""),
            "attempts": record.provenance.get("attempts", 1),
            "checks": record.checks,
            "answerable_status": record.checks.answerable_status,
        }
        self.retrieve()
        return self._export(self._record(), accepted=record.accepted)
