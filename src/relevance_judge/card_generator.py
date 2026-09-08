"""One function: `generate_card(lane) -> LaneContext`, cached to disk.

The generator is a stochastic policy over lane metadata. Determinism is enforced
via temperature=0 and by freezing generated cards into a parquet keyed by lane —
a rerun reads the cache, so re-runs of `card_transfer.py` compare the SAME
generated instance across arms, not a fresh sample.
"""

from __future__ import annotations

import json
import re
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Final

import litellm
import pandas as pd
from litellm import completion

from relevance_judge.config import RelevanceJudgeConfig
from relevance_judge.judge import INSTRUCTION
from relevance_judge.lane_context import LaneContext
from relevance_judge.sources import Sources


_GENERATOR_INSTRUCTION: Final[str] = (
    "You write judging cards for a strict LLM relevance judge. The universal "
    "judge instruction you extend appears below in --INSTRUCTION--. Your card "
    "NARROWS what counts as relevant for one collection; it MAY NOT loosen the "
    "standard of evidence the instruction demands.\n\n"
    "Read the lane metadata in --METADATA-- and write four fields:\n"
    "TASK: what the retrieval job actually IS on this collection (one sentence).\n"
    "QUERIES: what a query looks like, including anything surprising about its "
    "shape (one sentence).\n"
    "GOLD: what the sample gold documents share that makes them count — describe "
    "only what the snippets and grade counts show (one sentence).\n"
    "JUDGING: the operative rule for THIS lane, phrased against the "
    "ASKED/EVIDENCE/MISSING lines the instruction defines. Name the lane's "
    "specific near-miss — the document a careless judge would wrongly accept "
    "(the IRRELEVANT snippets show it) — and the words-in-document test that "
    "separates it from a true positive. If the instruction below MIS-FRAMES this "
    "lane's task (the query asks no question; structured fields such as category "
    "lines are the document's own words), state the corrected framing — that is "
    "the ONLY way a card may raise recall. Never restate a rule the instruction "
    "below already carries. Four sentences max, plain declaratives.\n\n"
    "Do not invent facts the metadata does not carry. Do not exceed 100 words "
    "in any single field. Produce ONLY the four labelled lines, nothing else."
)

_FIELD = re.compile(r"^(TASK|QUERIES|GOLD|JUDGING)\s*:\s*(.*)$", re.IGNORECASE | re.MULTILINE)


def _parse(reply: str) -> LaneContext | None:
    """Four fields or nothing — a malformed reply is not a card."""
    fields = {name.upper(): " ".join(text.split())[:800]
              for name, text in _FIELD.findall(reply)}
    needed = ("TASK", "QUERIES", "GOLD", "JUDGING")
    if not all(k in fields and fields[k] for k in needed):
        return None
    return LaneContext(task=fields["TASK"], queries=fields["QUERIES"],
                       gold=fields["GOLD"], judging=fields["JUDGING"])


_SET_CUES = re.compile(
    r"\b(all|list of|which|kind of|type of|examples of|any|every|no |not)\b",
    re.IGNORECASE,
)


def _stratified_query_ids(queries: dict[str, str], k: int) -> list[str]:
    """Round-robin across four buckets (short/long × set-cue/no-cue) so a
    lane's distinctive failure mode — dbpedia's `all X` list queries lost among
    keyword topical queries; freshstack's negation lost in bare error phrases —
    is not silently under-represented by the first-k-that-appear sample."""
    buckets: dict[tuple[bool, bool], list[str]] = {(a, b): [] for a in (0, 1)
                                                   for b in (0, 1)}
    for qid, text in queries.items():
        long_q = len(text.split()) >= 4
        set_q = bool(_SET_CUES.search(text))
        buckets[(long_q, set_q)].append(qid)
    out: list[str] = []
    i = 0
    ordered = [b for b in buckets.values() if b]
    while len(out) < k and ordered:
        b = ordered[i % len(ordered)]
        if b:
            out.append(b.pop(0))
        if not b:
            ordered.pop(i % len(ordered))
            i -= 1  # ordered shrunk, don't skip the next survivor
        i += 1
    return out


def _lane_metadata(lane: str, sources: Sources, k_queries: int = 5,
                   k_neg_per_query: int = 3) -> str:
    """The metadata bundle the generator sees: docstring + STRATIFIED sample
    queries (round-robin across short/long × set-cue/no-cue buckets) + one
    top-grade gold snippet per query + up to `k_neg_per_query` human-judged
    IRRELEVANT snippets per query (the negative signal that lets JUDGING name
    a real failure mode instead of guessing one from the task genre) + the
    full grade distribution. Never carries route/label/score fields."""
    try:
        from dataset_registry import DATASETS
        card = next((d for d in DATASETS
                     if lane in {getattr(d, "name", None), getattr(d, "lane", None),
                                 getattr(d, "slug", None)}), None)
        docstring = (card.__doc__ or "").strip().split("\n\n")[0] if card else ""
    except Exception:  # noqa: BLE001 - registry lookup is best-effort context
        docstring = ""
    qrels = sources.base_qrels(lane)
    if qrels.empty:
        return f"LANE: {lane}\n(no qrels available)"

    top_grade = int(qrels["relevance"].max())
    grade_counts = qrels["relevance"].value_counts().sort_index().to_dict()

    # Draw a pool of distinct top-grade queries, load their text, THEN
    # stratify. Stratifying by text requires the text, and pulling more than
    # k_queries into the pool costs only lookups so the buckets aren't starved.
    top = qrels[qrels["relevance"] == top_grade]
    pool = top.drop_duplicates("query_id").head(k_queries * 4)
    pool_texts = sources.query_text(lane, set(pool["query_id"].astype(str)))
    query_ids = _stratified_query_ids(
        {qid: pool_texts.get(qid, "") for qid in pool["query_id"].astype(str)},
        k_queries,
    )
    if not query_ids:
        query_ids = pool["query_id"].astype(str).head(k_queries).tolist()
    # Gold: one top-grade doc per stratified query (from the same pool).
    gold_by_q = pool.set_index(pool["query_id"].astype(str))["doc_id"].astype(str)
    gold_pairs_raw = [(qid, gold_by_q[qid]) for qid in query_ids if qid in gold_by_q.index]

    # Multiple human-judged-irrelevant docs per SAME query id, so the bundle
    # exposes SEVERAL near-miss modes per query instead of one arbitrary
    # example — different failure axes surface (topical-adjacent, wrong-instance,
    # partial-attribute) when the qrels carry them. Some lanes (dbpedia) carry
    # neg doc_ids that never resolve to corpus text (entities not loaded), so
    # try up to `k_neg_per_query * 4` and keep the first `k_neg_per_query` that
    # actually have content — a blind head-slice silently ships zero snippets.
    neg = qrels[qrels["relevance"] < 1]
    neg_by_q: dict[str, list[str]] = {}
    for row in neg.itertuples(index=False):
        neg_by_q.setdefault(str(row.query_id), []).append(str(row.doc_id))
    neg_candidates = {
        qid: neg_by_q.get(qid, [])[:k_neg_per_query * 4] for qid in query_ids
    }
    queries = sources.query_text(lane, set(query_ids))
    docs = sources.corpus_text(
        lane,
        {gid for _, gid in gold_pairs_raw}
        | {did for cand in neg_candidates.values() for did in cand},
    )
    neg_pairs: list[tuple[str, str]] = []
    for qid in query_ids:
        kept = [did for did in neg_candidates[qid] if docs.get(did, "").strip()]
        neg_pairs.extend((qid, did) for did in kept[:k_neg_per_query])

    parts = [f"LANE: {lane}"]
    if docstring:
        parts.append(f"REGISTRY DOCSTRING:\n{docstring[:600]}")
    parts.append(f"RELEVANCE GRADES IN HUMAN QRELS: {grade_counts}; "
                 f"snippets below are top-grade ({top_grade}) positives and "
                 f"grade-0 human-judged irrelevant.")
    if queries:
        parts.append("SAMPLE QUERIES (stratified: short/long × set-cue/no-cue):")
        parts.extend(f"  - {queries.get(qid, '').strip()[:240]}"
                     for qid in query_ids if queries.get(qid))
    if gold_pairs_raw:
        parts.append(f"SAMPLE TOP-GRADE GOLD SNIPPETS (grade {top_grade}, one per query):")
        parts.extend(f"  - [query: {queries.get(qid, '')[:80]}] "
                     f"{docs.get(gid, '').strip()[:360]}"
                     for qid, gid in gold_pairs_raw if docs.get(gid))
    if neg_pairs:
        parts.append(f"SAMPLE HUMAN-JUDGED IRRELEVANT SNIPPETS (grade 0, up to "
                     f"{k_neg_per_query} per query — different near-miss modes):")
        parts.extend(f"  - [query: {queries.get(qid, '')[:80]}] "
                     f"{docs.get(did, '').strip()[:360]}"
                     for qid, did in neg_pairs if docs.get(did))
    return "\n".join(parts)


class CardGenerator:
    """Generate lane cards from metadata and cache them. Reruns hit the cache;
    `regenerate=True` forces a fresh call for that lane."""

    def __init__(self, config: RelevanceJudgeConfig | None = None,
                 sources: Sources | None = None,
                 cache_path: Path | None = None) -> None:
        self.config = config or RelevanceJudgeConfig()
        self.sources = sources or Sources(self.config)
        self.cache_path = cache_path or (
            self.config.artifacts / "generated_cards.parquet"
        )
        litellm.request_timeout = self.config.request_timeout_s
        litellm.suppress_debug_info = True

    def _cache(self) -> pd.DataFrame:
        if not self.cache_path.exists():
            return pd.DataFrame(columns=["lane", "task", "queries", "gold",
                                         "judging", "model", "generated_at"])
        return pd.read_parquet(self.cache_path)

    def _persist(self, frame: pd.DataFrame) -> None:
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(self.cache_path, index=False)

    def generate(self, lane: str, *, regenerate: bool = False) -> LaneContext:
        """One card for `lane`. Cached; the generator is temperature=0 but
        request-level nondeterminism (rare) is bypassed on cache hit."""
        cache = self._cache()
        hit = cache[cache["lane"] == lane]
        if not hit.empty and not regenerate:
            row = hit.iloc[-1]
            return LaneContext(task=row.task, queries=row.queries,
                               gold=row.gold, judging=row.judging)
        metadata = _lane_metadata(lane, self.sources)
        messages = [
            {"role": "system", "content":
                f"{_GENERATOR_INSTRUCTION}\n\n--INSTRUCTION--\n{INSTRUCTION}"},
            {"role": "user", "content": f"--METADATA--\n{metadata}"},
        ]
        # Same reasoning-off dance as the judge: luna is a reasoning model, and
        # WITHOUT this flag hidden reasoning tokens consume the completion
        # budget before any visible content is emitted (measured: 24-char
        # JUDGING for one lane, empty reply for the next). Two spellings for
        # the OpenRouter provider quirk (see judge.py:243).
        extra = (
            {
                "reasoning_effort": self.config.reasoning_effort,
                "extra_body": {"reasoning": {"effort": self.config.reasoning_effort}},
            }
            if self.config.reasoning_effort else {}
        )
        cap, reply = 1000, ""
        for attempt in range(3):
            response = completion(model=self.config.engine.model, messages=messages,
                                  max_tokens=cap, temperature=0.0, **extra)
            reply = response.choices[0].message.content or ""
            card = _parse(reply)
            if card is not None:
                break
            # a truncated reply parses to nothing — retry with double the room,
            # not just hope the next draw finishes.
            if getattr(response.choices[0], "finish_reason", None) == "length":
                cap *= 2
            time.sleep(1.0 * (attempt + 1))
        else:
            raise RuntimeError(f"card generator returned unparseable reply for {lane}: "
                               f"{reply[:400]!r}")
        row = {
            "lane": lane, "task": card.task, "queries": card.queries,
            "gold": card.gold, "judging": card.judging,
            "model": self.config.engine.model,
            "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        }
        cache = cache[cache["lane"] != lane]
        self._persist(pd.concat([cache, pd.DataFrame([row])], ignore_index=True))
        return card

    def generate_all(self, lanes: list[str], *, regenerate: bool = False
                     ) -> dict[str, LaneContext]:
        out = {}
        for lane in lanes:
            print(f"  generate: {lane}", end="", flush=True)
            out[lane] = self.generate(lane, regenerate=regenerate)
            print(f"  ok ({len(out[lane].judging)} chars judging)")
        return out


def _self_check() -> None:
    good = "TASK: entity retrieval.\nQUERIES: short keywords.\nGOLD: the entity page.\nJUDGING: relevant only when the page IS about the entity; a mention is not evidence."
    card = _parse(good)
    assert card is not None and card.task.startswith("entity")
    assert _parse("TASK: only.") is None
    print(f"generator self-check ok. cache: {CardGenerator().cache_path}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--lane", help="generate one card (prints as JSON)")
    ap.add_argument("--regenerate", action="store_true")
    args = ap.parse_args()
    if args.check:
        _self_check()
    elif args.lane:
        card = CardGenerator().generate(args.lane, regenerate=args.regenerate)
        print(json.dumps(card._asdict(), indent=2))
