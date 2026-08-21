"""The concurrency window: `windowed_map`'s contract, and that the lane rung
and the coherence judge keep serial semantics with eight calls in flight.

Every engine here is a stub — no network, no model. Replies are keyed by the
INPUT rather than by call order, so a serial run and a concurrent run see the
same (input -> reply) map and any difference in the output is the code's.
"""

import threading
import time
from random import Random

import pandas as pd
import pytest

from augmentation.config import AugmentationConfig, AugmentationPaths
from augmentation.constructed import ConstructedDocs
from augmentation.core import AnswerKeyPath, AugmentedCandidate, CreditGate
from augmentation.engine import AugmentationOutcome, Spend, windowed_map
from augmentation.judge import CoherenceJudge
from augmentation.loop import AugmentationLoop
from augmentation.pool import GeneratedPool
from augmentation.qrels import AugmentationQrels

WORKERS = 8


class Boom(RuntimeError):
    """The simulated crash — raised from a worker, surfaces in the consumer."""


def _settled(baseline: int, timeout: float = 5.0) -> int:
    """Live thread count once the pool has joined — polls rather than sleeps,
    so a leaked or blocked worker fails as a count, never as a hang."""
    deadline = time.monotonic() + timeout
    while threading.active_count() > baseline and time.monotonic() < deadline:
        time.sleep(0.01)
    return threading.active_count()


class Concurrency:
    """Peak simultaneous calls, and whether two ever overlapped at all."""

    def __init__(self) -> None:
        self.inside = 0
        self.peak = 0
        self._lock = threading.Lock()

    def enter(self) -> None:
        with self._lock:
            self.inside += 1
            self.peak = max(self.peak, self.inside)

    def leave(self) -> None:
        with self._lock:
            self.inside -= 1


# --------------------------------------------------------------------------
# windowed_map


def test_submission_order_survives_random_latency():
    """The whole point: item k is consumed before item k+1 no matter which
    call finished first."""
    latency = {n: Random(0).random() * 0.02 for n in range(40)}

    def slow(n):
        time.sleep(latency[n])
        return n * 10

    items = list(range(40))
    assert list(windowed_map(slow, items, WORKERS)) == [(n, n * 10) for n in items]


def test_the_window_bounds_calls_in_flight():
    """At most `workers` calls at once — and more than one, or the window is
    doing nothing."""
    seen = Concurrency()

    def counted(n):
        seen.enter()
        time.sleep(0.01)
        seen.leave()
        return n

    assert len(list(windowed_map(counted, range(40), 3))) == 40
    assert 1 < seen.peak <= 3


def test_a_serial_window_never_starts_a_thread():
    baseline = threading.active_count()
    out = list(windowed_map(lambda n: n, [3, 1, 2], 1))
    assert out == [(3, 3), (1, 1), (2, 2)]
    assert threading.active_count() == baseline


def test_empty_items_and_more_workers_than_items():
    assert list(windowed_map(lambda n: n, [], WORKERS)) == []
    assert list(windowed_map(lambda n: n, [], 1)) == []
    assert list(windowed_map(lambda n: n * 2, [1, 2], 64)) == [(1, 2), (2, 4)]


def test_a_raising_call_propagates_and_shuts_the_pool_down():
    baseline = threading.active_count()

    def raiser(n):
        time.sleep(0.01)
        if n == 5:
            raise Boom(n)
        return n

    with pytest.raises(Boom):
        list(windowed_map(raiser, range(40), WORKERS))
    assert _settled(baseline) == baseline


def test_abandoning_the_generator_midway_never_deadlocks():
    baseline = threading.active_count()
    consumed = []
    stream = windowed_map(lambda n: (time.sleep(0.01), n)[1], range(40), WORKERS)
    for _, result in stream:
        consumed.append(result)
        if len(consumed) == 4:
            break
    assert consumed == [0, 1, 2, 3]
    stream.close()
    assert _settled(baseline) == baseline


def test_eight_workers_beat_serial_on_latency_bound_work():
    def slow(n):
        time.sleep(0.05)
        return n

    items = range(40)
    start = time.monotonic()
    list(windowed_map(slow, items, 1))
    serial = time.monotonic() - start
    start = time.monotonic()
    list(windowed_map(slow, items, WORKERS))
    parallel = time.monotonic() - start
    assert serial / parallel >= 4.0, f"serial {serial:.2f}s, {WORKERS}w {parallel:.2f}s"


# --------------------------------------------------------------------------
# the lane rung

DOC_TEXT = "document {doc_id} discussing matters of record at some length"


class DocKeyedLane:
    """Replies keyed by the grounding document, with a per-document latency —
    an order-independent fixture, so serial and concurrent runs are comparable.
    A reply of '' is an unaccepted outcome (a fault); 'BOOM' raises."""

    def __init__(
        self,
        replies: dict[str, str],
        latency: dict[str, float] | None = None,
        probe: Concurrency | None = None,
    ) -> None:
        self.replies = replies
        self.latency = latency or {}
        self.probe = probe
        self.calls: list[str] = []
        self._lock = threading.Lock()

    def _doc_id(self, instruction: str) -> str:
        return next(d for d in self.replies if f"document {d} " in instruction)

    def run(self, instruction, prompt, targets, *, tool_loop=False):
        doc_id = self._doc_id(instruction)
        if self.probe:
            self.probe.enter()
        time.sleep(self.latency.get(doc_id, 0.0))
        with self._lock:
            self.calls.append(doc_id)
        if self.probe:
            self.probe.leave()
        reply = self.replies[doc_id]
        if reply == "BOOM":
            raise Boom(doc_id)
        return AugmentationOutcome(
            text=reply or None, accepted=bool(reply), attempts=1
        )


def _corpus(source, doc_ids: list[str]) -> None:
    source.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({
        "doc_id": doc_ids,
        "title": [""] * len(doc_ids),
        "text": [DOC_TEXT.format(doc_id=d) for d in doc_ids],
    }).to_parquet(source / "corpus.parquet", index=False)
    pd.DataFrame({"query_id": ["q0"], "query": ["an existing query"]}).to_parquet(
        source / "queries.parquet", index=False
    )


def _loop(data_dir, engine, workers: int) -> AugmentationLoop:
    paths = AugmentationPaths(data_dir=data_dir)
    return AugmentationLoop(
        pd.DataFrame(columns=["dataset", "query_id", "query", "checkable"]),
        config=AugmentationConfig(paths=paths, llm_workers=workers),
        engine=engine,
        sheet_path=data_dir / "sheet.parquet",
        pool=GeneratedPool(paths), qrels=AugmentationQrels(paths),
        docs=ConstructedDocs(paths),
    )


def _submission_order(source, config: AugmentationConfig) -> list[str]:
    """The order `synthesize_lane` walks fresh documents in — same shuffle the
    loop performs, so a test can name the Nth document it will consume."""
    ids = pd.read_parquet(source / "corpus.parquet", columns=["doc_id"])["doc_id"]
    return list(ids.astype(str).sample(frac=1.0, random_state=config.seed))


def test_the_lane_rung_gives_eight_workers_the_serial_result(tmp_path):
    """The equivalence that makes `llm_workers` a resource dial: same pool,
    same keys, same ids, same order — one worker or eight."""
    doc_ids = [f"d{i:03d}" for i in range(14)]
    source = tmp_path / "source"
    _corpus(source, doc_ids)
    jitter = Random(0)
    latency = {d: jitter.random() * 0.03 for d in doc_ids}
    replies = {d: f"what does {d} say about the record?" for d in doc_ids}
    replies["d003"] = replies["d004"] = "the very same question twice"
    replies["d005"] = ""                      # a fault, mid-stream
    replies["d006"] = "an existing query"     # rejected by the dup guard

    frames, stores = [], []
    for name, workers in (("serial", 1), ("concurrent", WORKERS)):
        data_dir = tmp_path / name
        loop = _loop(data_dir, DocKeyedLane(replies, latency), workers)
        frames.append(loop.synthesize_lane(
            "X", 8, source_dir=source, max_consecutive_faults=None
        ))
        stores.append((loop.pool.load(), loop.qrels.load()))

    pd.testing.assert_frame_equal(frames[0], frames[1])
    pd.testing.assert_frame_equal(stores[0][0], stores[1][0])
    pd.testing.assert_frame_equal(stores[0][1], stores[1][1])
    produced = frames[1]
    assert list(produced["query_id"]) == sorted(produced["query_id"], key=_suffix)
    # the dup guard admitted exactly one of the two identical replies
    assert list(produced["query"]).count("the very same question twice") == 1


def _suffix(query_id: str) -> int:
    return int(str(query_id).rsplit("-", 1)[1])


def test_the_dup_guard_admits_one_of_two_concurrent_twins(tmp_path):
    """Both replies are in flight at the same time; `taken` is the consumer's,
    so the second one still loses."""
    doc_ids = [f"d{i:03d}" for i in range(6)]
    source = tmp_path / "source"
    _corpus(source, doc_ids)
    probe = Concurrency()
    engine = DocKeyedLane(
        {d: "one query, six documents" for d in doc_ids},
        latency=dict.fromkeys(doc_ids, 0.02),
        probe=probe,
    )
    loop = _loop(tmp_path / "run", engine, WORKERS)

    produced = loop.synthesize_lane(
        "X", 6, source_dir=source, max_consecutive_faults=None
    )

    assert probe.peak > 1, "the fixture never actually ran two calls at once"
    assert len(produced) == 1
    assert len(loop.pool.load()) == 1 and len(loop.qrels.load()) == 1


def test_the_fault_streak_trips_on_submission_order_not_finish_order(tmp_path):
    """The three faults are the SLOWEST calls, so a finish-order streak would
    never see them consecutively — and the fast accepted call behind them must
    not be banked."""
    doc_ids = [f"d{i:03d}" for i in range(12)]
    source = tmp_path / "source"
    _corpus(source, doc_ids)
    config = AugmentationConfig(paths=AugmentationPaths(data_dir=tmp_path / "run"))
    order = _submission_order(source, config)
    faulty, behind = order[2:5], order[5:]
    replies = {d: f"what does {d} say?" for d in doc_ids}
    replies.update(dict.fromkeys(faulty, ""))
    engine = DocKeyedLane(
        replies, latency={**dict.fromkeys(faulty, 0.05)}
    )
    loop = _loop(tmp_path / "run", engine, WORKERS)

    produced = loop.synthesize_lane(
        "X", 12, source_dir=source, max_consecutive_faults=3
    )

    assert list(produced["grounding_doc_id"]) == order[:2]
    banked = set(loop.qrels.load()["doc_id"])
    assert banked == set(order[:2])
    assert not banked & set(behind)


def test_a_crash_mid_chunk_reruns_without_duplicates(tmp_path):
    """The crash lands after one flushed chunk and five unbanked accepts: the
    rerun advances past the FLUSHED rows only, and no document grounds twice."""
    doc_ids = [f"d{i:03d}" for i in range(60)]
    source = tmp_path / "source"
    _corpus(source, doc_ids)
    config = AugmentationConfig(paths=AugmentationPaths(data_dir=tmp_path / "run"))
    order = _submission_order(source, config)
    replies = {d: f"what does {d} say?" for d in doc_ids}
    replies[order[55]] = "BOOM"

    loop = _loop(tmp_path / "run", DocKeyedLane(replies), WORKERS)
    with pytest.raises(Boom):
        loop.synthesize_lane("X", 60, source_dir=source, max_consecutive_faults=None)

    pool, keys = loop.pool.load(), loop.qrels.load()
    assert len(pool) == 50 and len(keys) == 50      # one chunk banked, five lost
    assert list(pool["query_id"]) == [f"lane-X-{i}" for i in range(50)]

    replies[order[55]] = f"what does {order[55]} say?"
    again = _loop(tmp_path / "run", DocKeyedLane(replies), WORKERS)
    produced = again.synthesize_lane(
        "X", 60, source_dir=source, max_consecutive_faults=None
    )

    pool, keys = again.pool.load(), again.qrels.load()
    assert not pool["query_id"].duplicated().any()
    assert not keys["query_id"].duplicated().any()
    assert not keys["doc_id"].duplicated().any()    # no document grounds twice
    assert len(pool) == 60 and len(keys) == 60
    # ids advanced past the flushed rows only — the five lost ones are reused
    assert list(produced["query_id"]) == [f"lane-X-{i}" for i in range(50, 60)]
    assert _answer_key_mismatches(pool, keys) == []


def _answer_key_mismatches(pool: pd.DataFrame, keys: pd.DataFrame) -> list[str]:
    """Pool rows whose banked answer key names a document other than their
    own grounding document — a key pointing at the wrong text."""
    grounding = dict(zip(
        pool["query_id"].astype(str),
        pool["grounding_doc_id"].astype(str),
        strict=True,
    ))
    return [
        str(row.query_id)
        for row in keys.itertuples(index=False)
        if str(row.query_id) in grounding
        and grounding[str(row.query_id)] != str(row.doc_id)
    ]


def test_a_crash_between_the_two_banking_writes_keeps_keys_and_rows_paired(
    tmp_path,
):
    """One flush writes twice. A crash after the first leaves the two stores
    disagreeing about how far the run got — and the rerun must not pair a
    query with another query's document."""
    doc_ids = [f"d{i:03d}" for i in range(60)]
    source = tmp_path / "source"
    _corpus(source, doc_ids)
    replies = {d: f"what does {d} say?" for d in doc_ids}
    loop = _loop(tmp_path / "run", DocKeyedLane(replies), WORKERS)

    writes: list[str] = []

    def crash_on_the_last_write(name, banked):
        def wrapped(*args, **kwargs):
            writes.append(name)
            if len(writes) == 4:      # the second flush's second write
                raise Boom(name)
            return banked(*args, **kwargs)
        return wrapped

    loop.pool.append = crash_on_the_last_write("pool", loop.pool.append)
    loop.qrels.mint_constructed_many = crash_on_the_last_write(
        "qrels", loop.qrels.mint_constructed_many
    )
    with pytest.raises(Boom):
        loop.synthesize_lane("X", 60, source_dir=source, max_consecutive_faults=None)

    again = _loop(
        tmp_path / "run",
        DocKeyedLane({d: f"a revised question about {d}" for d in doc_ids}),
        WORKERS,
    )
    again.synthesize_lane("X", 60, source_dir=source, max_consecutive_faults=None)

    pool, keys = again.pool.load(), again.qrels.load()
    assert not pool["query_id"].duplicated().any()
    assert not keys["doc_id"].duplicated().any()
    assert _answer_key_mismatches(pool, keys) == []


# --------------------------------------------------------------------------
# the coherence judge


class KeyedJudge:
    """One verdict per prompt, keyed by the prompt — `ask` is the judge's only
    model contact."""

    model = "stub-judge"

    def __init__(self, boom_on: str | None = None) -> None:
        self.boom_on = boom_on
        self.asked: list[str] = []
        self._lock = threading.Lock()

    def ask(self, instruction: str, prompt: str) -> tuple[str, Spend]:
        with self._lock:
            self.asked.append(prompt)
        if self.boom_on and f"Query: {self.boom_on}\n" in prompt:
            raise Boom(self.boom_on)
        return "yes - the document answers it", Spend()


def _gated(index: int) -> AugmentedCandidate:
    return AugmentedCandidate(
        query_id=f"syn-{index:03d}", query=f"question {index:03d}", floor="cell",
        operator="lane_synthesize", provenance="doc_grounded", generated_from="",
        parent_dataset="X", home_lane="X", meaning_preserved=False,
        answer_key=AnswerKeyPath.MINTED, attempts=1,
        credit_gate=str(CreditGate.COHERENCE_GATE),
    )


def _judge(tmp_path, engine, staged: list[AugmentedCandidate]) -> CoherenceJudge:
    paths = AugmentationPaths(data_dir=tmp_path)
    paths.augmentation_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({
        "doc_id": [f"constructed-{c.query_id}-1" for c in staged],
        "source_dataset": ["X"] * len(staged),
        "for_query": [c.query_id for c in staged],
        "text": [f"the document for {c.query_id}" for c in staged],
    }).to_parquet(paths.constructed_docs, index=False)
    return CoherenceJudge(
        engine,
        config=AugmentationConfig(paths=paths, llm_workers=WORKERS),
        docs=ConstructedDocs(paths),
    )


def test_the_judge_banks_every_verdict_once_under_eight_workers(tmp_path):
    staged = [_gated(i) for i in range(105)]
    pool = pd.DataFrame([c.model_dump() for c in staged])
    judge = _judge(tmp_path, KeyedJudge(), staged)

    counts = judge.run(pool)

    assert counts["judged"] == 105 and counts["passed"] == 105
    banked = judge.load()
    assert not banked["query_id"].duplicated().any()
    # submission order survives the two chunk flushes
    assert list(banked["query_id"]) == [c.query_id for c in staged]


def test_the_judge_reruns_only_the_unflushed_verdicts(tmp_path):
    """The crash lands after the first 100-row chunk is on disk: the rerun
    skips exactly those and banks no verdict twice."""
    staged = [_gated(i) for i in range(105)]
    pool = pd.DataFrame([c.model_dump() for c in staged])
    judge = _judge(tmp_path, KeyedJudge(boom_on="question 104"), staged)

    with pytest.raises(Boom):
        judge.run(pool)
    assert len(judge.load()) == 100

    again = _judge(tmp_path, KeyedJudge(), staged)
    counts = again.run(pool)

    assert counts["skipped"] == 100 and counts["judged"] == 5
    banked = again.load()
    assert len(banked) == 105
    assert not banked["query_id"].duplicated().any()


# --------------------------------------------------------------------------
# the engine's own shared state


class SerialityProbe:
    """Stands in for the shared extractor `Augmenter.accept` re-measures with:
    two threads inside at once trip the barrier, one at a time breaks it."""

    def __init__(self) -> None:
        self.overlapped = False
        self._pair = threading.Barrier(2)

    def resolve(self, text: str, **kwargs):
        try:
            self._pair.wait(timeout=1.0)
            self.overlapped = True
        except threading.BrokenBarrierError:
            pass
        from query_taxonomy.features import QueryFeatures

        return QueryFeatures(query_text=text, spans={}, stats={}, segments={})


def test_local_verification_never_runs_on_two_threads_at_once(
    tmp_path, monkeypatch
):
    """`Augmenter.accept` re-measures on one shared extractor — spaCy's pinned
    pipeline and its parse cache live behind it, and neither is thread-safe.
    The lane rung calls `run` from the worker thread, so the guard is here."""
    from augmentation.engine import Augmenter

    class Reply:
        class Message:
            content = "a query about the record"
        choices = [type("Choice", (), {"message": Message()})()]
        usage = type("Usage", (), {"total_tokens": 7})()

    monkeypatch.setattr("augmentation.engine.completion", lambda **kw: Reply())
    probe = SerialityProbe()
    engine = Augmenter(extractor=probe)
    doc_ids = [f"d{i:03d}" for i in range(4)]
    source = tmp_path / "source"
    _corpus(source, doc_ids)
    loop = _loop(tmp_path / "run", engine, WORKERS)

    loop.synthesize_lane("X", 4, source_dir=source, max_consecutive_faults=None)

    assert not probe.overlapped, "two threads re-measured through one extractor"
