from types import SimpleNamespace

import pandas as pd
import pytest

from scripts.legb import (
    PILOT_LANES,
    QWEN_INSTRUCTION,
    LegBPilot,
    e5_dense_cfg,
    gemini_dense_cfg,
    indexable,
    opensearch_distill_sparse_cfg,
    qwen_dense_cfg,
    stack_flags,
)


class _CountingPool:
    """Stands in for LabelledPool; counts calls to labels(). `classify` is a
    passthrough — the pilot now asks for classified labels, and the columns it
    bands on (kind/oracle) are supplied by each test's own frame."""

    def __init__(self, frame: pd.DataFrame) -> None:
        self._frame = frame
        self.calls = 0

    def labels(self) -> pd.DataFrame:
        self.calls += 1
        return self._frame

    def classify(self, frame: pd.DataFrame) -> pd.DataFrame:
        return frame


def _merged(**cols) -> pd.DataFrame:
    return pd.DataFrame(cols)


def test_flag_is_uncertifiable_unless_both_legs_certified():
    m = _merged(
        kind_leg1=["all_zero", "undecisive", "fake_tie"],
        kind_legb=["decisive", "decisive", "genuine_tie"],
        route_class_leg1=["", "", ""],
        route_class_legb=["dense", "sparse", "hybrid"],
    )
    assert list(stack_flags(m)) == ["uncertifiable"] * 3


def test_flag_reads_the_class_not_the_route():
    """dense_only and pure_rrf both live under class 'dense'/'hybrid'; the flag
    must fire on the class, so a within-class change is not stack_specific."""
    m = _merged(
        kind_leg1=["decisive", "decisive"],
        kind_legb=["decisive", "decisive"],
        route_class_leg1=["dense", "dense"],
        route_class_legb=["dense", "sparse"],
    )
    assert list(stack_flags(m)) == ["stack_robust", "stack_specific"]


def test_a_genuine_tie_holding_hybrid_is_robust():
    m = _merged(
        kind_leg1=["genuine_tie"], kind_legb=["genuine_tie"],
        route_class_leg1=["hybrid"], route_class_legb=["hybrid"],
    )
    assert list(stack_flags(m)) == ["stack_robust"]


def test_collection_never_reuses_the_paid_one():
    pilot = LegBPilot(client=None, dense_cfg=e5_dense_cfg())
    for lane in PILOT_LANES:
        assert "_legb_" in pilot.collection(lane)
        assert pilot.collection(lane).endswith("_routes")


def test_collection_is_namespaced_by_the_encoder(monkeypatch):
    """The regression this exists for: two encoders sharing one collection
    name collide on dense_legb's fixed vector size the moment you switch
    configs — measured live, Qdrant rejects the upload with 'expected dim:
    1024, got 4096' rather than ensure_collection catching it earlier."""
    monkeypatch.setenv("OPEN_ROUTER_API_KEY", "sk-test")
    e5 = LegBPilot(client=None, dense_cfg=e5_dense_cfg())
    qwen = LegBPilot(client=None, dense_cfg=qwen_dense_cfg())
    for lane in PILOT_LANES:
        assert e5.collection(lane) != qwen.collection(lane)


def test_e5_does_not_fork_extra_model_copies():
    """The regression this exists for: `parallel=N` loads N full copies of the
    model in N processes. bge-small (0.067GB) tolerates parallel=4 (~270MB);
    e5-large (2.24GB) does not (~9GB) — measured, this is what exhausted RAM."""
    assert e5_dense_cfg().parallel is None


def test_the_default_supply_is_native_only():
    """The regression this exists for: the Literal/default once said "v3",
    which _supply_mask never branches on — it silently fell through to the
    "both" case. A caller that never passes `supply=` must still get native."""
    frame = pd.DataFrame({"native": [True, False]})
    assert list(LegBPilot(None, e5_dense_cfg())._supply_mask(frame)) == [True, False]


def test_supply_selects_native_v2_or_both():
    frame = pd.DataFrame({"native": [True, True, False, False]})
    got = {
        s: list(LegBPilot(None, e5_dense_cfg(), supply=s)._supply_mask(frame))
        for s in ("native", "v2", "both")
    }
    assert got["native"] == [True, True, False, False]
    assert got["v2"] == [False, False, True, True]
    assert got["both"] == [True, True, True, True]


def test_sample_narrows_to_one_shape_at_a_fixed_seed():
    frame = pd.DataFrame({
        "dataset": ["antique"] * 6, "query_id": [str(i) for i in range(6)],
        "query": ["q"] * 6, "native": [True] * 6,
        "shape": ["all_tied", "all_tied", "all_tied", "routes_differ",
                  "routes_differ", "all_zero"],
    })
    pilot = LegBPilot(None, e5_dense_cfg(), sample=("all_tied", 2))
    pilot._pool = _CountingPool(frame)

    first = pilot._selection("antique")
    second = pilot._selection("antique")
    assert len(first) == 2
    assert set(first["query_id"]).issubset({"0", "1", "2"})
    assert list(first["query_id"]) == list(second["query_id"])  # same seed


def test_sample_is_a_noop_when_the_shape_has_fewer_rows_than_n():
    frame = pd.DataFrame({
        "dataset": ["antique"], "query_id": ["0"], "query": ["q"],
        "native": [True], "shape": ["all_tied"],
    })
    pilot = LegBPilot(None, e5_dense_cfg(), sample=("all_tied", 30))
    pilot._pool = _CountingPool(frame)
    assert len(pilot._selection("antique")) == 1


def test_a_short_draw_says_so(capsys):
    """A quietly-short set is what let a 30-row draw label 24 without comment."""
    frame = pd.DataFrame({
        "dataset": ["antique"], "query_id": ["0"], "query": ["q"],
        "native": [True], "shape": ["all_tied"],
    })
    pilot = LegBPilot(None, e5_dense_cfg(), sample=("all_tied", 30))
    pilot._pool = _CountingPool(frame)
    pilot._selection("antique")
    assert "asked 30" in capsys.readouterr().out


def test_natural_only_drops_supplemented_rows():
    """The 24-vs-30 bug: augmented rows live in the pool but not in the lane's
    queries.parquet, so QuerySubset dropped them silently after they were drawn."""
    frame = pd.DataFrame({
        "dataset": ["antique"] * 4,
        "query_id": ["1", "aug-corruption-light-2", "3", "aug-x-4"],
        "query": ["q"] * 4, "native": [True] * 4, "shape": ["all_tied"] * 4,
        "scored_against": ["natural", "supplemented", "natural", "supplemented"],
    })
    pilot = LegBPilot(None, e5_dense_cfg())
    pilot._pool = _CountingPool(frame)
    assert sorted(pilot._selection("antique")["query_id"]) == ["1", "3"]

    both = LegBPilot(None, e5_dense_cfg(), natural_only=False)
    both._pool = _CountingPool(frame)
    assert len(both._selection("antique")) == 4


def test_band_splits_on_the_ceiling():
    """At ceiling every route already ranks the judged doc first, so a better
    encoder has nothing left to win — 93% of native pilot-lane ties sit there,
    and an unbanded draw spends its budget on rows that cannot move up."""
    frame = pd.DataFrame({
        "dataset": ["antique"] * 4, "query_id": [str(i) for i in range(4)],
        "query": ["q"] * 4, "native": [True] * 4, "shape": ["all_tied"] * 4,
        "scored_against": ["natural"] * 4,
        "oracle": [1.0, 0.9995, 0.884, 0.116],
    })
    for band, expected in (("at_ceiling", ["0", "1"]), ("below", ["2", "3"])):
        pilot = LegBPilot(None, e5_dense_cfg(), band=band)
        pilot._pool = _CountingPool(frame)
        assert sorted(pilot._selection("antique")["query_id"]) == expected


def test_sample_reaches_kind_populations_shape_cannot():
    """`decisive` is a kind, not a shape — shape=='routes_differ' is only ~19%
    decisive, so it buys ~4 useless rows per useful one."""
    frame = pd.DataFrame({
        "dataset": ["antique"] * 4, "query_id": [str(i) for i in range(4)],
        "query": ["q"] * 4, "native": [True] * 4,
        "shape": ["routes_differ"] * 4, "scored_against": ["natural"] * 4,
        "kind": ["decisive", "undecisive", "undecisive", "decisive"],
    })
    pilot = LegBPilot(None, e5_dense_cfg(), sample=("decisive", 10))
    pilot._pool = _CountingPool(frame)
    assert sorted(pilot._selection("antique")["query_id"]) == ["0", "3"]


def test_leg1_labels_are_read_once_per_pilot():
    """The regression this exists for: `_selection` (called once per lane in
    both plan() and index_and_label()) and `compare()` each called
    `pool.labels()` fresh — 7 redundant full-pool re-reads/re-merges in one
    notebook run."""
    frame = pd.DataFrame({
        "dataset": ["antique", "antique"], "query_id": ["1", "2"],
        "query": ["a", "b"], "native": [True, True],
    })
    pilot = LegBPilot(None, e5_dense_cfg(), lanes=("antique",))
    pilot._pool = _CountingPool(frame)

    pilot._selection("antique")
    pilot._selection("antique")
    assert pilot._pool.calls == 1


def test_qwen_cfg_requires_the_api_key(monkeypatch):
    monkeypatch.delenv("OPEN_ROUTER_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="OPEN_ROUTER_API_KEY"):
        qwen_dense_cfg()


def test_qwen_cfg_carries_no_doc_prompt(monkeypatch):
    """Qwen3-Embedding's own convention: instruction on the query only,
    documents get no prefix at all — unlike e5's symmetric query:/passage:."""
    monkeypatch.setenv("OPEN_ROUTER_API_KEY", "sk-test")
    cfg = qwen_dense_cfg()
    assert cfg.cloud is True
    assert cfg.query_prompt == QWEN_INSTRUCTION
    assert cfg.doc_prompt == ""
    assert cfg.provider_options["openrouter-api-key"] == "sk-test"
    assert cfg.model_id.startswith("openrouter/")


def test_index_and_label_rejects_a_lane_not_in_this_pilot():
    """Validated before any I/O: client=None here would crash the moment the
    method actually tried to use it, so a clean ValueError proves the check
    runs first."""
    pilot = LegBPilot(client=None, dense_cfg=e5_dense_cfg(), lanes=("antique",))
    with pytest.raises(ValueError, match="crumb-legal-qa"):
        pilot.index_and_label(lane="crumb-legal-qa")


class _FakePool:
    """Stands in for LabelledPool: canned labels(), and classify() that adds the
    columns compare() reads without the real qrels-depth I/O.

    `classify` REFUSES an already-classified frame, mirroring the real one: its
    `_qrels_depth` merges a `depth`-bearing frame against the labels, so a second
    pass suffixes `depth` to depth_x/depth_y and raises KeyError. A permissive
    stub here hid exactly that bug once."""

    def __init__(self, leg1: pd.DataFrame) -> None:
        self._leg1 = leg1
        self.classify_calls = 0

    def labels(self) -> pd.DataFrame:
        return self._leg1

    def classify(self, frame: pd.DataFrame) -> pd.DataFrame:
        if "depth" in frame.columns:
            raise KeyError("depth")  # what pandas does on the doubled merge
        self.classify_calls += 1
        return frame.assign(kind="decisive", route_class="dense", route="dense_only",
                             margin=0.5, depth=1, oracle=1.0)


def test_compare_scopes_to_lanes_actually_labelled(tmp_path):
    """The regression this exists for: compare() used to filter leg-1 by every
    CONFIGURED lane, so a one-lane index_and_label run compared against lanes
    that were never tested — reading as a false 'no signal' verdict for lanes
    that simply had not run yet, rather than 'not run yet'."""
    leg1 = pd.DataFrame({
        "dataset": ["antique", "crumb-legal-qa"], "query_id": ["1", "2"],
        "native": [True, True], "score_dense_only": [0.5, 0.5],
    })
    legb = pd.DataFrame({  # only antique was actually indexed and labelled
        "dataset": ["antique"], "query_id": ["1"], "score_dense_only": [0.9],
    })
    legb.to_parquet(tmp_path / "labels.parquet")

    pilot = LegBPilot(
        None, e5_dense_cfg(), lanes=("antique", "crumb-legal-qa"), out_dir=tmp_path,
    )
    pilot._pool = _FakePool(leg1)

    merged = pilot.compare()
    assert set(merged["dataset"]) == {"antique"}


def test_compare_classifies_each_leg_exactly_once(tmp_path):
    """The regression this exists for: `_leg1_labels` returns classified rows,
    so compare()'s own `classify(leg1)` was a second pass — and the real
    classify raises KeyError('depth') on already-classified input."""
    leg1 = pd.DataFrame({
        "dataset": ["antique"], "query_id": ["1"],
        "native": [True], "score_dense_only": [0.5],
    })
    pd.DataFrame({
        "dataset": ["antique"], "query_id": ["1"], "score_dense_only": [0.9],
    }).to_parquet(tmp_path / "labels.parquet")

    pilot = LegBPilot(None, e5_dense_cfg(), lanes=("antique",), out_dir=tmp_path)
    pilot._pool = _FakePool(leg1)

    pilot.compare()   # would raise KeyError('depth') on a double pass
    # once for leg-1 (inside _leg1_labels), once for leg-2 — never twice on one
    assert pilot._pool.classify_calls == 2


def test_readout_reports_untested_lanes(capsys):
    pilot = LegBPilot(client=None, dense_cfg=e5_dense_cfg(),
                       lanes=("antique", "crumb-legal-qa"))
    merged = pd.DataFrame({
        "dataset": ["antique"], "kind_leg1": ["all_zero"], "kind_legb": ["decisive"],
        "flag": ["uncertifiable"], "route_class_legb": ["dense"], "dense_delta": [0.1],
    })
    pilot.readout(merged)
    out = capsys.readouterr().out
    assert "crumb-legal-qa" in out and "NOT YET TESTED" in out


class _IndexClient:
    """Answers the three calls `_index` makes of a client: which collections
    exist, how many points one holds, and (through the stub indexer) nothing
    else."""

    def __init__(self, live: set[str] | None = None, count: int = 0) -> None:
        self.live, self._count = live or set(), count

    def collection_exists(self, name: str) -> bool:
        return name in self.live

    def count(self, name, exact=True):
        del name, exact
        return SimpleNamespace(count=self._count)


class _StubIndexer:
    """Records the size of every upload; treats every doc as absent."""

    uploads: list[int] = []

    def __init__(self, *args, **kwargs) -> None:
        del args, kwargs

    def ensure_collection(self) -> None: ...

    def missing(self, docs):
        return docs

    def upload(self, docs, **kwargs):
        del kwargs
        _StubIndexer.uploads.append(len(docs))


def _corpus(n: int) -> pd.DataFrame:
    return pd.DataFrame({
        "doc_id": [str(i) for i in range(n)],
        "title": [""] * n,
        "text": [f"doc {i}" for i in range(n)],
    })


def _gemini_pilot(client) -> LegBPilot:
    return LegBPilot(client, gemini_dense_cfg(),
                     sparse_cfg=opensearch_distill_sparse_cfg())


@pytest.fixture
def openrouter_key(monkeypatch):
    monkeypatch.setenv("OPEN_ROUTER_API_KEY", "sk-test")


def test_dense_source_is_none_until_a_sibling_has_paid(openrouter_key):
    """`None` is the expensive answer — it is what the spend guard reads."""
    base = "antique_legb_gemini-embedding-001_routes"
    assert _gemini_pilot(_IndexClient()).dense_source("antique") is None
    pilot = _gemini_pilot(_IndexClient(live={base}))
    assert pilot.dense_source("antique") == base
    assert pilot.dense_source("antique") != pilot.collection("antique")


def test_a_local_dense_leg_never_claims_a_paid_source():
    """e5 embeds locally, so there is no provider bill to avoid and no cloud
    vector to copy — `dense_source` must not point at a sibling collection."""
    assert LegBPilot(_IndexClient(), e5_dense_cfg()).dense_source("antique") is None


def test_index_refuses_an_unpaid_lane_without_authorisation(openrouter_key):
    """The regression this exists for: a cloud dense slot embeds every document
    through the provider from inside the upsert, so `index("orcas")` and
    `index("quest")` look identical at the call site and differ by an invoice."""
    with pytest.raises(RuntimeError, match="allow_paid_dense"):
        _gemini_pilot(_IndexClient()).index("antique")


def test_index_uploads_in_slices(monkeypatch, openrouter_key):
    """The regression this exists for: `upload` embeds its whole item list
    before the first upsert, so one call per lane held every copied 3072-dim
    vector at once — measured 243MB per 2,000, which quest's 72,080 docs do
    not survive."""
    monkeypatch.setattr("scripts.legb.CorpusIndexer", _StubIndexer)
    _StubIndexer.uploads = []
    _gemini_pilot(_IndexClient())._index("antique", _corpus(4500))
    assert _StubIndexer.uploads == [2000, 2000, 500]


def test_a_complete_lane_is_settled_by_id_not_by_count(monkeypatch, openrouter_key):
    """The regression this exists for: the count test compared against
    `len(corpus)`, which no lane carrying a blank doc ever reaches — scirgen-geo-en
    finishes at 3,349 of 3,354 rows and would rescan forever, while a stale
    collection of unrelated points would pass as done."""
    monkeypatch.setattr("scripts.legb.CorpusIndexer", _StubIndexer)
    _StubIndexer.uploads = []
    client = _IndexClient(count=10_000)  # count says done, ids say otherwise
    _gemini_pilot(client)._index("antique", _corpus(10))
    assert _StubIndexer.uploads == [10]


def test_indexable_counts_the_points_a_lane_will_hold():
    """`upload` drops blank embed text and uuid5 collapses duplicate doc_ids,
    so neither is part of the target a run verifies against."""
    corpus = pd.DataFrame({
        "doc_id": ["a", "b", "c", "c"],
        "title": ["", "  ", "t", "t"],
        "text": ["real", "", "x", "x"],
    })
    assert indexable(corpus) == 2
