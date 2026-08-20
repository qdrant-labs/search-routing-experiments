import pandas as pd
import pytest

from hybrid_search_rrf_dataset.labels import (
    AcceptabilityLabels,
    RouteLabels,
    _augmented_rows,
    route_label,
)
from hybrid_search_rrf_dataset.fusion import StrategyName
from hybrid_search_rrf_dataset.golden import GoldenRoutingBuilder
from hybrid_search_rrf_dataset.objective import NDCGObjective, RouterObjective
from hybrid_search_rrf_dataset.retrieval import QuerySupplement


@pytest.fixture
def labels(tmp_path):
    """Two lanes with hand-computable scores.

    lane-a r1: dense 1.0 / rrf 0.15 / sparse 0.0  -> decisive, dense wins
    lane-a r2: dense 0.0 / rrf 0.90 / sparse 0.1  -> decisive, rrf wins
    lane-b r3: all zero                            -> all_zero, not decisive
    """
    frame = pd.DataFrame(
        {
            "dataset": ["lane-a", "lane-a", "lane-b"],
            "query_id": ["1", "2", "3"],
            "score_dense_only": [1.0, 0.0, 0.0],
            "score_pure_rrf": [0.15, 0.9, 0.0],
            "score_sparse_only": [0.0, 0.1, 0.0],
        }
    )
    out_dir = tmp_path / "route_labels"
    out_dir.mkdir()
    frame.to_parquet(out_dir / "labels.parquet", index=False)
    selection = pd.DataFrame({"dataset": [], "query_id": []})
    return RouteLabels(selection, out_dir=out_dir)


def test_decisive_margin():
    assert RouterObjective().decisive_margin == pytest.approx(0.4)
    assert RouterObjective(hit_weight=0.3, ndcg_weight=0.7).decisive_margin == float(
        "inf"
    )
    assert NDCGObjective().decisive_margin == float("inf")


def test_headroom_per_lane_and_pooled(labels):
    table = labels.headroom().set_index("dataset")

    lane_a = table.loc["lane-a"]
    # lane-a means: dense 0.5, rrf 0.525, sparse 0.05 -> best constant rrf
    assert lane_a["best_constant"] == "pure_rrf"
    assert lane_a["constant"] == pytest.approx(0.525)
    assert lane_a["oracle"] == pytest.approx(0.95)  # (1.0 + 0.9) / 2
    assert lane_a["headroom"] == pytest.approx(0.425)
    assert lane_a["decisive_share"] == pytest.approx(1.0)
    assert lane_a["all_zero_share"] == pytest.approx(0.0)

    lane_b = table.loc["lane-b"]
    assert lane_b["headroom"] == pytest.approx(0.0)
    assert lane_b["headroom_pct"] == pytest.approx(0.0)  # guarded 0/0
    assert lane_b["all_zero_share"] == pytest.approx(1.0)

    pooled = table.loc["POOLED"]
    assert pooled["labelled"] == 3
    # pooled means: dense 1/3, rrf 0.35, sparse 0.0333 -> best constant rrf
    assert pooled["best_constant"] == "pure_rrf"
    assert pooled["constant"] == pytest.approx(0.35)
    assert pooled["oracle"] == pytest.approx((1.0 + 0.9 + 0.0) / 3, abs=1e-3)


def test_headroom_decomposition_levels(labels):
    levels = labels.headroom_decomposition()
    assert list(levels["score"]) == [
        pytest.approx(0.35),  # global constant (pure_rrf pooled mean)
        pytest.approx((2 * 0.525 + 1 * 0.0) / 3),  # per-collection, n-weighted
        pytest.approx((1.0 + 0.9 + 0.0) / 3),  # oracle
    ]
    assert pd.isna(levels["gain_vs_previous_pct"].iloc[0])


def test_decisive_winners_split(labels):
    table = labels.decisive_winners().set_index("dataset")
    assert table.loc["lane-a", "dense_only"] == 1
    assert table.loc["lane-a", "pure_rrf"] == 1
    assert table.loc["lane-a", "sparse_only"] == 0
    assert "lane-b" not in table.index  # no decisive rows there
    assert table.loc["POOLED"].sum() == 2


@pytest.fixture
def shaped_frame():
    """One row per interesting shape, scores as (dense, rrf, sparse)."""
    return pd.DataFrame(
        {
            "score_dense_only": [1.0, 0.2, 1.0, 0.0, 1.0, 1.0],
            "score_pure_rrf": [0.6, 0.1, 1.0, 0.0, 0.7, 1.0],
            "score_sparse_only": [1.0, 0.9, 1.0, 0.0, 0.5, 0.2],
        },
        index=["near_top", "sparse_wins", "all_tied", "all_zero",
               "hit_parity", "two_way_tie"],
    )


def test_tolerance_zero_reproduces_stored_route(shaped_frame):
    # the d60 must-pass: at tolerance=0 the view's serve IS today's label,
    # including the two-way tie resolving dense (cheapest of the tied best).
    view = AcceptabilityLabels(shaped_frame, tolerance=0.0).frame()
    score_cols = [c for c in shaped_frame.columns if c.startswith("score_")]
    for row_name, row in shaped_frame.iterrows():
        scores = {c.removeprefix("score_"): row[c] for c in score_cols}
        stored = route_label(scores)
        served = view.loc[row_name, "serve"]
        # both nulls (all_zero) count as agreement — parquet stores each as NaN
        assert (pd.isna(served) and stored is None) or served == stored, row_name


def test_hit_parity_default_tolerance(shaped_frame):
    view = AcceptabilityLabels(shaped_frame)
    assert view.tolerance == pytest.approx(RouterObjective().ndcg_weight)


def test_acceptability_per_shape(shaped_frame):
    view = AcceptabilityLabels(shaped_frame).frame()  # tolerance 0.3

    tied = view.loc["all_tied"]
    assert bool(tied.ok_dense_only) and bool(tied.ok_pure_rrf)
    assert bool(tied.ok_sparse_only) and tied.serve == "sparse_only"

    zero = view.loc["all_zero"]
    assert pd.isna(zero.ok_dense_only) and pd.isna(zero.serve)

    # 0.7 sits exactly at hit parity (1.0 - 0.3): same top-1 band, acceptable
    parity = view.loc["hit_parity"]
    assert bool(parity.ok_pure_rrf) and not bool(parity.ok_sparse_only)
    assert parity.serve == "dense_only"

    # sparse alone clears; the others miss by more than the tolerance
    assert view.loc["sparse_wins", "serve"] == "sparse_only"
    assert not bool(view.loc["sparse_wins", "ok_dense_only"])


class _StubSource:
    """A RetrievalDataset stand-in: fixed frames, no corpus/retrieval work."""

    name = "nfcorpus"

    def __init__(self, queries: pd.DataFrame, qrels: pd.DataFrame) -> None:
        self._queries, self._qrels = queries, qrels

    def queries(self) -> pd.DataFrame:
        return self._queries

    def qrels(self) -> pd.DataFrame:
        return self._qrels

    def excluded(self) -> pd.DataFrame:
        return pd.DataFrame(columns=["query_id", "doc_id"])

    def provenance(self) -> pd.DataFrame:
        return pd.DataFrame(columns=["query_id", "provenance"])


def test_query_supplement_provenance_reports_only_the_extra_rows():
    source = _StubSource(
        queries=pd.DataFrame({"query_id": ["1"], "text": ["natural"]}),
        qrels=pd.DataFrame({"query_id": ["1"], "doc_id": ["D1"], "relevance": [1]}),
    )
    supplement = QuerySupplement(
        source,
        pd.DataFrame({
            "query_id": ["aug-1"], "text": ["augmented"], "provenance": ["doc_grounded"],
        }),
        pd.DataFrame({"query_id": ["aug-1"], "doc_id": ["D2"], "relevance": [1]}),
    )
    prov = supplement.provenance()
    assert list(prov["query_id"]) == ["aug-1"]
    assert list(prov["provenance"]) == ["doc_grounded"]


def test_query_supplement_provenance_empty_without_a_provenance_column():
    source = _StubSource(pd.DataFrame(columns=["query_id", "text"]), pd.DataFrame())
    supplement = QuerySupplement(
        source,
        pd.DataFrame({"query_id": ["aug-1"], "text": ["augmented"]}),
        pd.DataFrame(columns=["query_id", "doc_id", "relevance"]),
    )
    assert supplement.provenance().empty


def test_query_supplement_adds_rows_without_touching_the_source():
    source = _StubSource(
        queries=pd.DataFrame({"query_id": ["1"], "text": ["natural"]}),
        qrels=pd.DataFrame({"query_id": ["1"], "doc_id": ["D1"], "relevance": [1]}),
    )
    supplement = QuerySupplement(
        source,
        pd.DataFrame({"query_id": ["aug-1"], "text": ["augmented"]}),
        pd.DataFrame({"query_id": ["aug-1"], "doc_id": ["D2"], "relevance": [1]}),
    )
    assert list(supplement.queries()["query_id"]) == ["1", "aug-1"]
    assert list(source.queries()["query_id"]) == ["1"], "source frame is untouched"
    assert list(supplement.qrels()["doc_id"]) == ["D1", "D2"]


def test_augmented_rows_floor_based_ignores_admission():
    """A pre-d51 bare-label row counts even when `wanted` never admitted it —
    admission never applied to that generation (SPEC d61d)."""
    pool = pd.DataFrame([{
        "home_lane": "nfcorpus", "floor": "marker:greeting",
        "query_id": "q1", "query": "hi there", "credit_gate": "none",
    }])
    wanted = pd.DataFrame({"query_id": []})
    rows = _augmented_rows(pool, wanted, "nfcorpus", "floor_based")
    assert list(rows["query_id"]) == ["q1"]


def test_augmented_rows_floor_based_still_excludes_gated_rows():
    """2026-08-10 regression: a bare (non-cell) floor can still be gated —
    id:medical/id:network/etc are floor_based AND coherence_gate. Admission
    never applying to floor_based must not be read as 'never gated' too."""
    pool = pd.DataFrame([
        {"home_lane": "nfcorpus", "floor": "marker:greeting",
         "query_id": "ungated", "query": "hi there", "credit_gate": "none"},
        {"home_lane": "nfcorpus", "floor": "id:medical",
         "query_id": "gated", "query": "MED-123", "credit_gate": "coherence_gate"},
    ])
    wanted = pd.DataFrame({"query_id": []})
    rows = _augmented_rows(pool, wanted, "nfcorpus", "floor_based")
    assert list(rows["query_id"]) == ["ungated"]


def test_augmented_rows_cell_based_requires_admission():
    """A post-d51 cell-name row only counts once `wanted` (cell_selection)
    already holds it; an unreviewed pool row must not leak in (SPEC d61d)."""
    pool = pd.DataFrame([
        {"home_lane": "nfcorpus", "floor": "bare_concept_token",
         "query_id": "admitted", "query": "x", "credit_gate": "none"},
        {"home_lane": "nfcorpus", "floor": "bare_concept_token",
         "query_id": "not_admitted", "query": "y", "credit_gate": "none"},
    ])
    wanted = pd.DataFrame({"query_id": ["admitted"]})
    rows = _augmented_rows(pool, wanted, "nfcorpus", "cell_based")
    assert list(rows["query_id"]) == ["admitted"]


class _StubStrategy:
    """A FusionStrategy stand-in that records every query it's asked to
    rank — the proof an already-labelled query_id was never rescored."""

    def __init__(self, name: StrategyName, rankings: dict, seen: list) -> None:
        self.name = name
        self._rankings = rankings
        self._seen = seen

    def rank(self, query: str) -> dict:
        self._seen.append(query)
        return self._rankings.get(query, {})


def test_label_never_rescores_a_query_id_already_on_disk(tmp_path):
    """2026-08-10: label() must be incremental — a query_id already in
    labels.parquet is never re-ranked, only newly-arrived ones are. Proven
    by _StubStrategy's call log, not just the row count."""
    from augmentation.config import AugmentationPaths

    aug_paths = AugmentationPaths(data_dir=tmp_path)  # no pool.parquet here —
    # _augmented_rows must see an empty pool, never the real project's

    seen: list[str] = []
    rankings = {"alpha": {"D1": 1.0}, "beta": {"D2": 1.0}}

    def strategies():
        return (
            _StubStrategy(StrategyName.DENSE_ONLY, rankings, seen),
            _StubStrategy(StrategyName.PURE_RRF, rankings, seen),
            _StubStrategy(StrategyName.SPARSE_ONLY, rankings, seen),
        )

    selection = pd.DataFrame({
        "dataset": ["nfcorpus", "nfcorpus"], "query_id": ["q1", "q2"],
    })
    labels = RouteLabels(selection, out_dir=tmp_path, augmentation_paths=aug_paths)
    source = _StubSource(
        queries=pd.DataFrame({"query_id": ["q1", "q2"], "text": ["alpha", "beta"]}),
        qrels=pd.DataFrame({
            "query_id": ["q1", "q2"], "doc_id": ["D1", "D2"], "relevance": [1, 1],
        }),
    )

    first = labels.label(source, *strategies(), dataset="nfcorpus")
    assert sorted(first["query_id"]) == ["q1", "q2"]
    assert sorted(seen) == ["alpha", "alpha", "alpha", "beta", "beta", "beta"]

    # a 3rd query lands in the selection later (composition growth); q1/q2
    # must not be reranked even though the same dataset key is reused
    seen.clear()
    rankings["gamma"] = {"D3": 1.0}
    labels.selection = pd.DataFrame({
        "dataset": ["nfcorpus"] * 3, "query_id": ["q1", "q2", "q3"],
    })
    source3 = _StubSource(
        queries=pd.DataFrame({
            "query_id": ["q1", "q2", "q3"], "text": ["alpha", "beta", "gamma"],
        }),
        qrels=pd.DataFrame({
            "query_id": ["q1", "q2", "q3"], "doc_id": ["D1", "D2", "D3"],
            "relevance": [1, 1, 1],
        }),
    )

    second = labels.label(source3, *strategies(), dataset="nfcorpus")
    assert list(second["query_id"]) == ["q3"]
    assert seen == ["gamma", "gamma", "gamma"]

    on_disk = labels.load()
    assert sorted(on_disk["query_id"]) == ["q1", "q2", "q3"]


def test_label_with_force_rescores_everything(tmp_path):
    """`force=True` is still the escape hatch for a real redo (a strategy or
    objective change) — it must rescore query_ids that already have a row."""
    from augmentation.config import AugmentationPaths

    aug_paths = AugmentationPaths(data_dir=tmp_path)
    seen: list[str] = []
    rankings = {"alpha": {"D1": 1.0}}
    selection = pd.DataFrame({"dataset": ["nfcorpus"], "query_id": ["q1"]})
    labels = RouteLabels(selection, out_dir=tmp_path, augmentation_paths=aug_paths)
    source = _StubSource(
        queries=pd.DataFrame({"query_id": ["q1"], "text": ["alpha"]}),
        qrels=pd.DataFrame({"query_id": ["q1"], "doc_id": ["D1"], "relevance": [1]}),
    )
    def strategies():
        return (
            _StubStrategy(StrategyName.DENSE_ONLY, rankings, seen),
            _StubStrategy(StrategyName.PURE_RRF, rankings, seen),
            _StubStrategy(StrategyName.SPARSE_ONLY, rankings, seen),
        )

    labels.label(source, *strategies(), dataset="nfcorpus")
    seen.clear()
    labels.label(source, *strategies(), dataset="nfcorpus", force=True)
    assert seen == ["alpha", "alpha", "alpha"]


class _BoomStrategy(_StubStrategy):
    """A route that dies on one query — the network blip 90% into a lane."""

    def __init__(
        self, name: StrategyName, rankings: dict, seen: list, boom: str
    ) -> None:
        super().__init__(name, rankings, seen)
        self._boom = boom

    def rank(self, query: str) -> dict:
        if query == self._boom:
            raise RuntimeError(f"qdrant timed out on {query!r}")
        return super().rank(query)


def _lane(tmp_path, count: int = 6, judged: tuple[str, ...] | None = None):
    """A stub lane of `count` queries (`judged` narrows which have qrels) plus
    its call log and a strategy factory whose `boom` makes one query raise."""
    from augmentation.config import AugmentationPaths

    ids = [f"q{i}" for i in range(1, count + 1)]
    texts = [f"t{i}" for i in range(1, count + 1)]
    docs = [f"D{i}" for i in range(1, count + 1)]
    seen: list[str] = []
    rankings = {text: {doc: 1.0} for text, doc in zip(texts, docs)}
    kept = [i for i in range(count) if judged is None or ids[i] in judged]
    source = _StubSource(
        queries=pd.DataFrame({"query_id": ids, "text": texts}),
        qrels=pd.DataFrame({
            "query_id": [ids[i] for i in kept],
            "doc_id": [docs[i] for i in kept],
            "relevance": [1] * len(kept),
        }),
    )
    labels = RouteLabels(
        pd.DataFrame({"dataset": ["nfcorpus"] * count, "query_id": ids}),
        out_dir=tmp_path,
        augmentation_paths=AugmentationPaths(data_dir=tmp_path),
    )

    def strategies(boom: str | None = None):
        names = (
            StrategyName.DENSE_ONLY,
            StrategyName.PURE_RRF,
            StrategyName.SPARSE_ONLY,
        )
        if boom is None:
            return tuple(_StubStrategy(n, rankings, seen) for n in names)
        return tuple(_BoomStrategy(n, rankings, seen, boom) for n in names)

    return labels, source, seen, strategies


def test_label_keeps_the_chunks_that_finished_before_a_crash(tmp_path):
    """The property chunking buys: a crash costs one chunk, not the lane.
    Six queries, chunks of two, the fifth kills retrieval -> the two finished
    chunks are on disk, the third is not."""
    labels, source, _, strategies = _lane(tmp_path)

    with pytest.raises(RuntimeError):
        labels.label(
            source, *strategies(boom="t5"), dataset="nfcorpus", chunk_size=2
        )

    on_disk = labels.load()
    assert sorted(on_disk["query_id"]) == ["q1", "q2", "q3", "q4"]
    assert set(on_disk["dataset"]) == {"nfcorpus"}
    assert on_disk["shape"].notna().all()


def test_label_resumes_from_the_chunks_a_crash_left_behind(tmp_path):
    """Re-running after a partial failure re-ranks only the missing tail —
    the checkpoint is worth nothing if the retry pays for it again."""
    labels, source, seen, strategies = _lane(tmp_path)
    with pytest.raises(RuntimeError):
        labels.label(
            source, *strategies(boom="t5"), dataset="nfcorpus", chunk_size=2
        )

    seen.clear()
    out = labels.label(source, *strategies(), dataset="nfcorpus", chunk_size=2)

    assert sorted(set(seen)) == ["t5", "t6"], "a finished chunk was re-ranked"
    assert sorted(out["query_id"]) == ["q5", "q6"]
    assert sorted(labels.load()["query_id"]) == [f"q{i}" for i in range(1, 7)]


def test_label_below_one_chunk_makes_a_single_build_call(tmp_path, monkeypatch):
    """A small labelling call must not pay for chunking at all."""
    sizes: list[int] = []
    build = GoldenRoutingBuilder.build

    def counted(self, dataset, *args, **kwargs):
        sizes.append(len(dataset.queries()))
        return build(self, dataset, *args, **kwargs)

    monkeypatch.setattr(GoldenRoutingBuilder, "build", counted)
    labels, source, _, strategies = _lane(tmp_path, count=3)

    labels.label(source, *strategies(), dataset="nfcorpus")  # default 500
    assert sizes == [3]

    sizes.clear()
    labels.label(
        source, *strategies(), dataset="nfcorpus", force=True, chunk_size=2
    )
    assert sizes == [2, 1]


def test_label_tolerates_a_chunk_the_qrels_cover_none_of(tmp_path):
    """Only a call that labels nothing at all is the ValueError callers skip a
    lane on — an unjudged chunk in the middle of a good run is not."""
    labels, source, _, strategies = _lane(tmp_path, count=4, judged=("q1", "q4"))

    out = labels.label(source, *strategies(), dataset="nfcorpus", chunk_size=1)

    assert sorted(out["query_id"]) == ["q1", "q4"]
    assert sorted(labels.load()["query_id"]) == ["q1", "q4"]
