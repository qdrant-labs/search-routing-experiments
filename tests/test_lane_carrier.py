"""The lane rung: residual allocation by measured yield, doc-grounded minting
with real-doc keys, the dup/verbatim guards, and the labelling selection."""

import pandas as pd

from augmentation.config import AugmentationConfig, AugmentationPaths
from augmentation.constructed import ConstructedDocs
from augmentation.engine import AugmentationOutcome
from augmentation.loop import AugmentationLoop
from augmentation.pool import GeneratedPool
from augmentation.qrels import AugmentationQrels
from composition.objectives import LaneOrder
from composition.recipe import Recipe

RECIPE = Recipe.v3(target_total=1_000)


class ScriptedLane:
    """Returns the scripted query texts in order, all accepted."""

    def __init__(self, replies: list[str]) -> None:
        self.replies = list(replies)

    def run(self, instruction, prompt, targets, *, tool_loop=False):
        text = self.replies.pop(0) if self.replies else ""
        return AugmentationOutcome(
            text=text or None, accepted=bool(text), attempts=1, checks=(),
        )


def _loop(tmp_path, engine) -> AugmentationLoop:
    paths = AugmentationPaths(data_dir=tmp_path)
    return AugmentationLoop(
        pd.DataFrame(columns=["dataset", "query_id", "query", "checkable"]),
        config=AugmentationConfig(paths=paths, llm_workers=1), engine=engine,
        sheet_path=tmp_path / "sheet.parquet",
        pool=GeneratedPool(paths), qrels=AugmentationQrels(paths),
        docs=ConstructedDocs(paths),
    )


def _lane_fixture(tmp_path, docs: list[tuple[str, str]], queries: list[str]):
    source = tmp_path / "X"
    source.mkdir()
    pd.DataFrame({
        "doc_id": [d for d, _ in docs],
        "title": [""] * len(docs),
        "text": [t for _, t in docs],
    }).to_parquet(source / "corpus.parquet", index=False)
    pd.DataFrame({
        "query_id": [f"q{i}" for i in range(len(queries))],
        "query": queries,
    }).to_parquet(source / "queries.parquet", index=False)
    return source


def test_lane_order_allocates_by_yield_under_caps():
    yields = pd.DataFrame({
        "blind_labelled": [100, 100],
        "yield_dense": [0.1, 0.0],
        "yield_sparse": [0.5, 0.1],
        "yield_hybrid": [0.0, 0.0],
    }, index=pd.Index(["hi", "lo"], name="dataset"))
    pool = pd.DataFrame({
        "dataset": pd.Series(dtype=str),
        "route_class_any": pd.Series(dtype=str),
        "is_waste": pd.Series(dtype=bool),
    })
    order = LaneOrder(RECIPE).build(
        {"dense": 370, "sparse": 430, "hybrid": 100},
        yields, pool, capacity={"hi": 400, "lo": 100},
    )
    # hi stops at the 20% sparse share (90 rows / 0.5), then its fresh room
    # serves dense; lo's grounding capacity (100 docs) binds before its cap
    assert order.at["hi", "rows_to_mint"] == 400
    assert order.at["lo", "rows_to_mint"] == 100
    footer = order.loc[LaneOrder.FOOTER]
    assert footer["expected_sparse"] == 210  # 200 from hi + 10 from lo


def test_synthesize_lane_keys_queries_to_real_docs(tmp_path):
    source = _lane_fixture(tmp_path, [
        ("d1", "Paris is the capital of France. It hosts the Louvre."),
        ("d2", "The Nile is the longest river in Africa."),
        ("d3", "Mount Fuji is the highest mountain in Japan."),
    ], queries=["existing query"])
    loop = _loop(tmp_path, ScriptedLane(["capital of France?", "longest river africa"]))

    produced = loop.synthesize_lane("X", 2, source_dir=source)

    assert len(produced) == 2
    assert set(produced["operator"]) == {"lane_synthesize"}
    assert set(produced["floor"]) == {"lane:X"}
    keys = loop.qrels.load()
    assert len(keys) == 2 and keys["doc_id"].isin({"d1", "d2", "d3"}).all()
    assert set(keys["relevance"]) == {1}  # lane X is not in LANES
    # each query is keyed to ITS grounding doc
    assert set(produced["grounding_doc_id"]) == set(keys["doc_id"])

    # rerun: banked ids advance, spent grounding docs never reused
    more = loop.synthesize_lane("X", 2, source_dir=source, )
    assert len(more) == 0  # scripted engine is exhausted -> faults, no mints
    loop2 = _loop(tmp_path, ScriptedLane(["mount fuji height"]))
    third = loop2.synthesize_lane("X", 2, source_dir=source)
    assert len(third) == 1  # only one fresh grounding doc remained
    assert third.iloc[0]["query_id"] == "lane-X-2"  # banked ids advance
    assert not set(third["grounding_doc_id"]) & set(keys["doc_id"])


def test_synthesize_lane_rejects_dups_and_verbatim(tmp_path):
    text = "Paris is the capital of France. It hosts the Louvre."
    source = _lane_fixture(
        tmp_path, [("d1", text), ("d2", text)], queries=["existing query"]
    )
    loop = _loop(tmp_path, ScriptedLane(
        ["existing query", "paris is the capital of france"]
    ))
    produced = loop.synthesize_lane("X", 1, source_dir=source)
    assert produced.empty
    assert loop.qrels.load().empty


def test_lane_minted_selection_filters(tmp_path, monkeypatch):
    import scripts.label_routes_v3 as mod

    frame = pd.DataFrame({
        "query_id": ["lane-a-0", "lane-a-1", "lane-a-2", "aug-x-1"],
        "query": ["one", "two", "three", "four"],
        "operator": ["lane_synthesize"] * 3 + ["inject"],
        "home_lane": ["a"] * 4,
    })

    class FakePool:
        def load(self):
            return frame

    (tmp_path / "v3").mkdir()
    pd.DataFrame({"dataset": [], "query_id": []}).to_parquet(
        tmp_path / "labels_v2.parquet", index=False
    )
    pd.DataFrame({"dataset": ["a"], "query_id": ["lane-a-1"]}).to_parquet(
        tmp_path / "v3" / "labels.parquet", index=False
    )
    monkeypatch.setattr("augmentation.pool.GeneratedPool", lambda: FakePool())
    monkeypatch.setattr(mod, "V2_LABELS", tmp_path / "labels_v2.parquet")
    monkeypatch.setattr(mod, "V3_DIR", tmp_path / "v3")
    monkeypatch.setattr(mod, "AUGMENTED_DIR", tmp_path / "v3" / "augmented")

    # lane-a-0 passed; lane-a-1 passed but already labelled; lane-a-2 unjudged
    selection = mod.lane_minted_selection({"lane-a-0", "lane-a-1"})

    assert list(selection["query_id"]) == ["lane-a-0"]
    assert list(selection["dataset"]) == ["a"]
    assert list(selection["query"]) == ["one"]


def test_lane_operator_never_enters_the_dispatcher():
    from augmentation.lane_synthetic import LaneSyntheticOperator
    from augmentation.operators import OPERATOR_FAMILIES

    assert LaneSyntheticOperator not in OPERATOR_FAMILIES
    assert not LaneSyntheticOperator(AugmentationConfig()).serves("lane:limit")


def test_a_refusal_is_rejected_at_mint():
    from augmentation.lane_synthetic import LaneSyntheticOperator

    op = LaneSyntheticOperator(AugmentationConfig())
    assert op.rejects("I cannot generate a query for this document.",
                      "some doc text", set()) == "model refusal, not a query"
    assert op.rejects("capital of France?", "some doc text", set()) is None
