"""The final combiner's trust attribution and the extracted llm_a / r1 helpers."""

from scripts.final_combiner import (
    DENSE_QUERY_ROUTES,
    ROUTES,
    attribute,
    combine,
    gemini_l2_cost,
    project_to_router_frame,
    sweep,
)
from scripts.llm_a import parse, route_vector
from scripts.r1_rerank import r1_route_scores

_Z = {r: 0.0 for r in ROUTES}


def _row(l1=None, l2=None, r1=None, llm_a=None):
    return {"l1": {**_Z, **(l1 or {})}, "l2": {**_Z, **(l2 or {})},
            "r1": {**_Z, **(r1 or {})}, "llm_a": {**_Z, **(llm_a or {})}}


def test_gemini_l2_cost_counts_one_embed_per_dense_route():
    # 40 chars -> 10 tokens; 10 tokens * 2 dense routes / 1e6 * $0.15 = 3e-6
    rows = [{"query": "x" * 40}]
    assert len(DENSE_QUERY_ROUTES) == 2
    assert abs(gemini_l2_cost(rows) - 10 * 2 / 1e6 * 0.15) < 1e-12


def _named(qid, bucket, **legs):
    return {"dataset": "lane", "query_id": qid, "query": f"q{qid}", "bucket": bucket, **_row(**legs)}


def test_project_normalizes_to_unit_range_and_matches_router_schema():
    # combined dense = max(l1,l2)=1.0 + 0.8*r1(0.5) = 1.4; /(1+0.1+0.8)=1.9 -> 0.737
    rows = [_named("q1", "all_zero", l2={"dense_only": 1.0}, r1={"dense_only": 0.5})]
    frame = project_to_router_frame(rows, alpha=0.1, beta=0.8)
    assert {f"score_{r}" for r in ROUTES} <= set(frame.columns)  # router-native score triple
    triple = frame[[f"score_{r}" for r in ROUTES]].to_numpy()
    assert (triple >= 0).all() and (triple <= 1.0 + 1e-9).all()  # normalized to [0,1]
    assert abs(frame["score_dense_only"].iloc[0] - (1.0 + 0.8 * 0.5) / 1.9) < 1e-9
    assert frame["combiner_trust"].iloc[0] == "measurement"  # max(l1,l2) alone picks dense


def test_project_drops_only_all_zero_rows():
    # all legs zero -> no signal at all -> nothing to label -> dropped
    rows = [_named("q1", "all_zero"), _named("q2", "decisive", l2={"sparse_only": 1.0})]
    frame = project_to_router_frame(rows, alpha=0.1, beta=0.8)
    assert list(frame["query_id"]) == ["q2"]


def test_project_hedges_fused_tie_to_rrf():
    # sparse and pure_rrf tie at the top -> combine abstains -> hedge to pure_rrf, not dropped
    rows = [_named("q1", "all_tied", l2={"sparse_only": 1.0, "pure_rrf": 1.0})]
    frame = project_to_router_frame(rows, alpha=0.1, beta=0.8)
    assert list(frame["query_id"]) == ["q1"], "a fused tie must be hedged to rrf, not dropped"
    r = frame.iloc[0]
    triple = r[[f"score_{rt}" for rt in ROUTES]].astype(float)
    assert triple.idxmax() == "score_pure_rrf", "hedge must be the strict argmax"
    assert r["combiner_trust"] == "llm_a_only"


def test_combine_winner_and_margin():
    w, m, _ = combine(_row(l2={"dense_only": 1.0, "pure_rrf": 0.2}), 0.5, 0.5)
    assert w == "dense_only" and abs(m - 0.8) < 1e-9


def test_attribute_measurement():   # max(l1,l2) alone picks the winner
    r = _row(l2={"dense_only": 1.0})
    w, _, _ = combine(r, 0.5, 0.5)
    assert attribute(r, 0.5, 0.5, w) == "measurement"


def test_attribute_r1_assisted():   # measurement silent, r1 creates the winner
    r = _row(r1={"sparse_only": 0.8})
    w, _, _ = combine(r, 0.5, 0.5)
    assert w == "sparse_only" and attribute(r, 0.5, 0.5, w) == "r1_assisted"


def test_attribute_llm_a_only():    # only alpha*llm_a secures it (opinion, low trust)
    r = _row(llm_a={"dense_only": 1.0, "sparse_only": 0.5})
    w, _, _ = combine(r, 0.5, 0.5)
    assert w == "dense_only" and attribute(r, 0.5, 0.5, w) == "llm_a_only"


def test_attribute_ungrounded():    # nobody spoke
    r = _row()
    w, _, _ = combine(r, 0.5, 0.5)
    assert attribute(r, 0.5, 0.5, w) == "ungrounded"


def test_three_way_tie_is_ungrounded():   # l1=1 on every route + no other signal = not a decision
    r = _row(l1={"dense_only": 1.0, "sparse_only": 1.0, "pure_rrf": 1.0})
    w, m, _ = combine(r, 0.5, 0.5)
    assert w is None and m == 0.0 and attribute(r, 0.5, 0.5, w) == "ungrounded"


def test_llm_a_breaks_l1_three_way_tie():   # tied l1 + non-flat llm_a routes the vote through llm_a
    r = _row(l1={"dense_only": 1.0, "sparse_only": 1.0, "pure_rrf": 1.0},
             llm_a={"sparse_only": 1.0, "pure_rrf": 0.5})
    w, _, _ = combine(r, 0.5, 0.0)
    assert w == "sparse_only" and attribute(r, 0.5, 0.0, w) == "llm_a_only"


def test_sweep_shapes_and_goal():
    rows = [_row(l2={"dense_only": 1.0}),          # measurement
            _row(r1={"sparse_only": 0.8}),          # r1
            _row(llm_a={"pure_rrf": 1.0}),          # llm_a only
            _row()]                                 # ungrounded
    df = sweep(rows, [0.5], [0.5])
    row = df.iloc[0]
    assert row["decisive"] == 0.75 and row["grounded"] == 0.5   # 3/4 decisive, 2/4 grounded
    assert row["llm_a_only"] == 0.25


def test_route_vector_and_parse():
    assert route_vector("dense") == {"dense_only": 1.0, "sparse_only": 0.0, "pure_rrf": 0.0}
    assert route_vector("sparse") == {"dense_only": 0.0, "sparse_only": 1.0, "pure_rrf": 0.0}
    assert route_vector(None) == _Z                             # abstain -> no contribution
    route, why, abstain = parse("ROUTE: sparse\nWHY: rare statute token")
    assert route == "sparse" and "rare" in why and abstain is None
    route, why, abstain = parse("ABSTAIN: signals conflict")
    assert route is None and abstain == "signals conflict"


def test_r1_route_scores_rewards_ranking_relevant_first():
    doc_scores, judged = {"g": 0.9, "x": 0.1}, ["g"]
    rankings = {"dense_only": ["g", "x"], "sparse_only": ["x", "g"], "pure_rrf": ["g", "x"]}
    out = r1_route_scores(doc_scores, judged, rankings, min_relevance=1)
    assert out["dense_only"] > out["sparse_only"]   # dense ranks the relevant doc first
