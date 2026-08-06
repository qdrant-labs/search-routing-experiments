"""The lane table (SPEC d39c): composition key → acquisition owner.

One entry per composition dataset that ships a relevance signal — qrels,
clicks (ORCAS), or an answer passage doubling as the gold doc (GooAQ).
Done lanes skip via `label()`'s idempotency; trec-dl-2022 is parked (d39g) but keeps its entry so coverage
reads it honestly. `min_relevance` binarizes graded qrels per d37(a)/d39(d);
the corpus policy is not a column — one threshold rule decides it (d39e).
"""

from typing import NamedTuple

from dataset_registry.core import DatasetName

from hybrid_search_rrf_dataset.retrieval import (
    AntiqueLane,
    BrightLane,
    ClercLane,
    CrumbLane,
    DBPediaLane,
    LimitLane,
    FreshStackLane,
    GooaqLane,
    LotteLane,
    MiraclLane,
    MSMarcoDev,
    NFCorpus,
    OrcasLane,
    QuestLane,
    RarbLane,
    RetrievalDataset,
    ScirgenGeoLane,
    TrecDL2022,
    WebFaqLane,
)


class Lane(NamedTuple):
    source: RetrievalDataset
    min_relevance: int = 1
    corpus_target: int | None = None
    """Explicit exception to the computed 20/80 recipe (`CorpusRecipe` in
    retrieval.py — answer_share/floor/ceiling are THE tunable parameters).
    Set only when the recipe cannot decide: relevant docs exceed its
    ceiling, or the lane's design requires the full corpus. None = the
    target is computed from the lane's own qrels at materialize time."""


LANES: dict[str, Lane] = {
    "beir-nfcorpus": Lane(NFCorpus()),
    # own materialize(): MSMarcoDev sizes from corpus_size, not the recipe;
    # corpus_target is its fallback for that param (100K matches the value
    # this lane was already indexed at) — pin it, don't let the recipe apply.
    "msmarco-passage-dev": Lane(MSMarcoDev(), corpus_target=100_000),
    "trec-dl-2022": Lane(TrecDL2022(), min_relevance=2),
    "rarb-math": Lane(RarbLane("math")),
    # rarb-code: indexed at 100K before the 20/80 recipe — kept, cost paid
    "rarb-code": Lane(RarbLane("code")),
    "bright-aops": Lane(BrightLane("aops")),
    "bright-leetcode": Lane(BrightLane("leetcode")),
    "bright-theoremqa-questions": Lane(BrightLane("theoremqa_questions")),
    # spelled out, not generated: this package does not import dataset_registry
    "bright-biology": Lane(BrightLane("biology")),
    "bright-earth-science": Lane(BrightLane("earth_science")),
    "bright-economics": Lane(BrightLane("economics")),
    "bright-pony": Lane(BrightLane("pony")),
    "bright-psychology": Lane(BrightLane("psychology")),
    "bright-robotics": Lane(BrightLane("robotics")),
    "bright-stackoverflow": Lane(BrightLane("stackoverflow")),
    "bright-sustainable-living": Lane(BrightLane("sustainable_living")),
    "bright-theoremqa-theorems": Lane(BrightLane("theoremqa_theorems")),
    "crumb-clinical-trial": Lane(CrumbLane("clinical_trial")),
    # exception: 108,782 relevant (median 23 relevant docs/query — the
    # benchmark's design) > recipe ceiling. Floor-plus-pad: every answer
    # force-included + ~11K distractors. Was full-232K; shrunk 2026-07-29
    # for embedding budget — the hard confusables (other queries' answers)
    # are all in the forced set either way.
    "crumb-code-retrieval": Lane(CrumbLane("code_retrieval"), corpus_target=120_000),
    "crumb-legal-qa": Lane(CrumbLane("legal_qa")),
    "crumb-paper-retrieval": Lane(CrumbLane("paper_retrieval")),
    "crumb-set-operation-entity-retrieval": Lane(
        CrumbLane("set_operation_entity_retrieval")
    ),
    "crumb-stack-exchange": Lane(CrumbLane("stack_exchange")),
    "crumb-theorem-retrieval": Lane(CrumbLane("theorem_retrieval")),
    "crumb-tip-of-the-tongue": Lane(CrumbLane("tip_of_the_tongue")),
    "quest": Lane(QuestLane()),
    # exception: full 50K by design — 46 relevant docs would compute to the
    # floor and delete the stress test the dataset exists for
    "limit": Lane(LimitLane(), corpus_target=50_000),
    "dbpedia-entity": Lane(DBPediaLane()),
    "miracl-en-dev": Lane(MiraclLane()),
    "orcas": Lane(OrcasLane()),
    "freshstack-angular": Lane(FreshStackLane("angular")),
    "freshstack-godot": Lane(FreshStackLane("godot")),
    "freshstack-langchain": Lane(FreshStackLane("langchain")),
    "freshstack-laravel": Lane(FreshStackLane("laravel")),
    "freshstack-yolo": Lane(FreshStackLane("yolo")),
    # levels 1-4; level 2 is "does not answer the question"
    "antique": Lane(AntiqueLane(), min_relevance=3),
    "lotte-technology-search": Lane(LotteLane("technology", "search")),
    "lotte-technology-forum": Lane(LotteLane("technology", "forum")),
    "webfaq-eng": Lane(WebFaqLane()),
    "scirgen-geo-en": Lane(ScirgenGeoLane()),
    "clerc": Lane(ClercLane()),
    "gooaq": Lane(GooaqLane()),
}

LANELESS: frozenset[DatasetName] = frozenset()
"""Registered datasets with no lane yet — a gap, not a design choice."""
