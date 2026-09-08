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
    AmazonEsciLane,
    BrightLane,
    ClercLane,
    CrumbLane,
    DBPediaLane,
    LimitLane,
    FreshStackLane,
    FinderLane,
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
    TechQaLane,
    Touche2020Lane,
    TrecCast2020HistoryLane,
    TrecDL2022,
    WebFaqLane,
    WandsLane,
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
    # grade 1 (95% of the qrel file) marks weakly-related docs; counting
    # them as successes lets every route "succeed" and manufactures ties
    "beir-nfcorpus": Lane(NFCorpus(), min_relevance=2),
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
    # min_relevance=2: grades 0/1/2 present; grade-1 partials are the
    # topical-adjacent "trial for a related condition" a strict judge correctly
    # rejects (SPEC d71, three-agent triangulation). Scoring against grade≥2
    # only stops counting those rejections as false negatives — same judge,
    # honest metric.
    "crumb-clinical-trial": Lane(CrumbLane("clinical_trial"), min_relevance=2),
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
    # min_relevance=2: BEIR DBpedia grade 1 = "partially relevant" (topically
    # adjacent entity, not THE entity), which a strict identity judge rejects;
    # scoring against grade 2 alone measures against the identity standard.
    "dbpedia-entity": Lane(DBPediaLane(), min_relevance=2),
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
    # wave 3: 394K relevant products across all hard queries exceed the recipe
    # ceiling, so LaneCorpora narrows metadata to composition query ids first.
    "amazon-esci-en-hard": Lane(AmazonEsciLane()),
    # deep-judged calibration lane; retain all candidate products.
    # min_relevance=2: WANDS grade 1 = "partial" (same category, wrong function
    # or missing attribute — "glam standard recliner" for "accent chair
    # recliner"). A strict attribute-match judge rejects those correctly; the
    # 5-lane paired A/B saw a card TRY to loosen for them and go UNSAFE.
    "wands": Lane(WandsLane(), corpus_target=42_994, min_relevance=2),
    "finder": Lane(FinderLane()),
    # conversational multi-turn: query = prior turns + [CURRENT] + utterance
    "trec-cast-2020-history": Lane(TrecCast2020HistoryLane()),
    # pilot substrate: rojagtap/tech-qa carries the 1,400 questions with
    # their linked Technote text, not the full 802K Technote corpus
    "techqa": Lane(TechQaLane()),
    "beir-touche-2020": Lane(Touche2020Lane()),
}

LANELESS: frozenset[DatasetName] = frozenset()
"""Registered datasets with no lane yet — a gap, not a design choice."""
