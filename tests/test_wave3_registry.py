from dataset_registry import (
    DATASETS,
    Grounding,
    QueryProvenance,
    Scope,
    SourceKind,
)
from hybrid_search_rrf_dataset.lanes import LANES
from hybrid_search_rrf_dataset.retrieval import (
    AmazonEsciLane,
    FinderLane,
    TechQaLane,
    Touche2020Lane,
    TrecCast2020HistoryLane,
    WandsLane,
)

WAVE3_NAMES = {
    "amazon-esci-en-hard",
    "wands",
    "finder",
    "trec-cast-2020-history",
    "techqa",
    "beir-touche-2020",
}


def test_wave3_build_next_sources_are_registered_as_retrieval_lanes():
    cards = {dataset.name: dataset.card for dataset in DATASETS}

    assert set(cards) >= WAVE3_NAMES
    assert set(LANES) >= WAVE3_NAMES
    assert isinstance(LANES["amazon-esci-en-hard"].source, AmazonEsciLane)
    assert isinstance(LANES["wands"].source, WandsLane)
    assert isinstance(LANES["finder"].source, FinderLane)
    assert isinstance(
        LANES["trec-cast-2020-history"].source, TrecCast2020HistoryLane
    )
    assert isinstance(LANES["techqa"].source, TechQaLane)
    assert isinstance(LANES["beir-touche-2020"].source, Touche2020Lane)
    assert LANES["wands"].corpus_target == 42_994

    amazon = cards["amazon-esci-en-hard"]
    assert (
        amazon.source,
        amazon.grounding,
        amazon.query_provenance,
        amazon.scope,
        amazon.non_trivial,
    ) == (
        SourceKind.URL,
        Grounding.QQ,
        QueryProvenance.HUMAN,
        Scope.SPECIFIC,
        True,
    )

    wands = cards["wands"]
    assert (wands.source, wands.grounding, wands.query_provenance) == (
        SourceKind.URL,
        Grounding.QQ,
        QueryProvenance.HUMAN,
    )

    finder = cards["finder"]
    assert (finder.source, finder.grounding, finder.query_provenance) == (
        SourceKind.HUGGINGFACE,
        Grounding.QQ,
        QueryProvenance.UNKNOWN,
    )

    cast = cards["trec-cast-2020-history"]
    assert (
        cast.source,
        cast.grounding,
        cast.query_provenance,
        cast.scope,
        cast.non_trivial,
    ) == (
        SourceKind.IR_DATASETS,
        Grounding.QQ,
        QueryProvenance.HUMAN,
        Scope.GENERAL,
        True,
    )

    techqa = cards["techqa"]
    assert (
        techqa.source,
        techqa.grounding,
        techqa.query_provenance,
        techqa.scope,
        techqa.non_trivial,
    ) == (
        SourceKind.HUGGINGFACE,
        Grounding.QQ,
        QueryProvenance.HUMAN,
        Scope.SPECIFIC,
        True,
    )

    touche = cards["beir-touche-2020"]
    assert (
        touche.source,
        touche.grounding,
        touche.query_provenance,
        touche.non_trivial,
    ) == (
        SourceKind.IR_DATASETS,
        Grounding.QQ,
        QueryProvenance.HUMAN,
        True,
    )
