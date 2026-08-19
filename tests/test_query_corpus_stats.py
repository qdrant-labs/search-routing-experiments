"""Core-logic check for per-query corpus stats: the pair enumeration that
feeds PMI (distinct query term-pairs + the vocabulary that prunes documents)."""

from scripts.collection_features import QueryCorpusStats


def test_wanted_pairs_are_distinct_query_term_pairs():
    pairs, vocab = QueryCorpusStats._wanted_pairs(
        [["a", "b", "c"], ["a", "b"], ["a", "a"]]  # dup token, repeated pair
    )
    assert vocab == {"a", "b", "c"}
    # {a,a} is not a pair; {a,b} appears twice but is one distinct pair
    assert pairs == {
        frozenset(("a", "b")),
        frozenset(("a", "c")),
        frozenset(("b", "c")),
    }


def test_no_pairs_when_every_query_is_a_single_token():
    pairs, vocab = QueryCorpusStats._wanted_pairs([["x"], ["y"]])
    assert pairs == set()
    assert vocab == {"x", "y"}
