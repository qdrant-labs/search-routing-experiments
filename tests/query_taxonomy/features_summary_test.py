from query_taxonomy.features import FeatureExtractor
from query_taxonomy.taxonomy import FeatureGroup


def test_summary_sections_by_group_and_domain():
    corpus = FeatureExtractor().extract(
        [
            "upgrade to v1.9.2 after CVE-2024-3094",
            "see DOI 10.1145/3539618 for details",
            "no identifiers in this one",
        ]
    )

    report = corpus.summary()

    # third query is tagged too: "no" is a sentence-marker (negation) hit
    assert report.splitlines()[0] == "queries: 3 tagged: 3 (100.0%)"
    assert "== structured_identifiers" in report
    assert "== sentence_markers" in report
    assert "-- tech" in report
    assert "-- media" in report
    assert "cve" in report and "top: CVE-2024-3094 (1)" in report
    assert str(corpus) == report


def test_groups_filter_limits_extraction():
    corpus = FeatureExtractor().extract(
        ["no identifiers in this one"],
        groups=[FeatureGroup.STRUCTURED_IDENTIFIERS],
    )
    assert corpus.summary() == "queries: 1 tagged: 0 (0.0%)"


def test_summary_handles_untagged_corpus():
    corpus = FeatureExtractor().extract(["walking shoes", "camping stoves"])
    report = corpus.summary()
    # spans absent -> untagged; stats always present for every query
    assert report.splitlines()[0] == "queries: 2 tagged: 0 (0.0%)"
    assert "== statistical_metrics (2 types)" in report
    assert "length.length_tokens  docs=2" in report
