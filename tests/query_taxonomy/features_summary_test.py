from query_taxonomy.features import CorpusIdentifierExtractor


def test_summary_groups_by_domain_and_counts():
    corpus = CorpusIdentifierExtractor().extract(
        [
            "upgrade to v1.9.2 after CVE-2024-3094",
            "see DOI 10.1145/3539618 for details",
            "no identifiers in this one",
        ]
    )

    report = corpus.summary()

    assert report.splitlines()[0] == "queries: 3 tagged: 2 (66.7%)"
    assert "-- tech" in report
    assert "-- media" in report
    assert "cve" in report and "top: CVE-2024-3094 (1)" in report
    assert str(corpus) == report


def test_summary_handles_untagged_corpus():
    corpus = CorpusIdentifierExtractor().extract(["hello there", "how are you"])
    assert corpus.summary() == "queries: 2 tagged: 0 (0.0%)"
