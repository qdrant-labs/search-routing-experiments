"""Model-engine bank checks — skipped per-class when the heavy artifacts
(`model` group for gliner2/torch, downloaded spaCy model for the tagger)
are missing. importorskip stays inside fixtures: at module level it would
skip the whole file and hide the other engine's tests."""

import pytest


@pytest.fixture(scope="module")
def gliner2():
    return pytest.importorskip("gliner2")


class TestGliner2Banks:
    def test_person_bank_finds_lowercase_person(self, gliner2):
        from query_taxonomy.entities import PersonBank

        spans = PersonBank().compute("who is chef mike ward")
        assert any("mike ward" in span.text for span in spans)

    def test_no_person_in_plain_instructional_query(self, gliner2):
        # NOTE: degenerate inputs are NOT safe negatives — the model tags
        # bare "42" as a person (high-confidence junk, same calibration
        # lesson as the audit); pin a realistic negative instead
        from query_taxonomy.entities import PersonBank

        assert PersonBank().compute("how to boil rice") == []

    def test_layered_temporal_regex_claims_first(self, gliner2):
        from query_taxonomy.entities import Gliner2TemporalBank
        from query_taxonomy.features import FeatureExtractor
        from query_taxonomy.logical import LOGICAL_BANKS
        from query_taxonomy.taxonomy import FeatureGroup, LogicalStructure

        extractor = FeatureExtractor(
            {FeatureGroup.LOGICAL_STRUCTURES: LOGICAL_BANKS + (Gliner2TemporalBank,)}
        )
        features = extractor.resolve("bitcoin price today")
        temporal = features.spans[FeatureGroup.LOGICAL_STRUCTURES][
            LogicalStructure.TEMPORAL
        ]
        # exactly one claim for "today": the regex layer wins, the model
        # backstop is blocked by the claim registry
        assert [span.text for span in temporal].count("today") == 1


@pytest.fixture(scope="module")
def pos_bank():
    pytest.importorskip("spacy")
    from query_taxonomy.metrics.pos import SPACY_MODEL, PosProfileBank

    import spacy

    if not spacy.util.is_package(SPACY_MODEL):
        pytest.skip(f"{SPACY_MODEL} not downloaded")
    return PosProfileBank()


class TestPosProfileBank:
    def test_histogram_and_shares(self, pos_bank):
        stats = {
            stat.name: stat.value
            for stat in pos_bank.compute("cheap laptops in berlin")
        }
        assert stats["pos_noun"] >= 1
        assert stats["pos_adj"] >= 1
        assert 0.0 < stats["open_class_share"] <= 1.0
        assert stats["verb_presence"] == 0.0

    def test_empty_text(self, pos_bank):
        stats = {stat.name: stat.value for stat in pos_bank.compute("")}
        assert stats["open_class_share"] == 0.0
        assert stats["pos_noun"] == 0.0

    def test_morphology_counts_inflections(self, pos_bank):
        from query_taxonomy.metrics.pos import MorphologyBank

        stats = {
            stat.name: stat.value
            for stat in MorphologyBank().compute("running shoes for flat feet")
        }
        # running -> run, shoes -> shoe, feet -> foot
        assert stats["inflected_count"] >= 2
        assert 0.0 < stats["inflected_share"] <= 1.0

    def test_syntactic_depth_separates_deep_from_flat(self, pos_bank):
        from query_taxonomy.metrics.pos import SyntacticDepthBank

        bank = SyntacticDepthBank()
        flat = {s.name: s.value for s in bank.compute("attention mechanism paper")}
        deep = {
            s.name: s.value
            for s in bank.compute(
                "the paper that introduced the attention mechanism used in transformers"
            )
        }
        assert deep["parse_depth"] > flat["parse_depth"]
        assert deep["clause_count"] >= 2
