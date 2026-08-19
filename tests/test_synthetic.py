"""The synthetic rung, with the model stubbed: a query written from a cell's
predicate and the document that answers it, banked together with an answer key
that depends on no corpus.
"""

import pandas as pd
import pytest
from query_taxonomy.features import FeatureExtractor

from augmentation.config import AugmentationConfig, AugmentationPaths
from augmentation.constructed import ConstructedDocs
from augmentation.engine import AugmentationOutcome
from augmentation.loop import AugmentationLoop
from augmentation.pool import GeneratedPool
from augmentation.qrels import AugmentationQrels
from augmentation.synthetic import SyntheticOperator
from taxonomy_generators.verify import verify

CELL = "symbol_pile_no_grammar"
QUERY = "cPGES lipoxinA4"
DOC = "cPGES and lipoxinA4 are lipid mediators. Both are studied in inflammation."


class ScriptedPair:
    """Query on the first call, document on the second — told apart by the
    prompt, exactly as `synthesize` sends them."""

    def __init__(self, query: str = QUERY, doc: str = DOC) -> None:
        self.query, self.doc = query, doc
        self.calls: list[str] = []
        self._extractor = FeatureExtractor(engines=None)

    def run(self, instruction, prompt, targets, *, tool_loop=False):
        self.calls.append(instruction)
        text = self.doc if prompt.startswith("Query:") else self.query
        report = verify(text, targets, extractor=self._extractor)
        return AugmentationOutcome(
            text=text, accepted=report.passed, attempts=1, checks=report.checks,
        )


def _loop(tmp_path, engine) -> AugmentationLoop:
    paths = AugmentationPaths(data_dir=tmp_path)
    sheet = tmp_path / "sheet.parquet"
    pd.DataFrame([{
        "slice": "cell", "floor": CELL, "amount": 10.0,
        "credit": 0.0, "missing": 5.0, "reason": "exhausted",
    }]).to_parquet(sheet, index=False)
    return AugmentationLoop(
        pd.DataFrame(columns=["dataset", "query_id", "query", "checkable"]),
        config=AugmentationConfig(paths=paths), engine=engine, sheet_path=sheet,
        pool=GeneratedPool(paths), qrels=AugmentationQrels(paths),
        docs=ConstructedDocs(paths),
    )


def test_a_synthetic_row_banks_query_document_and_key(tmp_path):
    engine = ScriptedPair()
    loop = _loop(tmp_path, engine)

    produced = loop.synthesize(CELL, 1, source_dataset="beir-nfcorpus")

    assert len(produced) == 1
    row = produced.iloc[0]
    assert row["operator"] == "synthesize"
    assert row["provenance"] == "synthetic"
    assert row["generated_from"] == "", "a synthetic row has no parent"
    assert row["answer_key"] == "minted"

    docs = loop.docs.load()
    assert list(docs["for_query"]) == [row["query_id"]]
    assert list(docs["source_dataset"]) == ["beir-nfcorpus"]
    assert row["grounding_doc_id"] == docs.iloc[0]["doc_id"]

    keys = loop.qrels.load()
    assert list(keys["source"]) == ["constructed"]
    assert list(keys["doc_id"]) == [row["grounding_doc_id"]]
    assert len(engine.calls) == 2, "one call for the query, one for the document"


def test_a_query_that_misses_its_bands_never_buys_a_document(tmp_path):
    """Query first is the whole point: a row that fails its cell costs one
    completion, not two."""
    engine = ScriptedPair(query="what is the capital of France")
    loop = _loop(tmp_path, engine)

    produced = loop.synthesize(CELL, 1, source_dataset="beir-nfcorpus")

    assert produced.empty
    assert loop.docs.load().empty, "no document for a rejected query"
    assert all("passage" not in call for call in engine.calls)


def test_a_rerun_advances_the_id_space_instead_of_rewriting(tmp_path):
    """Like `run()`, a second call produces MORE rows rather than topping up
    to a total — but it must never reissue a banked id or rewrite its
    document, which is what the written-for check is for."""
    loop = _loop(tmp_path, ScriptedPair())
    first = loop.synthesize(CELL, 1, source_dataset="beir-nfcorpus")
    second = loop.synthesize(CELL, 1, source_dataset="beir-nfcorpus")

    assert len(set(first["query_id"]) | set(second["query_id"])) == 2
    docs = loop.docs.load()
    assert len(docs) == 2 and docs["doc_id"].is_unique
    assert loop.qrels.load()["query_id"].is_unique


def test_the_rung_refuses_a_floor_that_is_not_a_cell(tmp_path):
    loop = _loop(tmp_path, ScriptedPair())
    with pytest.raises(ValueError, match="not a cell"):
        loop.synthesize("marker:greeting", 1, source_dataset="beir-nfcorpus")


def test_it_is_absent_from_the_dispatch_registry():
    """Registered, the dispatcher would hand it real parents at QUERY_ONLY and
    every cell would route to synthesis — the opposite of the gate's verdict."""
    from augmentation.operators import OPERATOR_FAMILIES

    assert SyntheticOperator not in OPERATOR_FAMILIES


def test_the_instruction_names_every_band_of_its_branch():
    op = SyntheticOperator()
    parent = pd.Series({"branch_index": 0})
    text = op.instruction(CELL, parent)
    assert "code_identifier" in text and "length_words" in text
    assert "\n" not in text.split("shapes that count:")[-1].split(")")[0], (
        "a sampled example with a newline would read as a second answer"
    )
