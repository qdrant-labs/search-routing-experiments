"""Operators that serve their floor without a model: each `apply()` satisfies
its own targets and structural check, and a cell whose every step is
deterministic reaches the engine's local verify but never its completion.
"""

import re

import pandas as pd
import pytest
from query_taxonomy.features import FeatureExtractor

from augmentation.config import AugmentationConfig, AugmentationPaths
from augmentation.loop import AugmentationLoop
from augmentation.operators import (
    DecorateOperator,
    InjectOperator,
    SHARE_COLUMN,
    WORDS_COLUMN,
    OperatorSyntaxRewrite,
    StatRewrite,
    _shorten,
    default_operators,
)
from augmentation.pool import GeneratedPool
from augmentation.qrels import AugmentationQrels
from taxonomy_generators.verify import verify

from tests.test_augmentation_loop import IDENT, OneParent, PassThroughInject, _sheet


@pytest.fixture(scope="module")
def extractor() -> FeatureExtractor:
    return FeatureExtractor(engines=None)


def _parent(**overrides) -> pd.Series:
    row = {
        "dataset": "beir-nfcorpus", "query_id": "q1", "checkable": True,
        "query": "alternative medicine for chronic pain", "floors": [],
    }
    return pd.Series({**row, **overrides})


def test_decorate_lands_its_own_marker_target(extractor):
    op = DecorateOperator()
    parent = _parent()
    text = op.apply(parent, "marker:politeness", str(parent["query"]))
    report = verify(text, op.targets("marker:politeness", parent), extractor=extractor)
    assert report.passed, f"{text!r} did not carry a politeness span"
    assert not op.structural(parent, text, op.targets("marker:politeness", parent))


def test_decorate_reproduces_from_the_row(extractor):
    op, parent = DecorateOperator(), _parent()
    once = op.apply(parent, "marker:greeting", str(parent["query"]))
    assert once == op.apply(parent, "marker:greeting", str(parent["query"]))
    # a different row must not get the same phrase and position every time
    other = op.apply(_parent(query_id="q2"), "marker:greeting", str(parent["query"]))
    assert isinstance(other, str)


def test_inject_places_every_surface_and_passes_its_structural_check(extractor):
    op = InjectOperator()
    parent = _parent(
        surfaces=("cPGES", "lipoxinA4"), bank="code_identifier",
        grounding_doc_id="MED-1",
    )
    text = op.apply(parent, "symbol_pile_no_grammar", str(parent["query"]))
    targets = op.targets("symbol_pile_no_grammar", parent)
    assert "cPGES" in text and "lipoxinA4" in text
    assert verify(text, targets, extractor=extractor).passed
    assert not op.structural(parent, text, targets)


def test_inject_defers_when_the_bank_will_not_claim_a_bare_append():
    """version_string is keyword-gated — `SPSS version 22.0` is a declared
    positive, `p = 0.05` a declared negative. Appending a bare `17.2` is
    literally present and still measures zero, so the row must reach the model
    rather than drop against a target no placement of ours can meet."""
    op = InjectOperator()
    parent = _parent(
        query="home child care definition", surfaces=("17.2",),
        bank="version_string", grounding_doc_id="D-1",
    )
    assert op.apply(parent, "version_pinned_technical", str(parent["query"])) is None


def test_inject_still_serves_a_bank_that_claims_its_surface_standalone(extractor):
    """The other side of the same check: code_identifier needs no gate word,
    so the append stands and the row costs nothing."""
    op = InjectOperator()
    parent = _parent(surfaces=("cPGES",), bank="code_identifier")
    text = op.apply(parent, "code_symbol_named_in_prose", str(parent["query"]))
    assert text is not None
    assert verify(
        text, op.targets("code_symbol_named_in_prose", parent), extractor=extractor
    ).passed


def test_inject_is_idempotent_when_a_surface_is_already_there():
    op = InjectOperator()
    parent = _parent(surfaces=("cPGES",), bank="code_identifier")
    already = "alternative medicine cPGES"
    assert op.apply(parent, "id:tech", already) == already


def test_operator_syntax_uppercases_written_coordination():
    op = OperatorSyntaxRewrite()
    parent = _parent(query="python and rust benchmarks")
    assert op.apply(parent, "logical:operator_syntax", str(parent["query"])) == (
        "python AND rust benchmarks"
    )


def test_operator_syntax_promotes_a_comma_list():
    op = OperatorSyntaxRewrite()
    text = "python, rust, go benchmarks"
    assert op.apply(_parent(), "logical:operator_syntax", text) == (
        "python AND rust AND go benchmarks"
    )


def test_operator_syntax_never_emits_not():
    op = OperatorSyntaxRewrite()
    parent = _parent(query="python and rust but not go")
    text = op.apply(parent, "logical:operator_syntax", str(parent["query"]))
    assert not op.structural(parent, text, op.targets("logical:operator_syntax", parent))


def test_operator_syntax_defers_when_there_is_nothing_to_rewrite():
    """No conjunction and no comma: the model still has a job, so `apply` must
    say so rather than return the text unchanged."""
    op = OperatorSyntaxRewrite()
    assert op.apply(_parent(), "logical:operator_syntax", "plain query text") is None


def test_the_cut_keeps_surfaces_and_gold_terms_and_hits_the_limit():
    text = "how to configure the nginx reverse proxy for a docker container 2.1.3"
    out = _shorten(text, limit=6, protected=("2.1.3",), gold="nginx reverse proxy")
    assert len(re.findall(r"\w+", out)) <= 6
    assert "2.1.3" in out, "a copied surface is never droppable"
    assert "nginx" in out, "a term the gold document uses outranks filler"
    assert " the " not in f" {out} ", "stopwords go first"


def test_the_cut_never_breaks_a_surface_into_tokens():
    """`2.1.3` is three `\\w+` tokens; cutting must not rebuild it as '2 1 3'."""
    out = _shorten("please find version 2.1.3 of the package", limit=4,
                   protected=("2.1.3",), gold="")
    assert "2.1.3" in out


def test_stat_rewrite_cuts_deterministically_but_defers_expansion():
    op = StatRewrite()
    long_parent = _parent(
        query="how to configure the nginx reverse proxy for a docker container",
        surfaces=(), gold_text="nginx proxy docker", stat_value=11.0,
    )
    cut = op.apply(long_parent, "version_pinned_technical", str(long_parent["query"]))
    assert cut is not None and len(re.findall(r"\w+", cut)) < 10

    short_parent = _parent(query="nginx", surfaces=(), stat_value=1.0)
    assert op.apply(short_parent, "version_pinned_technical", "nginx") is None, (
        "expansion has no machine definition and must stay with the model"
    )


def _matched(share: float, words: float) -> pd.DataFrame:
    return pd.DataFrame([{
        "query_id": "q1", "surfaces": ("CVE-2021-44228",),
        SHARE_COLUMN: share, WORDS_COLUMN: words,
    }])


def test_headroom_drops_a_parent_the_injection_would_push_out_of_the_cell():
    """`code_symbol_named_in_prose` floors nl_share at 0.4. A 4-word parent at
    0.5 falls to 0.5*4/7 = 0.29 once a 3-token CVE lands — it would fail a band
    no instruction mentions, so it must never be picked."""
    assert InjectOperator._keeps_headroom(
        _matched(share=0.5, words=4.0), "code_symbol_named_in_prose"
    ).empty
    # the same share on a long parent barely moves: 0.5*40/43 = 0.465
    assert not InjectOperator._keeps_headroom(
        _matched(share=0.5, words=40.0), "code_symbol_named_in_prose"
    ).empty


def test_headroom_is_a_no_op_for_a_cell_with_no_share_floor():
    kept = InjectOperator._keeps_headroom(
        _matched(share=0.0, words=2.0), "symbol_pile_no_grammar"
    )
    assert len(kept) == 1, "a cell that never bands nl_share must not be filtered"


def test_identifier_spans_is_measurable_rather_than_always_failing():
    """It was a span count filed as a stat, so `verify` looked it up in
    stat_values, found nothing, and failed every time."""
    from taxonomy_generators.verify import StatTarget, Targets

    target = Targets(stats=(StatTarget(stat="identifier_spans", max_value=0.999),))
    assert verify("what is the capital of France", target).passed
    assert not verify("see CVE-2021-44228 here", target).passed


def test_the_cut_never_eats_the_span_the_row_is_generated_for(extractor):
    """`bare_acronym` bands length_words < 3, so two tokens survive — and the
    acronym is why the parent was selected. Ranking by rarity alone loses it
    to any other rare word, which is what dropped 8 of 13 rows."""
    op = StatRewrite()
    parent = _parent(
        query="why are Americans attacking Arabs in the USA today",
        surfaces=(), stat_value=9.0,
    )
    cut = op.apply(parent, "bare_acronym", str(parent["query"]))
    assert "USA" in cut, f"the acronym was cut away: {cut!r}"


class PoisonRunEngine:
    """Accepts exactly as the real engine does, but fails the test if the loop
    ever spends a completion."""

    def __init__(self) -> None:
        self._extractor = FeatureExtractor(engines=None)

    def accept(self, text, targets):
        return verify(text, targets, extractor=self._extractor)

    def run(self, *args, **kwargs):
        raise AssertionError("engine.run() was called on a deterministic cell")


def test_a_fully_deterministic_cell_never_spends_a_completion(tmp_path):
    """The Phase 1 branch: `symbol_pile_no_grammar` is served by Inject alone,
    so `produce()` must chain `apply()` locally instead of calling the model."""
    parent = {
        "dataset": "beir-nfcorpus", "query_id": "q1", "checkable": True,
        "query": "alternative medicine", "floors": [],
        "surfaces": ("cPGES", "lipoxinA4"), "bank": "code_identifier",
        "grounding_doc_id": "MED-1",
        f"{IDENT}code_identifier": 0.0,
        "natural_language_signal.natural_language_share": 0.0,
        "length.length_words": 2.0,
    }
    paths = AugmentationPaths(data_dir=tmp_path)
    sheet_path = tmp_path / "sheet.parquet"
    _sheet("symbol_pile_no_grammar").to_parquet(sheet_path, index=False)
    config = AugmentationConfig(paths=paths)
    loop = AugmentationLoop(
        pd.DataFrame(columns=["dataset", "query_id", "query", "checkable"]),
        config=config,
        engine=PoisonRunEngine(),
        operators=(PassThroughInject(config),),
        sheet_path=sheet_path,
        pool=GeneratedPool(paths),
        qrels=AugmentationQrels(paths),
        parents=OneParent(parent),
    )

    banked = loop.run("symbol_pile_no_grammar", n=1)
    assert len(banked) == 1, "the deterministic path should still bank a row"
    row = banked.iloc[0]
    assert row["hops"] == 0 and row["tokens"] == 0, "a free row must record no spend"
    assert "cPGES" in row["query"] and "lipoxinA4" in row["query"]


def test_a_mixed_call_still_falls_through_to_the_model():
    """One step without `apply()` sends the WHOLE call to the LLM — the brief is
    written per call, so half-serving it would silently drop the other half."""
    from augmentation.dispatch import Call, Stage, Step

    inject, stat = (
        next(o for o in default_operators() if o.declaration.operator == name)
        for name in ("inject", "stat_rewrite")
    )
    # short parent -> StatRewrite must EXPAND, which only the model can do
    parent = _parent(
        query="nginx", surfaces=("2.1.3",), bank="version_string", stat_value=1.0,
    )
    call = Call(
        steps=(
            Step((), Stage.CORPUS, inject),
            Step((), Stage.CONSTRAIN, stat),
        ),
        verified=(),
    )
    assert AugmentationLoop._deterministic(
        "version_pinned_technical", call, parent, "nginx"
    ) is None
