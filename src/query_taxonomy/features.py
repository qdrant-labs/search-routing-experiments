import math
from collections.abc import Iterable
from functools import cache, cached_property

from pydantic import BaseModel, ConfigDict, computed_field

from query_taxonomy.banks import (
    BANKS,
    Domain,
    IdentifierBank,
    StructuralIdentifier,
)
from query_taxonomy.core import FeatureSpan


@cache
def _domain_by_type() -> dict[StructuralIdentifier, Domain]:
    return {bank.name: bank.domain for bank in (cls() for cls in BANKS)}

def normalized_idf(df: int, n_docs: int) -> float:
    """
    Smoothed idf in [0, 1]: log((N+1)/(df+1)) / log(N+1).
    df=0 (absent from corpus) maps to 1.0, df=N to 0.0.
    """
    if n_docs == 0:
        return 0.0

    return math.log((n_docs + 1) / (df + 1)) / math.log(n_docs + 1)


class DocumentIdentifier(BaseModel):
    model_config = ConfigDict(frozen=True)

    type: StructuralIdentifier
    # doc_ids -> list of identified matches
    spans: dict[str, list[FeatureSpan]]

    @computed_field
    @cached_property
    def diversity(self) -> int:
        """Corpus-wide: distinct surface forms of this type."""
        return len({
            m.text for matches in self.spans.values() for m in matches
        })

    @computed_field
    @cached_property
    def dfs(self) -> dict[str, int]:
        """Per surface form: number of docs containing it."""
        counts: dict[str, int] = {}
        for matches in self.spans.values():
            for text in {m.text for m in matches}:
                counts[text] = counts.get(text, 0) + 1
        return counts


class QueryIdentifiers(BaseModel):
    """Identifier signals of one query, joined against a corpus profile."""

    model_config = ConfigDict(frozen=True)

    query_text: str

    spans: dict[StructuralIdentifier, list[FeatureSpan]]

    @computed_field
    @property
    def tfs(self) -> dict[StructuralIdentifier, int]:
        return {k: len(v) for k, v in self.spans.items()}


class CorpusIdentifiers(BaseModel):
    model_config = ConfigDict(frozen=True)

    documents: dict[StructuralIdentifier, DocumentIdentifier]
    """Per type present in the corpus: doc-keyed spans profile."""
    queries: list[QueryIdentifiers]
    """One entry per input query, in input order."""

    def summary(self) -> str:
        """Domain-grouped identifier report. The per-domain view doubles as
        the FP smell-test: hits from an off-topic domain (finance types on
        a QA corpus) are a priori suspect."""
        tagged_total = sum(1 for query in self.queries if query.spans)
        total = len(self.queries)
        share = 100 * tagged_total / total if total else 0.0
        lines = [f"queries: {total} tagged: {tagged_total} ({share:.1f}%)"]

        by_domain: dict[Domain, list[DocumentIdentifier]] = {}
        for doc in self.documents.values():
            by_domain.setdefault(_domain_by_type()[doc.type], []).append(doc)

        def tagged(docs: list[DocumentIdentifier]) -> int:
            return len({query_id for doc in docs for query_id in doc.spans})

        pad = max(
            (len(doc.type.value) for doc in self.documents.values()), default=0
        )
        for domain, docs in sorted(
            by_domain.items(), key=lambda item: -tagged(item[1])
        ):
            lines.append("")
            lines.append(
                f"-- {domain.value} ({len(docs)} types, {tagged(docs)} tagged)"
            )
            for doc in sorted(docs, key=lambda doc: -len(doc.spans)):
                matches = sum(len(spans) for spans in doc.spans.values())
                lines.append(
                    f"{doc.type.value:<{pad}}  queries={len(doc.spans):3d} "
                    f"matches={matches:3d} diversity={doc.diversity:3d}"
                )
                top = sorted(doc.dfs.items(), key=lambda item: -item[1])[:3]
                forms = ", ".join(f"{text} ({df})" for text, df in top)
                lines.append(f"{'':<{pad}}  top: {forms}")
        return "\n".join(lines)

    def __str__(self) -> str:
        return self.summary()


class CorpusIdentifierExtractor:
    """
    Runs all banks over a corpus of queries in AmbiguityTier order
    (RIGID first, AMBIGUOUS last; BANKS order within a tier). A per-text
    registry of claimed char ranges blocks lower-priority banks from
    re-claiming any overlapping text — NUMBER never steals `1.0` out of
    a claimed `v1.0.0`. First claim wins; this replaces the earlier
    longest-match-wins resolution with tier-priority-wins.
    """

    def __init__(self, banks: Iterable[type[IdentifierBank]] = BANKS) -> None:
        # stable sort -> BANKS order breaks ties within a tier
        self._banks: list[IdentifierBank] = sorted(
            (bank() for bank in banks), key=lambda bank: bank.ambiguity
        )

    def resolve(self, text: str) -> dict[StructuralIdentifier, list[FeatureSpan]]:
        """Registry-resolved matches for one text, grouped by type."""
        registry: list[FeatureSpan] = []
        by_type: dict[StructuralIdentifier, list[FeatureSpan]] = {}
        for bank in self._banks:
            for match in bank.matches(text):
                if any(match.start < c.end and c.start < match.end for c in registry):
                    continue
                registry.append(match)
                by_type.setdefault(bank.name, []).append(match)
        return by_type

    def extract(self, queries: Iterable[str]) -> CorpusIdentifiers:
        """Single pass: fills DocumentIdentifier per type and QueryIdentifiers per query."""
        doc_spans: dict[StructuralIdentifier, dict[str, list[FeatureSpan]]] = {}
        query_models: list[QueryIdentifiers] = []
        for index, query in enumerate(queries):
            by_type = self.resolve(query)
            query_models.append(QueryIdentifiers(query_text=query, spans=by_type))
            for type_, matches in by_type.items():
                doc_spans.setdefault(type_, {})[str(index)] = matches
        documents = {
            type_: DocumentIdentifier(type=type_, spans=spans)
            for type_, spans in doc_spans.items()
        }
        return CorpusIdentifiers(documents=documents, queries=query_models)
