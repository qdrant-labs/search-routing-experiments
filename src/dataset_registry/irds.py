from abc import ABC
from collections.abc import Iterator
from typing import ClassVar, override

import ir_datasets

from dataset_registry.core import (
    Availability,
    DatasetCard,
    DatasetName,
    Grounding,
    Query,
    QueryProvenance,
    RegistryDataset,
    Scope,
    SourceKind,
)


class IRDatasetsBacked(RegistryDataset, ABC):
    """Datasets served by the ir_datasets catalog: subclasses set `irds_id`."""

    irds_id: ClassVar[str]

    def _fetch_queries(self) -> Iterator[Query]:
        for query in ir_datasets.load(self.irds_id).queries_iter():
            yield Query(str(query.query_id), query.text)

    def _fetch_total(self) -> int | None:
        # served from ir_datasets' shipped metadata, no download triggered
        return ir_datasets.load(self.irds_id).queries_count()

    def describe_source(self) -> str:
        return f"ir_datasets:{self.irds_id}"


class MSMarcoPassageDev(IRDatasetsBacked):
    """Full MS MARCO passage dev split (~101K Bing queries). Deliberately
    NOT the /judged slice: mining wants the full source distribution,
    unjudged tail included (SPEC d23, d30d)."""

    irds_id = "msmarco-passage/dev"

    @property
    @override
    def card(self) -> DatasetCard:
        return DatasetCard(
            name=DatasetName.MSMARCO_PASSAGE_DEV,
            source=SourceKind.IR_DATASETS,
            grounding=Grounding.QQ,
            llm_target=False,
            query_provenance=QueryProvenance.HUMAN,
            scope=Scope.GENERAL,
            non_trivial=False,
            multilingual=False,
            multimodal=False,
            availability=Availability.OPEN,
            homepage="https://ir-datasets.com/msmarco-passage.html",
        )


class TrecDL2022(IRDatasetsBacked):
    """TREC Deep Learning 2022 test queries over MS MARCO v2 passages —
    all 500, not the 76-query /judged slice (SPEC d23: the old snapshot
    coupling cost 424/500 queries; the registry must not repeat it)."""

    irds_id = "msmarco-passage-v2/trec-dl-2022"

    @property
    @override
    def card(self) -> DatasetCard:
        return DatasetCard(
            name=DatasetName.TREC_DL_2022,
            source=SourceKind.IR_DATASETS,
            grounding=Grounding.QQ,
            llm_target=False,
            query_provenance=QueryProvenance.HUMAN,
            scope=Scope.GENERAL,
            non_trivial=False,
            multilingual=False,
            multimodal=False,
            availability=Availability.OPEN,
            homepage="https://ir-datasets.com/msmarco-passage-v2.html",
        )


class BeirNFCorpus(IRDatasetsBacked):
    """BEIR NFCorpus test split: NutritionFacts queries over PubMed abstracts."""

    irds_id = "beir/nfcorpus/test"

    @property
    @override
    def card(self) -> DatasetCard:
        return DatasetCard(
            name=DatasetName.BEIR_NFCORPUS,
            source=SourceKind.IR_DATASETS,
            grounding=Grounding.QQ,
            llm_target=False,
            query_provenance=QueryProvenance.HUMAN,
            scope=Scope.SPECIFIC,
            non_trivial=False,
            multilingual=False,
            multimodal=False,
            availability=Availability.OPEN,
            homepage="https://ir-datasets.com/beir.html#beir/nfcorpus",
        )


class Orcas(IRDatasetsBacked):
    """ORCAS click log: 10.4M unique real Bing queries (18.8M clicks) over
    MS MARCO documents — the one real query log that is openly
    downloadable. Always sampled, never profiled whole."""

    irds_id = "msmarco-document/orcas"

    @property
    @override
    def card(self) -> DatasetCard:
        return DatasetCard(
            name=DatasetName.ORCAS,
            source=SourceKind.IR_DATASETS,
            grounding=Grounding.QC,
            llm_target=False,
            query_provenance=QueryProvenance.HUMAN,
            scope=Scope.GENERAL,
            non_trivial=False,
            multilingual=False,
            multimodal=False,
            availability=Availability.OPEN,
            homepage="https://ir-datasets.com/msmarco-document.html#msmarco-document/orcas",
            recommended_sample=100_000,
        )


class DBPediaEntity(IRDatasetsBacked):
    """BEIR DBPedia-entity test: 400 bare-entity keyword queries — the
    sparse-leaning register the natural-question datasets lack."""

    irds_id = "beir/dbpedia-entity/test"

    @property
    @override
    def card(self) -> DatasetCard:
        return DatasetCard(
            name=DatasetName.DBPEDIA_ENTITY,
            source=SourceKind.IR_DATASETS,
            grounding=Grounding.QQ,
            llm_target=False,
            query_provenance=QueryProvenance.HUMAN,
            scope=Scope.GENERAL,
            non_trivial=False,
            multilingual=False,
            multimodal=False,
            availability=Availability.OPEN,
            homepage="https://ir-datasets.com/beir.html#beir/dbpedia-entity",
        )
