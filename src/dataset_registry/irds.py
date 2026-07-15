from abc import ABC
from collections.abc import Iterator
from typing import ClassVar

import ir_datasets

from dataset_registry.core import (
    Availability,
    DatasetCard,
    DatasetName,
    Grounding,
    Query,
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
    """Judged slice of the MS MARCO passage dev split (~55.5K Bing queries)."""

    irds_id = "msmarco-passage/dev/judged"

    @property
    def card(self) -> DatasetCard:
        return DatasetCard(
            name=DatasetName.MSMARCO_PASSAGE_DEV,
            source=SourceKind.IR_DATASETS,
            grounding=Grounding.QQ,
            llm_target=False,
            scope=Scope.GENERAL,
            non_trivial=False,
            multilingual=False,
            multimodal=False,
            availability=Availability.OPEN,
            homepage="https://ir-datasets.com/msmarco-passage.html",
        )


class TrecDL2022(IRDatasetsBacked):
    """TREC Deep Learning 2022 judged queries over MS MARCO v2 passages."""

    irds_id = "msmarco-passage-v2/trec-dl-2022/judged"

    @property
    def card(self) -> DatasetCard:
        return DatasetCard(
            name=DatasetName.TREC_DL_2022,
            source=SourceKind.IR_DATASETS,
            grounding=Grounding.QQ,
            llm_target=False,
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
    def card(self) -> DatasetCard:
        return DatasetCard(
            name=DatasetName.BEIR_NFCORPUS,
            source=SourceKind.IR_DATASETS,
            grounding=Grounding.QQ,
            llm_target=False,
            scope=Scope.SPECIFIC,
            non_trivial=False,
            multilingual=False,
            multimodal=False,
            availability=Availability.OPEN,
            homepage="https://ir-datasets.com/beir.html#beir/nfcorpus",
        )
