"""Retrieval lanes, split by backend the way `dataset_registry` is. Every
public name stays importable from `hybrid_search_rrf_dataset.retrieval`.

Re-exported lazily: `hf`, `irds` and `wave3` each import `datasets` or
`ir_datasets` at module scope, so an eager `__init__` made
`from ...retrieval.base import RetrievalDataset` — which `qrels.QrelStore`
needs and which imports neither — pay for both. `base` is the light one.
"""

from lazy_exports import lazy_exports

_BASE = "hybrid_search_rrf_dataset.retrieval.base"
_HF = "hybrid_search_rrf_dataset.retrieval.hf"
_IRDS = "hybrid_search_rrf_dataset.retrieval.irds"
_WAVE3 = "hybrid_search_rrf_dataset.retrieval.wave3"

_EXPORTS = {
    "CORPUS_COLUMNS": _BASE,
    "QREL_COLUMNS": _BASE,
    "QUERY_COLUMNS": _BASE,
    "CorpusRecipe": _BASE,
    "MaterializedDataset": _BASE,
    "QuerySubset": _BASE,
    "QuerySupplement": _BASE,
    "RetrievalDataset": _BASE,
    "SnapshotDataset": _BASE,
    "BrightLane": _HF,
    "ClercLane": _HF,
    "CrumbLane": _HF,
    "FreshStackLane": _HF,
    "GooaqLane": _HF,
    "LimitLane": _HF,
    "QuestLane": _HF,
    "RarbLane": _HF,
    "ScirgenGeoLane": _HF,
    "WebFaqLane": _HF,
    "AntiqueLane": _IRDS,
    "BeirDataset": _IRDS,
    "DBPediaLane": _IRDS,
    "IRDatasetsMaterialized": _IRDS,
    "LotteLane": _IRDS,
    "MiraclLane": _IRDS,
    "MSMarcoDev": _IRDS,
    "NFCorpus": _IRDS,
    "OrcasLane": _IRDS,
    "TrecDL2022": _IRDS,
    "AmazonEsciLane": _WAVE3,
    "FinderLane": _WAVE3,
    "HomeDepotLane": _WAVE3,
    "TechQaLane": _WAVE3,
    "Touche2020Lane": _WAVE3,
    "TrecCast2020HistoryLane": _WAVE3,
    "WandsLane": _WAVE3,
}

__all__ = sorted(_EXPORTS)

__getattr__, __dir__ = lazy_exports(__name__, _EXPORTS, globals())
