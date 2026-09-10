"""Registered query sources and the registry over them.

Every name below is re-exported lazily. All of `hf`, `irds`, `url`, `wave2` and
`wave3` import `datasets` or `ir_datasets` at module scope, so an eager
`__init__` made `from dataset_registry.core import RegistryDataset` — which
needs neither — pay for both. `core` is the only light submodule here.
"""

from lazy_exports import lazy_exports

_CORE = "dataset_registry.core"
_HF = "dataset_registry.hf"
_IRDS = "dataset_registry.irds"
_REGISTRY = "dataset_registry.registry"
_URL = "dataset_registry.url"
_WAVE3 = "dataset_registry.wave3"

_EXPORTS = {
    "Availability": _CORE,
    "DatasetCard": _CORE,
    "DatasetName": _CORE,
    "Grounding": _CORE,
    "Query": _CORE,
    "QueryProvenance": _CORE,
    "RegistryDataset": _CORE,
    "Scope": _CORE,
    "SourceKind": _CORE,
    "BrightSplit": _HF,
    "CrumbTask": _HF,
    "MiraclDev": _HF,
    "Quest": _HF,
    "RarbPool": _HF,
    "BeirNFCorpus": _IRDS,
    "DBPediaEntity": _IRDS,
    "IRDatasetsBacked": _IRDS,
    "MSMarcoPassageDev": _IRDS,
    "Orcas": _IRDS,
    "TrecDL2022": _IRDS,
    "DATASETS": _REGISTRY,
    "DatasetRegistry": _REGISTRY,
    "Limit": _URL,
    "AmazonEsci": _WAVE3,
    "BeirTouche2020": _WAVE3,
    "Finder": _WAVE3,
    "TechQa": _WAVE3,
    "TrecCast2020History": _WAVE3,
    "Wands": _WAVE3,
}

__all__ = sorted(_EXPORTS)

__getattr__, __dir__ = lazy_exports(__name__, _EXPORTS, globals())
