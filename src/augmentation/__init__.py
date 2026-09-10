"""Augmentation loop over the order sheet (SPEC d42).

Demand from the sheet, supply from the tables, meaning by declaration,
truth by re-measurement. The LLM only weaves.

Re-exports are lazy. `campaign` reaches `judge` -> `parents` ->
`composition.compose` -> `dataset_registry`, so eager re-exports made
`augmentation.operators` — which needs none of that — pull ir_datasets and
`datasets`. `config`, `core` and `operators` are the light path this package's
own consumers use most.
"""

from lazy_exports import lazy_exports

_CONFIG = "augmentation.config"
_CORE = "augmentation.core"
_OPERATORS = "augmentation.operators"

_EXPORTS = {
    "AugmentationConfig": _CONFIG,
    "AugmentationPaths": _CONFIG,
    "EngineSettings": _CONFIG,
    "StatDeclarations": _CONFIG,
    "StatDirection": _CONFIG,
    "StatEntry": _CONFIG,
    "AnswerKeyPath": _CORE,
    "AugmentedCandidate": _CORE,
    "CreditGate": _CORE,
    "Declaration": _CORE,
    "Operator": _CORE,
    "SurfaceOrigin": _CORE,
    "OPERATOR_FAMILIES": _OPERATORS,
    "DecorateOperator": _OPERATORS,
    "InjectOperator": _OPERATORS,
    "OperatorSyntaxRewrite": _OPERATORS,
    "StatRewrite": _OPERATORS,
    "default_operators": _OPERATORS,
    "operator_for": _OPERATORS,
    "AugmentationCampaign": "augmentation.campaign",
    "AugmentationOutcome": "augmentation.engine",
    "Augmenter": "augmentation.engine",
    "CoherenceJudge": "augmentation.judge",
    "AugmentationLoop": "augmentation.loop",
    "GeneratedPool": "augmentation.pool",
    "AugmentationQrels": "augmentation.qrels",
    "SupplyIndex": "augmentation.supply",
}

__all__ = sorted(_EXPORTS)

__getattr__, __dir__ = lazy_exports(__name__, _EXPORTS, globals())
