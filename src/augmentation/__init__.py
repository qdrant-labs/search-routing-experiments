"""Augmentation loop over the order sheet (SPEC d42).

Demand from the sheet, supply from the tables, meaning by declaration,
truth by re-measurement. The LLM only weaves.
"""

from augmentation.config import (
    AugmentationConfig,
    AugmentationPaths,
    EngineSettings,
    StatDeclarations,
    StatDirection,
    StatEntry,
)
from augmentation.core import (
    AnswerKeyPath,
    AugmentedCandidate,
    CreditGate,
    Declaration,
    SurfaceOrigin,
    Operator,
)
from augmentation.campaign import AugmentationCampaign
from augmentation.engine import AugmentationOutcome, Augmenter
from augmentation.loop import AugmentationLoop
from augmentation.operators import (
    OPERATOR_FAMILIES,
    DecorateOperator,
    InjectOperator,
    OperatorSyntaxRewrite,
    StatRewrite,
    default_operators,
    operator_for,
)
from augmentation.pool import GeneratedPool
from augmentation.qrels import AugmentationQrels
from augmentation.supply import SupplyIndex

__all__ = [
    "OPERATOR_FAMILIES",
    "AnswerKeyPath",
    "AugmentationCampaign",
    "AugmentationConfig",
    "AugmentationLoop",
    "AugmentationOutcome",
    "AugmentationPaths",
    "AugmentationQrels",
    "AugmentedCandidate",
    "Augmenter",
    "CreditGate",
    "Declaration",
    "DecorateOperator",
    "EngineSettings",
    "GeneratedPool",
    "SurfaceOrigin",
    "InjectOperator",
    "Operator",
    "OperatorSyntaxRewrite",
    "StatDeclarations",
    "StatDirection",
    "StatEntry",
    "StatRewrite",
    "SupplyIndex",
    "default_operators",
    "operator_for",
]
