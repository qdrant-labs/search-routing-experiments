"""Augmentation loop over the order sheet (SPEC d42).

Demand from the sheet, supply from the tables, meaning by declaration,
truth by re-measurement. The LLM only weaves.
"""

from augmentation.core import (
    AnswerKeyPath,
    AugmentedCandidate,
    CreditGate,
    Declaration,
    Grounding,
    Operator,
)
from augmentation.engine import AugmentationOutcome, Augmenter
from augmentation.loop import AugmentationLoop
from augmentation.operators import OPERATORS, DecorateOperator, operator_for
from augmentation.pool import GeneratedPool

__all__ = [
    "OPERATORS",
    "AnswerKeyPath",
    "AugmentationLoop",
    "AugmentationOutcome",
    "AugmentedCandidate",
    "Augmenter",
    "CreditGate",
    "Declaration",
    "DecorateOperator",
    "GeneratedPool",
    "Grounding",
    "Operator",
    "operator_for",
]
