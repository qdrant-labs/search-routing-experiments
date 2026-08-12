"""Encoder router: a query-only route model whose privileged branches are
supervised by dataset-native artifacts (cell predicates, corpus and gold-doc
profiles, outcome rates) and feed their estimates forward at serve time.
Model and evaluation import torch lazily — table and target artifacts work
without it."""

from encoder_router.table import NgramSvd, QueryEmbeddings, TrainingTable
from encoder_router.targets import CorpusProfile, GoldDocProfile

__all__ = [
    "CorpusProfile",
    "GoldDocProfile",
    "NgramSvd",
    "QueryEmbeddings",
    "TrainingTable",
]
