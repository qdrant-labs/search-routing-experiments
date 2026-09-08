"""Encoder router: a query-only route model whose privileged branches are
supervised by dataset-native artifacts (cell predicates, corpus and gold-doc
profiles, outcome rates) and feed their estimates forward at serve time.
Model and evaluation import torch lazily — table and target artifacts work
without it.

Re-exports nothing on purpose: importing `encoder_router.table` to serve a
saved arm must not pay for the training stack."""
