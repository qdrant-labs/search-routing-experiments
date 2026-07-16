# TODOS

Gates / next actions (2026-07-15, GLiNER2 sidestep — SPEC.md decision 13):

- [ ] GLiNER2 smoke eval (the measure-then-ship gate): 200 stratified
      queries (50 each msmarco/trec-dl/nfcorpus/miracl-en), labels
      person/org/location/product/date, hand-audited; accept at precision
      ≥ 0.8 on person/org/location, offset round-trip, lowercase parity.
      Fail → rerun same 200 on GLiNER v1 small before considering fine-tune.
- [ ] Acronym regex bank (moved from MODEL tier; AMBIGUOUS tier, shape-based).
- [ ] Temporal-expression regex bank (patterns + closed vocabulary:
      today/yesterday/last week...); GLiNER2 date label only as fallback.
- [ ] POS profile extractor (SPEC decision 14): spaCy UD-17 histogram +
      derived scalar views, fixed-case tests, correlation diagnostic vs
      stopword ratio over cached datasets.

Deferred by grill-me session 2026-07-15 (see SPEC.md):

- [ ] Recipe values: total size, per-feature quotas, within-feature strata
      quotas, minimum natural share, harvest-target N.
- [ ] MODEL-tier features: pick the model + testing story for
      non-deterministic extractors (arch-validator session).
- [ ] Corpus-relative features: explicitly decide whether to lift the
      registry's queries-only restriction.
- [ ] Strategy labeling stage: empirical dense/sparse/hybrid labels via NDCG
      in src/hybrid_search_rrf_dataset.
- [ ] ILP escalation path if greedy quota-fill conflicts (solver,
      formulation).
- [ ] Register next datasets: ORCAS (recommended_sample cap), BRIGHT, CLERC,
      more BEIR subsets.
- [ ] MCP wrapper around verify() for interactive generation.
- [ ] Demo (b) infra: corpus indexing + local Qdrant for the
      strategy-disagreement measurement.

## From feature-review of extractor-model generalization, 2026-07-16
- [x] (done 2026-07-16) Extend bank-case enforcement to all groups: parametrize the
      test_every_bank_has_cases invariant over BANKS + MARKER_BANKS (and
      future group tuples) — deferred: tests scoped out of code-implementer.
- [x] (done 2026-07-16, SPEC decision 15) features.py multi-group generalization encodes within-group claim
      resolution structurally (group-partitioned registries); today it is
      type-hint-only on CorpusIdentifierExtractor.
- [ ] Fan-out: logical/, corruption/, metrics/ directories + remaining marker
      banks (greetings, politeness, interjections, comparatives) on the
      markers/ pattern.
- [ ] Watch: edify ignore_case() flag scope if phrase_alternation is ever
      composed with case-sensitive parts inside one define().
