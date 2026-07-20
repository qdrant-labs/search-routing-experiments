# TODOS

Gates / next actions (2026-07-15, GLiNER2 sidestep — SPEC.md decision 13):

- [x] (done 2026-07-16 — gate FAILED on organization; outcome ratified in
      SPEC d13: adopted label-scoped {person, location, proper noun} +
      temporal fallback; org/product dropped) GLiNER2 smoke eval: 200 stratified
      queries (50 each msmarco/trec-dl/nfcorpus/miracl-en), labels
      person/org/location/product/date, hand-audited; accept at precision
      ≥ 0.8 on person/org/location, offset round-trip, lowercase parity.
      Fail → rerun same 200 on GLiNER v1 small before considering fine-tune.
- [x] (done 2026-07-16) Acronym regex bank (AMBIGUOUS, cased shape + dotted).
- [x] (done 2026-07-16/17 — regex layer + Gliner2TemporalBank backstop,
      first layered feature) Temporal-expression regex bank (closed vocabulary:
      today/yesterday/last week...); GLiNER2 date label only as fallback.
- [x] (done 2026-07-17 — extractor + live tests; diagnostic still open, see
      below) POS profile extractor (SPEC decision 14): spaCy UD-17 histogram +
      derived scalar views.
- [ ] POS domain-shift diagnostic (d14 guardrail): one-off
      closed_class_share ↔ stopword-ratio correlation over the cached
      datasets — natural to run with the first full profile.

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
- [x] (done 2026-07-17) Fan-out: logical/, corruption/, metrics/ directories + remaining marker
      banks (greetings, politeness, interjections, comparatives) on the
      markers/ pattern.
- [ ] Watch: edify ignore_case() flag scope if phrase_alternation is ever
      composed with case-sensitive parts inside one define().

## From stage-1 close-out grill (2026-07-17, SPEC decision 17)
- [ ] PMI (avg pointwise mutual information) — blocked on a design question:
      which background co-occurrence source? (wordfreq is unigram-only;
      using the query corpus itself makes it corpus-relative)
- [ ] Char-level typos — blocked on: which dictionary for OOV/edit-distance?
- [x] (resolved 2026-07-17, SPEC d18: lingua-py, LANGUAGE_SET + CODE_SWITCHING
      in SEMANTICAL) Mono vs multilingual
- [ ] Corruption degree (clean/light/heavy) — derived view over corruption
      features once more of them exist (SPEC decision 8)
- [ ] Gliner2 fine-tune path for lowercase acronyms / org — silver labels
      from banks, audit.csv becomes the held-out test set (SPEC d13)

## From multilingual grill (2026-07-17, SPEC decisions 18-19)
- [ ] lingua-py dependency (scoped to the MIRACL-14 via from_languages, not
      all 75 — memory guardrail) + LanguageSetBank / CodeSwitchingBank in a
      new semantical/ package (SemanticFeature enum already in taxonomy.py).
- [ ] `languages` parameter plumbing: resolve/extract signature,
      supported_languages ClassVar on GeneralBank, extractor skip logic +
      init warning via spacy.util.get_installed_models().
- [ ] Language-keyed spaCy caches: _pipeline(lang), _doc(text, lang); pinned
      per-language model table; loud failure with download command.
- [ ] English-gated identifier audit: flag banks with English keyword gates
      (error/HTTP, Sprint, odds of...) and script-locked shapes (\b, [A-Z])
      for CJK/Cyrillic behavior.
- [ ] UD-based marker layer: negation via Polarity=Neg morphology as the
      multilingual replacement for the English word list — prototype and
      measure vs the regex bank on English first.
- [ ] Per-language gate: rerun the smoke-eval harness on MIRACL topics per
      language before any engine ships in that language.
- [ ] LLMBank: reserved last-resort engine — deferred by design (d18);
      LLM's active lane is teacher (silver labels) + judgment features.
