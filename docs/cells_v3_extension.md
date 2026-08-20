# Cell validation + taxonomy extension (v3, additive)

Two jobs: install the membership gate the adversarial review found missing
(Tier 2A "GATE 3"), and give the 26 orphaned identifier banks and the corruption
degree ladder a predicate path. `cells.yaml` was never opened for writing.

```bash
poetry run pytest -q                                    # 479 passed, 17 xfailed
poetry run ruff check src/composition src/scripts tests
poetry run python src/scripts/build_v3_catalog.py --force   # user-run; backfills 25 columns
```

## 1. The membership gate — `tests/test_cell_membership.py`

58 cells × (2 positives authored from `name` + `looks_like` alone, before any
band was read) × (2 blatant non-members). One module-scoped `mini_catalog` with
`FeatureExtractor(engines=None)`; 7s for the whole file.

**Verdict: 42/58 admit their own archetype. 16 do not, and 1 admits a blatant
non-member.** No case was reshaped to pass — every failure below is
`xfail(strict=True)` with the measurement in its `reason`, and is a quarantine
candidate. TOTAL = no prose-literal instance can pass; PARTIAL = one named form
in the prose has a path, another does not.

| Cell | Why the predicate rejects its own prose | |
|---|---|---|
| `deep_nesting_single_sentence` | `CLAUSAL_DEPS` counts the relcl/advcl/ccomp the prose tells you to stack, so `statement_count < 2` rejects every 6-deep sentence (measured 5 and 6) | TOTAL |
| `comparative_multi_entity` | "X versus Y" — the prose's first named form — yields 0 conj arcs, so `widest_list_size >= 2` is unreachable; `versus` is also not a comparative marker | TOTAL |
| `pasted_code_fragment` | `for`/`in`/`at`/`if` are closed-class: a Python snippet scores 0.3, a Java trace 0.2, against a `natural_language_share < 0.1` ceiling | TOTAL |
| `boolean_operator_query` | uppercase AND/OR (CCONJ) and NOT (PART) count as natural language: "cats AND dogs NOT birds" = 0.4 vs a `< 0.25` ceiling. Operator density inverts the predicate | TOTAL |
| `env_var_configuration` | the `env_var` bank needs a `$` sigil; the prose demands the documentation form (bare `ALL_CAPS_UNDERSCORE`), which `code_identifier` claims instead | TOTAL |
| `legal_citation_canonical` | a case-reporter citation is claimed by `http_status_code` ("410 U") + `number`; `legal_citation` only fires on the U.S.C. statute form. **This is why `p_natural = 0`** | TOTAL |
| `travel_transport_code` | `airport_airline_code` is keyword-gated ("flight LHR"); a natural travel query naming bare IATA codes goes to `stock_ticker_like`, in no branch of the cell | TOTAL |
| `wide_flat_enumeration` | prose permits "at most a thin frame"; the predicate needs `natural_language_share >= 0.1`, which a bare comma list cannot reach (0.0). An `and`-joined list passes only because `and` is a function word | PARTIAL |
| `symbol_pile_no_grammar` | `code_identifier` fires on snake_case only; the prose's "pasted stack-trace fragment" claims 0 (the C trace goes to `host_port`) | PARTIAL |
| `relative_temporal_no_dates` | the prose's own example "since the last release" emits no temporal span — `TemporalBank` is a closed list (latest / now / last week) | PARTIAL |
| `short_quantified_spec` | `value_with_unit` misses closed-up electrical units: "220v"/"60hz" claim nothing, "8mm"/"5kg"/"60 Hz" claim | PARTIAL |
| `bare_machine_token` | a bare CIDR is 5 `length_words` by itself, so the cell's 6-word budget affords one framing word, not the "few" the prose promises | PARTIAL |
| `logistics_catalog_token` | the prose's "dash-joined alphanumerics" SKU goes to `ticket_like` (not in the cell); `sku` needs the digits closed up (SKU12345) | PARTIAL |
| `geo_coordinate_postal` | the prose names "a coordinate pair", but `geo_coordinate` is an orphaned bank in no branch; the pair also costs 4 words against a 7-word ceiling | PARTIAL |
| `business_temporal_reference` | `business_temporal` claims only FY/quarter forms; the prose's "sprint number" and "calendar week" fall to `number` | PARTIAL |
| `typo_bearing_query` | `TypoBank` recall is uneven: "confgure" claims, "downlod" and "adress" do not | PARTIAL |

Over-admission (the half a one-sided gate cannot see): **`rare_term_query`
claims "coffee grinder"**. At two tokens `rare_share` is 0.5 whenever one word
is uncommon, so the cell takes the archetypal common concept.

`high_morphological_variation` **passed** — see §6.

## 2. Cells added (all in `cells_v3.yaml`; `CELLS` is still exactly 44)

Six pooled identifier cells for the orphaned banks, `length_words < 10` +
pooled `any_of`, each naming the v2 sibling it extends:

| Cell | Banks pooled | Extends |
|---|---|---|
| `network_device_identity` | mac_address, sim_subscriber_id | `bare_machine_token` (locators, no device identity) |
| `corporate_registry_code` | lei, market_code, industry_code, business_registration | `registry_structured_identifier` (instruments, no entity) |
| `statutory_instrument_citation` | neutral_citation, celex, eu_regulatory_citation, patent_number, national_id | `legal_citation_canonical` (US-shaped only) |
| `clinical_chemical_code` | clinical_coding, drug_id, hgvs_variant, chemical_id, clinical_trial_id, healthcare_provider_id | `bio_clinical_identifier` (2 of 8 medical banks) |
| `logistics_materials_code` | tracking_number, vin_container, customs_classification, material_grade, nsn, geo_coordinate | `logistics_catalog_token`; also plugs `geo_coordinate_postal`'s missing branch |
| `media_catalog_code` | iso_code, music_work_code, media_db_id | `bibliographic_catalog_identifier` |

Three corruption **degree** rungs on a new derived total (`typo_bearing_query`
and `artifact_corrupted_query` are kind-based and untouched):
`damage_free_query` (`< 1` span), `lightly_damaged_query` (exactly 1),
`heavily_damaged_query` (`>= 2`). Measured on a 2,000-row pool slice:
1,653 / 213 / 134.

`floors.py` gained `CORRUPTION_SPANS = "derived.corruption_spans"` alongside
`IDENTIFIER_SPANS` — `AxisBand` reads one column and cannot sum four. Purely
additive: no test asserted `with_derived`'s column set, and `encoder_router`
reindexes to `DOC_STATS` explicitly.

## 3. Column availability and the backfill

25 of the 26 orphaned banks have **no column in `catalog_v3.parquet`** (only
`market_code` does) — the v2 catalog predates them, and `build_v3_catalog`
sourced identifiers from the v2 join.

- `build_v3_catalog.py` now resolves `STRUCTURED_IDENTIFIERS` alongside the
  corruption/statistical groups and keeps the columns the join lacks, zero-filled
  where a bank never fires. The **whole** group is resolved, never a subset:
  claim priority within a group decides who wins a char range. Cost ~1 min over
  46K rows, regex only. **Not run** — the user runs builds.
- Until that build, the six identifier cells are **BLOCKED**:
  `select_v3_prototype._active_cells(catalog)` now drops any cell banding on a
  column the catalog in hand lacks, so `attach_strata` cannot KeyError.
  `stratum_coverage` still declares them, so they read as *declared, uncovered*
  rather than vanishing — the "known-uncovered archetypes" line Phase 2 asks for.
- `media_catalog_code`'s `iso_code` branch is **structurally dead**:
  `stock_ticker_like` sits earlier in `BANKS` at the same ambiguity tier and
  takes the region subtag of every BCP-47 form (en-US, pt-BR, zh-Hans-CN). The
  branch stays declared (the limit is extractor expressiveness, not supply); the
  `looks_like` does not send a generator at it.

## 4. Corpus-relative cells — authored, commented out, BLOCKED

`query_corpus.*` lands in `catalog_v3` only after `--force`, so four cells sit
commented in `cells_v3.yaml` with measured edges rather than hand numbers:
`thin_idf_corpus_query` (`avg_idf < 0.3075`, p25 — the `[0, 0.2, 0.4, 1.0]`
edges give a degenerate 2.7% low band), `sharp_idf_corpus_query`
(`>= 0.5004`, p75), `out_of_collection_vocabulary_query`
(`oov_share >= 1e-6`; `at_least: 0` claims 100%), and
`never_co_occurring_terms_query` — a **sentinel-membership** cell,
`{at_least: -1.0, below: -0.999}`, because `min_pmi == -1.0` exactly on 46.2% of
rows is PMIBank's "never co-occurs" marker and `below: -1.0` claims zero rows.
`mean_pmi`/`min_pmi` are NaN on 698 rows (1.5%), and NaN fails every band
silently, so these cells under-count. Diversity axis only: within-lane eta² is
0.0000–0.0118 and Cramér's V(band, lane) 0.63–0.66 — the bands are largely a
lane restatement and are authored for representation, not route power.

## 5. Multilingual — draft only, not in the live yaml

The SEMANTICAL group ships (`semantical/core.py`, `SegmentationBank`, LANGID
engine). The blocker is **two layers below the catalog**: `feature_columns` — the
catalog column convention — has no case for `LanguageSpan`, so `engines=None`
emits *no* `semantical.*` column at all, in any catalog. Every lane is English.

Why the cells matter anyway: on non-English queries today
`natural_language_share` measures **0.0** (the closed-class list is English),
`rare_share` runs 0.29–0.60, and "comment configurer le reverse proxy" is claimed
as a `corruption.typo` — so non-English rows silently land in `rare_term_query`
and `keyword_telegram_short`.

| Draft cell | Bands it would use | `predicts` | Note |
|---|---|---|---|
| `code_switched_query` | `semantical.language_count >= 2` | hybrid | mechanical switch only; the "culturally embedded" nuance is out of scope upstream |
| `non_english_monolingual_query` | `semantical.language_count < 2` + carrier language != en | dense | the control rung; needs a carrier-language column, not just a count |
| `latin_script_borrowing_query` | `semantical.language_count >= 2` + `unknown_token_rate >= 0.25` | sparse | borrowed technical terms inside a non-English frame — the case where exact match is the only handle |

Prerequisites, in order: a `LanguageSpan` case in `feature_columns`; a
non-English lane in `lanes.py`; then these bands.

## 6. Doctrine notes and contradictions to flag

- **`K_cap = 5` decouples breadth from mass.** A cell is added because the
  archetype exists, not because supply does — allocation is capped separately at
  `min(target, 5 · p_natural · T)`, a breach being declared generation-only. The
  nine new cells are therefore judged by **coverage** (does the archetype have a
  predicate path at all), never by yield.
- **The brief expected `high_morphological_variation` to fail; it passes.** Two
  blind prose-literal cases were admitted. The review's failing example
  (`natural_language_share` 0.18) is not the only reachable form.
- **The brief's Task 4 contradicts the plan's step 2.8**, which says corpus-stat
  cells are NOT authored. Resolved by authoring them commented-out with measured
  edges: the yaml carries the design, the selector carries none of it, and 2.8's
  refutation is recorded in the block comment.
- **Corruption is still computed twice, and the two disagree.** On the same
  2,000 rows, the catalog columns give clean/light/heavy 1,653/213/134 while
  `attach_strata`'s independent re-extraction gives 1,713/188/99 — ~3% of rows
  differ, because the catalog pass includes the TOKENIZER engine and the
  re-extraction does not. Plan step 2.4 (read `corruption.*` off the catalog)
  should land before any corruption floor is enforced.
- **Quarantine is still only a word.** The 17 xfails name the candidates; there
  is no quarantine set, no `_active_cells` filter for them, and no order-sheet
  suppression, so a quarantined cell would still bill for spend. Out of scope
  here (plan Phase 2, proof 2), but the gate now produces the list it needs.
