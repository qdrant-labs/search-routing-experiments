"""GLiNER2 smoke eval — the measure-then-ship gate for SPEC.md decision 13.

Modes:
    poetry run python src/smoke_eval_gliner2.py run
        Sample 50 queries per dataset, extract entities (cased + lowercase;
        the lowercase pass is skipped for queries that are already all-
        lowercase), write an audit CSV with an empty `verdict` column and
        print a summary.

    poetry run python src/smoke_eval_gliner2.py report
        After hand-filling `verdict` with TP/FP, compute per-label precision
        and a confidence-threshold sweep against the acceptance bar.
"""

import argparse
import csv
import re
import sys
from collections import Counter
from pathlib import Path

from tqdm.auto import tqdm

from dataset_registry.core import DatasetName
from dataset_registry.registry import DatasetRegistry

MODEL_ID = "fastino/gliner2-base-v1"
SAMPLE_PER_DATASET = 50
SEED = 0
# low on purpose: the audit must see the false positives so thresholds can
# be tuned upward from audited data instead of guessed
THRESHOLD = 0.3
ACCEPTANCE_BAR = 0.8
GATED_LABELS = ("person", "organization", "location")

ENTITY_SCHEMA = {
    # hand-audited: gated labels (SPEC decision 13) + genuinely non-regexable
    "person": "name of a person or people",
    "organization": "name of a company, institution, or organization",
    "location": "name of a place such as a city, country, region, or landmark",
    "product": "name of a product, device, model, or software",
    "temporal": (
        "a date, time, or temporal expression, absolute or relative, "
        "such as 1995, yesterday, q3 2024, the 90s, next week"
    ),
    "acronym": (
        "an acronym or abbreviation in any casing, including lowercase "
        "forms such as tv, dna, nasa, lgtm, n.y."
    ),
    "proper noun": (
        "a proper noun in any casing — the name of a specific person, "
        "place, organization, brand, title, or work, such as georgia, "
        "scooby-doo, windows defender, paint it black"
    ),
    "comparative": (
        "a comparative or superlative marker such as better, faster than, "
        "cheapest, most popular, worst"
    ),
    # calibration controls: ground truth is a closed word list, so these
    # labels are auto-verdicted and measure precision AND recall for free
    "pronoun": "a pronoun such as i, you, my, his, them, this, those, who",
    "greeting": "a greeting such as hi, hello, hey, good morning",
    "politeness": (
        "a politeness marker such as please, thank you, kindly, could you"
    ),
    "negation": (
        "a negation or exclusion word such as not, without, never, excluding"
    ),
    "interjection": (
        "an interjection or exclamation such as wow, ugh, oh, hmm, yay"
    ),
}

MARKER_LISTS: dict[str, frozenset[str]] = {
    "pronoun": frozenset(
        """i you he she it we they me him her us them my your his its our
        their mine yours hers ours theirs myself yourself himself herself
        itself ourselves themselves this that these those who whom whose
        which""".split()
    ),
    "greeting": frozenset(
        ["hi", "hello", "hey", "howdy", "greetings", "yo", "good morning",
         "good afternoon", "good evening"]
    ),
    "politeness": frozenset(
        ["please", "thanks", "thank you", "kindly", "could you", "would you",
         "can you", "may i", "excuse me"]
    ),
    "negation": frozenset(
        ["not", "no", "never", "without", "except", "excluding", "excluded",
         "neither", "nor", "don't", "doesn't", "isn't", "won't", "can't",
         "cannot", "vs", "versus"]
    ),
    "interjection": frozenset(
        ["wow", "ugh", "oh", "ah", "hmm", "yay", "ouch", "omg", "lol", "huh",
         "uh", "um", "wtf"]
    ),
}

_MARKER_RES: dict[str, re.Pattern[str]] = {}


def marker_spans(label: str, text: str) -> set[tuple[int, int]]:
    """Gold spans for a control label: word-boundary matches of its list."""
    if label not in _MARKER_RES:
        phrases = sorted(MARKER_LISTS[label], key=len, reverse=True)
        _MARKER_RES[label] = re.compile(
            r"\b(" + "|".join(re.escape(p) for p in phrases) + r")\b",
            re.IGNORECASE,
        )
    return {match.span() for match in _MARKER_RES[label].finditer(text)}

DATASETS = (
    DatasetName.MSMARCO_PASSAGE_DEV,
    DatasetName.TREC_DL_2022,
    DatasetName.BEIR_NFCORPUS,
    DatasetName.MIRACL_EN_DEV,
)

OUT_DIR = Path(__file__).resolve().parent / "data" / "smoke_eval"
AUDIT_CSV = OUT_DIR / "audit.csv"

FIELDS = [
    "dataset",
    "query_id",
    "variant",
    "label",
    "span",
    "start",
    "end",
    "confidence",
    "query_text",
    "verdict",
]


def run() -> None:
    from gliner2 import GLiNER2

    extractor = GLiNER2.from_pretrained(MODEL_ID)
    registry = DatasetRegistry()

    rows: list[dict] = []
    roundtrip_failures: list[dict] = []
    parity_cased = 0
    parity_matched = 0
    lower_only = 0
    lower_skipped = 0
    control_gold: Counter[str] = Counter()
    control_recalled: Counter[str] = Counter()

    for name in DATASETS:
        queries = registry.sample(name, SAMPLE_PER_DATASET, seed=SEED)
        for query in tqdm(queries, desc=name.value, unit="query"):
            hits_by_variant: dict[str, set[tuple[str, str]]] = {}
            variants = [("cased", query.text)]
            if query.text != query.text.lower():
                variants.append(("lower", query.text.lower()))
            else:
                # identical input would just duplicate every row (MS MARCO
                # queries ship pre-lowercased) — parity is trivially full
                lower_skipped += 1
            for variant, text in variants:
                # one text per call: fixed batch composition keeps
                # near-threshold scores reproducible run to run
                result = extractor.extract_entities(
                    text,
                    ENTITY_SCHEMA,
                    threshold=THRESHOLD,
                    include_spans=True,
                    include_confidence=True,
                )
                keys: set[tuple[str, str]] = set()
                for label, spans in result.get("entities", {}).items():
                    for span in spans:
                        row = {
                            "dataset": name.value,
                            "query_id": query.query_id,
                            "variant": variant,
                            "label": label,
                            "span": span["text"],
                            "start": span["start"],
                            "end": span["end"],
                            "confidence": round(span["confidence"], 4),
                            "query_text": text,
                            "verdict": "",
                        }
                        if label in MARKER_LISTS:
                            row["verdict"] = (
                                "TP"
                                if span["text"].lower() in MARKER_LISTS[label]
                                else "FP"
                            )
                        rows.append(row)
                        keys.add((label, span["text"].lower()))
                        if text[span["start"] : span["end"]] != span["text"]:
                            roundtrip_failures.append(row)
                hits_by_variant[variant] = keys
                if variant == "cased":
                    for control_label in MARKER_LISTS:
                        gold = marker_spans(control_label, text)
                        found = {
                            (span["start"], span["end"])
                            for span in result.get("entities", {}).get(
                                control_label, []
                            )
                        }
                        control_gold[control_label] += len(gold)
                        control_recalled[control_label] += len(gold & found)
            cased_keys = hits_by_variant["cased"]
            lower_keys = hits_by_variant.get("lower", cased_keys)
            parity_cased += len(cased_keys)
            parity_matched += len(cased_keys & lower_keys)
            lower_only += len(lower_keys - cased_keys)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with AUDIT_CSV.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\naudit file -> {AUDIT_CSV} ({len(rows)} hits)")
    print("\nhits per dataset/label (cased variant only):")
    counts = Counter(
        (row["dataset"], row["label"])
        for row in rows
        if row["variant"] == "cased"
    )
    for (dataset, label), count in sorted(counts.items()):
        print(f"  {dataset:<24} {label:<14} {count:3d}")

    print(f"\noffset round-trip failures: {len(roundtrip_failures)}")
    for failure in roundtrip_failures[:5]:
        print(f"  {failure['query_id']} {failure['label']!r} {failure['span']!r}")

    if parity_cased:
        parity = 100 * parity_matched / parity_cased
        print(
            f"lowercase parity: {parity:.1f}% of cased hits also found in "
            f"lowercased text ({parity_matched}/{parity_cased}); "
            f"{lower_only} lower-only hits; {lower_skipped} queries already "
            f"lowercase (second pass skipped, counted as full parity)"
        )
    print("\ncontrol labels (auto-verdicted vs closed word lists):")
    for control_label in MARKER_LISTS:
        hits = [
            row for row in rows
            if row["label"] == control_label and row["variant"] == "cased"
        ]
        tps = sum(row["verdict"] == "TP" for row in hits)
        precision = f"{100 * tps / len(hits):5.1f}% ({tps}/{len(hits)})" if hits else "   no hits"
        gold = control_gold[control_label]
        recalled = control_recalled[control_label]
        recall = f"{100 * recalled / gold:5.1f}% ({recalled}/{gold})" if gold else "no gold in sample"
        print(f"  {control_label:<13} precision {precision:<16} recall {recall}")

    hand_labels = sorted(set(ENTITY_SCHEMA) - set(MARKER_LISTS))
    print(
        f"\nnext: fill the `verdict` column with TP/FP for cased rows of "
        f"{', '.join(hand_labels)} (control rows are pre-filled), then run "
        "the `report` mode"
    )


def report() -> None:
    with AUDIT_CSV.open(newline="") as handle:
        rows = [row for row in csv.DictReader(handle) if row["variant"] == "cased"]
    unaudited = [row for row in rows if row["verdict"].strip().upper() not in ("TP", "FP")]
    if unaudited:
        sys.exit(f"{len(unaudited)} rows still lack a TP/FP verdict — finish the audit first")

    print(f"{'label':<14} {'n':>4} {'precision':>9}   best threshold (precision >= {ACCEPTANCE_BAR})")
    for label in ENTITY_SCHEMA:
        audited = [row for row in rows if row["label"] == label]
        if not audited:
            print(f"{label:<14} {0:>4} {'--':>9}")
            continue
        tps = sum(row["verdict"].strip().upper() == "TP" for row in audited)
        precision = tps / len(audited)
        # sweep: smallest confidence cutoff whose surviving rows clear the bar
        best = None
        for cutoff in sorted({float(row["confidence"]) for row in audited}):
            kept = [row for row in audited if float(row["confidence"]) >= cutoff]
            kept_tps = sum(row["verdict"].strip().upper() == "TP" for row in kept)
            if kept and kept_tps / len(kept) >= ACCEPTANCE_BAR:
                best = (cutoff, kept_tps, len(kept))
                break
        gate = " <- GATED" if label in GATED_LABELS else ""
        if best:
            cutoff, kept_tps, kept_n = best
            print(
                f"{label:<14} {len(audited):>4} {precision:>9.2f}   "
                f">= {cutoff:.2f} keeps {kept_tps}/{kept_n}{gate}"
            )
        else:
            print(f"{label:<14} {len(audited):>4} {precision:>9.2f}   never clears the bar{gate}")

    casing_splits = {
        # bank/casing-detectable slice vs the slice only a model can reach
        "acronym": lambda span: span == span.upper(),
        "proper noun": lambda span: span != span.lower(),
    }
    for split_label, detectable in casing_splits.items():
        hits = [row for row in rows if row["label"] == split_label]
        if not hits:
            continue
        print(f"\n{split_label} precision by casing:")
        for slice_name, subset in (
            ("casing-detectable", [r for r in hits if detectable(r["span"])]),
            ("lowercase (model territory)", [r for r in hits if not detectable(r["span"])]),
        ):
            if subset:
                tps = sum(row["verdict"].strip().upper() == "TP" for row in subset)
                print(f"  {slice_name:<28} {tps}/{len(subset)} = {tps / len(subset):.2f}")
            else:
                print(f"  {slice_name:<28} no hits")

    gated = [row for row in rows if row["label"] in GATED_LABELS]
    gated_tps = sum(row["verdict"].strip().upper() == "TP" for row in gated)
    overall = gated_tps / len(gated) if gated else 0.0
    verdict = "PASS" if overall >= ACCEPTANCE_BAR else "FAIL"
    print(
        f"\ngate ({'/'.join(GATED_LABELS)}): precision {overall:.2f} "
        f"at threshold {THRESHOLD} -> {verdict}"
    )
    if verdict == "FAIL":
        print("per SPEC decision 13: rerun the same 200 queries through GLiNER v1 small before conceding to fine-tuning")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["run", "report"], nargs="?", default="run")
    args = parser.parse_args()
    run() if args.mode == "run" else report()


if __name__ == "__main__":
    main()
