"""What each lane's retrieval task actually is, told to the judge.

One universal definition of "relevant" mis-reads lanes whose task is not
question-answering: measured per-lane recall against human gold ranges from
0.980 (rarb-math) to 0.055 (crumb-legal-qa), and reading the failures shows the
gap is task framing, not judgment. Provenance facts here come from the
`dataset_registry` class docstrings and from the validation predictions; each
card cites the evidence that produced it.
"""

from __future__ import annotations

from typing import Final, NamedTuple


class LaneContext(NamedTuple):
    """One lane's judging card. `benchmark` is for humans reading the registry —
    it is deliberately NOT sent to the model, because it never changes a verdict
    and would cost tokens on every call."""

    task: str
    """What the dataset is for — the retrieval job being done."""
    queries: str
    """What a query looks like, including anything surprising about its shape."""
    gold: str
    """How the gold was built, and what that implies about what counts."""
    judging: str
    """The operative instruction: what makes a document relevant HERE."""
    benchmark: str = ""
    """What the dataset is normally used to quantify. Human context only."""

    def render(self) -> str:
        return (
            f"About this collection:\n"
            f"- Task: {self.task}\n"
            f"- Queries: {self.queries}\n"
            f"- Gold documents: {self.gold}\n"
            f"- Judge accordingly: {self.judging}"
        )


LANE_CONTEXT: Final[dict[str, LaneContext]] = {
    # Evidence: 56 of 167 false negatives are disjunctive queries the judge
    # rejected for satisfying only one alternative — "It is an Eco book, but
    # lacks the other requested attributes". Registry: "natural queries with
    # implicit set operations".
    "quest": LaneContext(
        task="entity retrieval over encyclopedia articles: a query names categories "
             "and any entity satisfying the whole expression qualifies",
        queries="a set expression. 'or' and commas list ALTERNATIVES (any one is "
                "enough); 'and' lists REQUIRED categories (all must hold); a "
                "NEGATION names a category the entity must NOT be in — 'but not "
                "about capitalism', 'not found in Southeast Asia', 'that aren't "
                "palearctic'. Expressions mix all three.",
        gold="every entity satisfying the expression; a query commonly has many",
        judging="for ASKED name the one alternative the document could satisfy, "
                "or ALL the categories an 'and' requires. A named exclusion goes "
                "in MISSING only if the document's own words put the entity IN "
                "the excluded category — an article simply not mentioning it is "
                "NOT a gap, because articles never state what a thing is not. "
                "Infobox lines and the category list at the end of an article are "
                "the document's own words, usable as EVIDENCE; a bare title with "
                "neither is still boilerplate.",
        benchmark="retrieval over implicit set operations (ACL 2023)",
    ),
    # Evidence: queries end mid-citation ("...United States v. Bozza,"), so no
    # question is stated; judge replies "the document discusses Levitt, not
    # Lancaster". Registry: "gold documents are the cited cases' passages".
    "clerc": LaneContext(
        task="legal citation retrieval: recover the authority a passage is about "
             "to cite",
        queries="a passage from a court opinion or brief, TRUNCATED at the point "
                "where the citation would appear. It asks no question and is "
                "often cut mid-sentence — that missing citation IS the query.",
        gold="the passage of the case the source text cites at that point",
        judging="relevant means this is the authority the passage is reaching "
                "for — same doctrine, holding, or proposition being relied on. "
                "Do not require the document to answer a question; none is asked.",
        benchmark="long-context legal retrieval (arXiv 2406.17186)",
    ),
    # Evidence: gold sections carry the figures, judge demands the conclusion —
    # "reports revenue amounts but not the requested mix or dependency analysis",
    # "provides figures, not the requested stability interpretation".
    "finder": LaneContext(
        task="financial-filing retrieval: find the section of a report holding "
             "the data an analyst needs",
        queries="terse analyst shorthand, heavily abbreviated — 'LYB rev mix evol "
                "trade vs related parties 3yr div dependency', 'Emp. allocation "
                "in CE region; op. div. exp.'",
        gold="the filing section containing the required figures",
        judging="a document that SUPPLIES the requested figures is relevant. The "
                "analysis, trend, or interpretation is the analyst's job, not the "
                "filing's — do not require the document to draw the conclusion.",
        benchmark="retrieval over financial reports",
    ),
    # Evidence: queries are Definition-/Interpretation-suffixed and were written
    # FROM the gold record; judge replies "document only mentions an empirical
    # retrieval algorithm" for a query asking what that term means.
    "scirgen-geo-en": LaneContext(
        task="scientific dataset retrieval over geoscience data records",
        queries="a question GENERATED FROM the gold record — definitions, "
                "interpretations, and challenges phrased about its content",
        gold="the source record the question was written from; it was never "
             "authored as an answer, so it may not phrase one explicitly",
        judging="relevant means this is the record the question was derived from "
                "and whose content the question is about. Substance the question "
                "asks about being present counts, even if unstated as an answer.",
        benchmark="scientific dataset discovery",
    ),
    # Evidence: recall 0.980 already — the card records what works rather than
    # changing it. The 4 failures reject solutions with wrong arithmetic.
    "rarb-math": LaneContext(
        task="math problem to worked-solution retrieval",
        queries="a self-contained problem statement",
        gold="a worked solution to that problem",
        judging="ASKED is only 'does this document work THIS problem's stated "
                "quantities and question' — never 'what is the correct answer'. Do "
                "not check the arithmetic and do not compute the answer yourself. "
                "Never write 'wrong answer', 'incorrect', or 'arithmetic error' on "
                "the MISSING line: that is a correctness judgment, not a missing "
                "element. A solution to a DIFFERENT problem is still not relevant.",
        benchmark="reasoning-augmented retrieval",
    ),
}
"""Cards for the lanes the judge is applied to. A lane with no card judges under
the universal instruction alone — the safe default, since a wrong card is worse
than none.

DEFERRED BY RED-TEAM REVIEW (2026-09-05) — three card rewrites were proposed and
NOT taken, because each raises recall by making the judge more credulous, and
these lanes carry no negative labels, so the resulting false positives would be
invisible and permanent:

- `clerc`: redefining EVIDENCE as "states the same proposition" turns the judge
  into a topic-matcher exactly where legal boilerplate (standard of review,
  burden of proof) is restated verbatim across hundreds of uncited cases.
- `scirgen-geo-en`: accepting "the record mentions the substance" erodes the one
  signal separating the source record from a near-duplicate record in the same
  field — which IS this lane's whole retrieval difficulty.
- `finder`: accepting "a label plus a nearby number" invites stitching figures
  from unrelated tables into one answer.

Each needs a human-audited sample of the verdicts it would flip from no to yes
before it can be trusted. See SPEC d70.

DELIBERATELY ABSENT: `crumb-legal-qa`. Its gold pairs queries with statutes on
different subjects (an attorney-fees-for-eviction query against a
violations-against-elderly-persons statute), so recall there is 0.055. No card
can be written without rationalising labels that look simply wrong; the gold
needs auditing before this lane is judged or used as a referee."""


def context_for(dataset: str) -> str:
    """The lane's judging card as prompt text, empty when the lane has none."""
    card = LANE_CONTEXT.get(dataset)
    return card.render() if card else ""
