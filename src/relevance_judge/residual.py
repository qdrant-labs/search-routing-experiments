"""The judge queue: which rows measurement cannot reach, and which (query, doc)
pairs to judge for each. The population is an input — arch5k's draw carries the
l2 stack, the v2-100K labels carry the v2 stack — so the queue reads a score
triple, never a file.
"""

from __future__ import annotations

import json

import pandas as pd

from relevance_judge.config import RelevanceJudgeConfig
from relevance_judge.sources import ROUTES, Sources

TOL = 1e-9
POPULATION_COLUMNS = ("dataset", "query_id", "query", "triple")


def regime(triple: dict[str, float]) -> str:
    """all_zero / all_tied / low_margin / decisive_strong — the arch5k §8
    classifier over a route-score triple (copied, 4 lines, not imported)."""
    values = sorted(triple.values(), reverse=True)
    if values[0] <= TOL:
        return "all_zero"
    if values[0] - values[-1] <= TOL:
        return "all_tied"
    return "low_margin" if values[0] - values[1] < 0.1 else "decisive_strong"


def above_gold(orders: dict[str, list[str]], gold: set[str]) -> set[str]:
    """Docs ranked above the first gold doc in any route — the ones that move
    HitRate@1. A strict subset of `tail_docs`, which also moves NDCG. Gold
    itself is excluded (already judged). Takes orders, so live rankings and
    persisted ones share one definition."""
    above: set[str] = set()
    for order in orders.values():
        cut = next((i for i, doc in enumerate(order) if doc in gold), len(order))
        above.update(order[:cut])
    return above - gold


def tail_docs(orders: dict[str, list[str]], gold: set[str]) -> dict[str, int]:
    """`{doc_id: rank_spread}` for every non-gold doc some route retrieved,
    dropping docs at the identical rank in all routes: those add the same DCG
    and IDCG to each route, so judging them leaves any tie exactly as it was.
    Spread (widest route disagreement) orders the work list."""
    absent = max((len(order) for order in orders.values()), default=0)
    positions = [{doc: i for i, doc in enumerate(order)} for order in orders.values()]
    union = {doc for order in orders.values() for doc in order} - gold
    spreads = {}
    for doc in union:
        seq = [pos.get(doc, absent) for pos in positions]
        if len(set(seq)) > 1:
            spreads[doc] = max(seq) - min(seq)
    return spreads


class Arch5kDraw:
    """The arch5k 5,045-row draw, scored on the l2 (gemini) stack — the only
    artifact carrying l2, whose ranked lists were never persisted."""

    def __init__(self, config: RelevanceJudgeConfig) -> None:
        self.config = config

    def _rows(self) -> list[dict]:
        return json.loads(self.config.draw.read_text())

    def triples(self) -> pd.DataFrame:
        rows = [
            {"dataset": r["dataset"], "query_id": str(r["query_id"]),
             "query": r["query"], "triple": r["l2"]}
            for r in self._rows()
        ]
        return pd.DataFrame(rows, columns=list(POPULATION_COLUMNS))

    def absent(self) -> pd.DataFrame:
        """Rows whose gold never surfaced within 500 in any route."""
        qtext = {(r["dataset"], str(r["query_id"])): r["query"] for r in self._rows()}
        depth = pd.read_parquet(self.config.depth_probe).astype({"query_id": str})
        gone = depth[depth[[f"rank_{route}" for route in ROUTES]].isna().all(axis=1)]
        return pd.DataFrame([
            {"dataset": a.dataset, "query_id": str(a.query_id),
             "query": qtext.get((a.dataset, str(a.query_id)), "")}
            for a in gone.itertuples(index=False)
        ], columns=["dataset", "query_id", "query"])


class V2Labels:
    """The v2-100K labelled population, scored on the v2 stack — self-consistent
    with the persisted v2 rankings, and 18x the rows. No depth probe exists for
    it, so `absent()` is empty rather than guessed."""

    def __init__(self, config: RelevanceJudgeConfig) -> None:
        self.config = config

    def triples(self) -> pd.DataFrame:
        columns = ["dataset", "query_id", "query"] + [f"score_{r}" for r in ROUTES]
        frame = (
            pd.read_parquet(self.config.labels, columns=columns)
            .astype({"query_id": str})
            .drop_duplicates(["dataset", "query_id"])
        )
        frame["triple"] = [
            dict(zip(ROUTES, scores, strict=True))
            for scores in frame[[f"score_{r}" for r in ROUTES]].to_numpy()
        ]
        return frame[list(POPULATION_COLUMNS)].reset_index(drop=True)

    def absent(self) -> pd.DataFrame:
        return pd.DataFrame(columns=["dataset", "query_id", "query"])


class JudgeQueue:
    """Residual population + the pairs that could break each row's tie."""

    def __init__(
        self,
        config: RelevanceJudgeConfig | None = None,
        sources: Sources | None = None,
        population: Arch5kDraw | V2Labels | None = None,
    ) -> None:
        self.config = config or RelevanceJudgeConfig()
        self.sources = sources or Sources(self.config)
        self.population = population or Arch5kDraw(self.config)

    def residual(self) -> pd.DataFrame:
        """Every row measurement left unlabelled: all_tied ∪ absent@500.
        Columns dataset, query_id, query, regime, tied_value, sub1."""
        triples = self.population.triples()
        tied = triples[
            triples["triple"].map(lambda t: regime(t) == "all_tied")
        ].assign(
            regime="all_tied",
            tied_value=lambda f: f["triple"].map(lambda t: max(t.values())),
        )
        tied = tied.assign(sub1=tied["tied_value"] < 1.0 - TOL)
        gone = self.population.absent().assign(
            regime="absent@500", tied_value=float("nan"), sub1=False
        )
        columns = ["dataset", "query_id", "query", "regime", "tied_value", "sub1"]
        return pd.concat([tied[columns], gone[columns]], ignore_index=True)

    def precondition_audit(self) -> dict[str, int]:
        """§1.0 — how many residual rows can be rescored at all. A row without
        persisted route_rankings is unjudgeable spend (invariant §5a-3)."""
        res = self.residual()
        have = sum(
            row.query_id in self.sources.rankings(row.dataset)
            for row in res.itertuples(index=False)
        )
        return {
            "residual": len(res),
            "all_tied": int((res["regime"] == "all_tied").sum()),
            "sub1_ties": int(res["sub1"].sum()),
            "absent@500": int((res["regime"] == "absent@500").sum()),
            "with_rankings": have,
            "no_rankings": len(res) - have,
        }

    def _above_gold(self, dataset: str, query_id: str) -> set[str] | None:
        """`above_gold` over the persisted rankings. None when rankings are
        missing — precondition fail, drop the row."""
        orders = self.sources.rankings(dataset).get(query_id)
        if orders is None:
            return None
        return above_gold(orders, self.sources.manifest_gold(dataset).get(query_id, set()))

    def _tail_docs(self, dataset: str, query_id: str) -> dict[str, int] | None:
        """`tail_docs` over the persisted rankings. None when rankings are
        missing — precondition fail, drop the row."""
        orders = self.sources.rankings(dataset).get(query_id)
        if orders is None:
            return None
        return tail_docs(orders, self.sources.manifest_gold(dataset).get(query_id, set()))

    def tie_pairs(self, *, limit: int | None = None) -> pd.DataFrame:
        """Every tied row's judgeable pairs, widest route disagreement first.

        Serves ALL tied rows, not just the sub-1.0 ones: a tie scoring 1.0 on
        every route still has gold at rank 1 everywhere, so HitRate@1 cannot
        move, but judging a tail doc shifts NDCG and separates the routes.
        Restricting to above-gold docs (as this method's predecessor did)
        silently skipped those rows entirely.

        The margin such a break produces is bounded by the objective's NDCG
        weight — well under the margin the router calls decisive — so these
        atoms buy qrels depth, never decisive training rows.
        """
        res = self.residual()
        tied = res[res["regime"] == "all_tied"]
        if limit is not None:
            tied = tied.head(limit)

        rows: list[dict] = []
        dropped_norank = dropped_notext = no_tail = 0
        for row in tied.itertuples(index=False):
            spreads = self._tail_docs(row.dataset, row.query_id)
            if spreads is None:
                dropped_norank += 1
                continue
            if not spreads:
                # every retrieved doc sits at the same rank in every route:
                # the lists are identical, so no judgment can separate them.
                no_tail += 1
                continue
            texts = self.sources.corpus_text(row.dataset, set(spreads))
            for doc_id, spread in sorted(
                spreads.items(), key=lambda kv: (-kv[1], kv[0])
            ):
                text = texts.get(doc_id)
                if not text:
                    dropped_notext += 1
                    continue
                rows.append({
                    "dataset": row.dataset, "query_id": row.query_id,
                    "doc_id": doc_id, "query": row.query, "doc_text": text,
                    "rank_spread": spread,
                })
        print(
            f"tied rows: {len(tied)}; dropped {dropped_norank} no-rankings, "
            f"{no_tail} identical-lists, {dropped_notext} no-text; "
            f"{len(rows)} pairs to judge"
        )
        return pd.DataFrame(
            rows,
            columns=["dataset", "query_id", "doc_id", "query", "doc_text",
                     "rank_spread"],
        )

