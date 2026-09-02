"""The judge queue: which rows measurement cannot reach, and which (query, doc)
pairs to judge for each. Selection reuses the l2-triple regime classifier; pair
construction is the objective's own top-10 window, minus what cannot move a score.
"""

from __future__ import annotations

import json

import pandas as pd

from relevance_judge.config import RelevanceJudgeConfig
from relevance_judge.sources import ROUTES, Sources

TOL = 1e-9


def regime(triple: dict[str, float]) -> str:
    """all_zero / all_tied / low_margin / decisive_strong — the arch5k §8
    classifier over a route-score triple (copied, 4 lines, not imported)."""
    values = sorted(triple.values(), reverse=True)
    if values[0] <= TOL:
        return "all_zero"
    if values[0] - values[-1] <= TOL:
        return "all_tied"
    return "low_margin" if values[0] - values[1] < 0.1 else "decisive_strong"


class JudgeQueue:
    """Residual population + the pairs that could break each row's tie."""

    def __init__(
        self,
        config: RelevanceJudgeConfig | None = None,
        sources: Sources | None = None,
    ) -> None:
        self.config = config or RelevanceJudgeConfig()
        self.sources = sources or Sources(self.config)

    def _draw(self) -> list[dict]:
        return json.loads((self.config.arch5k / "rows.json").read_text())

    def residual(self) -> pd.DataFrame:
        """Every row measurement left unlabelled: l2 all_tied ∪ absent@500.
        Columns dataset, query_id, query, regime, tied_value, sub1."""
        rows = self._draw()
        qtext = {(r["dataset"], str(r["query_id"])): r["query"] for r in rows}

        tied = [
            {
                "dataset": r["dataset"], "query_id": str(r["query_id"]),
                "query": r["query"], "regime": "all_tied",
                "tied_value": max(r["l2"].values()),
                "sub1": max(r["l2"].values()) < 1.0 - TOL,
            }
            for r in rows if regime(r["l2"]) == "all_tied"
        ]

        depth = pd.read_parquet(
            self.config.arch5k / "depth_probe.parquet"
        ).astype({"query_id": str})
        rank_cols = [f"rank_{route}" for route in ROUTES]
        absent = depth[depth[rank_cols].isna().all(axis=1)]
        gone = [
            {
                "dataset": a.dataset, "query_id": str(a.query_id),
                "query": qtext.get((a.dataset, str(a.query_id)), ""),
                "regime": "absent@500", "tied_value": float("nan"), "sub1": False,
            }
            for a in absent.itertuples(index=False)
        ]
        return pd.DataFrame(tied + gone, columns=[
            "dataset", "query_id", "query", "regime", "tied_value", "sub1",
        ])

    def precondition_audit(self) -> dict[str, int]:
        """§1.0 — how many residual rows can be rescored at all. A row without
        persisted route_rankings is unjudgeable spend (invariant §5a-3)."""
        res = self.residual()
        have = sum(
            row.query_id in self.sources.rankings(row.dataset)
            for row in res.itertuples(index=False)
        )
        counts = {
            "residual": len(res),
            "all_tied": int((res["regime"] == "all_tied").sum()),
            "sub1_ties": int(res["sub1"].sum()),
            "absent@500": int((res["regime"] == "absent@500").sum()),
            "with_rankings": have,
            "no_rankings": len(res) - have,
        }
        return counts

    def _above_gold(self, dataset: str, query_id: str) -> set[str] | None:
        """Docs ranked above the gold in any route (the only docs whose relevance
        can break a sub-1.0 tie). None when rankings are missing — precondition
        fail, drop the row. Gold itself is excluded (already judged)."""
        ranks = self.sources.rankings(dataset).get(query_id)
        if ranks is None:
            return None
        gold = self.sources.manifest_gold(dataset).get(query_id, set())
        above: set[str] = set()
        for order in ranks.values():
            cut = next((i for i, doc in enumerate(order) if doc in gold), len(order))
            above.update(order[:cut])
        return above - gold

    def sub1_pairs(self, *, limit: int | None = None) -> pd.DataFrame:
        """The sub-1.0-tie pilot's work list: above-gold docs only, with text.
        Columns dataset, query_id, doc_id, query, doc_text."""
        res = self.residual()
        sub1 = res[(res["regime"] == "all_tied") & res["sub1"]]
        if limit is not None:
            sub1 = sub1.head(limit)

        rows: list[dict] = []
        dropped_norank = dropped_notext = empty_abovegold = 0
        for row in sub1.itertuples(index=False):
            above = self._above_gold(row.dataset, row.query_id)
            if above is None:
                dropped_norank += 1
                continue
            if not above:
                # gold already at rank 1 in every persisted (v2) route — the
                # sub-1.0 tie is an l2-stack fact these rankings can't express.
                empty_abovegold += 1
                continue
            texts = self.sources.corpus_text(row.dataset, above)
            for doc_id in above:
                text = texts.get(doc_id)
                if not text:
                    dropped_notext += 1
                    continue
                rows.append({
                    "dataset": row.dataset, "query_id": row.query_id,
                    "doc_id": doc_id, "query": row.query, "doc_text": text,
                })
        print(
            f"sub-1.0 ties: {len(sub1)} rows; dropped {dropped_norank} no-rankings, "
            f"{empty_abovegold} gold-at-v2-rank-1, {dropped_notext} no-text; "
            f"{len(rows)} pairs to judge"
        )
        return pd.DataFrame(
            rows, columns=["dataset", "query_id", "doc_id", "query", "doc_text"]
        )


def _self_check() -> None:
    assert regime({"a": 0.0, "b": 0.0, "c": 0.0}) == "all_zero"
    assert regime({"a": 0.3, "b": 0.3, "c": 0.3}) == "all_tied"
    assert regime({"a": 1.0, "b": 0.95, "c": 0.9}) == "low_margin"
    assert regime({"a": 1.0, "b": 0.2, "c": 0.1}) == "decisive_strong"
    print("queue regime self-check ok")


if __name__ == "__main__":
    _self_check()
