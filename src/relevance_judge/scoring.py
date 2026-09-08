"""Rescore the judged rows under human ∪ judged gold, and read off the pilot's
payoffs: the qrels-hole rate, the per-route discovered-relevance audit (doc §5),
and the tie-conversion rate (§3a payoff #1).

Merge is `QrelStore` precedence (human wins), exactly the rederive_labels path —
route_rankings scored by the existing objective, no retrieval. The rankings are
the persisted v2 stack, so `tie_conversion_rate` is only meaningful for a
v2-scored population (`V2Labels`); an l2-defined tie needs the l2 rankings,
which were never persisted.
"""

from __future__ import annotations

import pandas as pd

from hybrid_search_rrf_dataset.objective import RouterObjective
from hybrid_search_rrf_dataset.qrels import QrelStore
from relevance_judge.config import RelevanceJudgeConfig
from relevance_judge.judge import RelevanceJudge
from relevance_judge.residual import regime
from relevance_judge.sources import ROUTES, Sources


class PilotScorer:
    """Reads the judged atoms, reports what they bought."""

    def __init__(
        self,
        config: RelevanceJudgeConfig | None = None,
        sources: Sources | None = None,
        judge: RelevanceJudge | None = None,
    ) -> None:
        self.config = config or RelevanceJudgeConfig()
        self.sources = sources or Sources(self.config)
        self.judge = judge or RelevanceJudge(self.config)
        self.objective = RouterObjective(min_relevance=self.config.min_relevance)

    def _discovered_per_route(self, atoms: pd.DataFrame) -> dict:
        """Share of each route's judged top-10 docs that turned out relevant —
        the retriever-free sparse blind-spot audit. Sparse discovering relevant
        docs at a higher rate than dense = a direct sparse-truth-gap signal."""
        tally = {route: {"judged": 0, "relevant": 0} for route in ROUTES}
        for dataset, group in atoms.groupby("dataset"):
            ranks = self.sources.rankings(str(dataset))
            for row in group.itertuples(index=False):
                order_map = ranks.get(str(row.query_id))
                if not order_map:
                    continue
                for route, order in order_map.items():
                    if str(row.doc_id) in order:
                        tally[route]["judged"] += 1
                        tally[route]["relevant"] += int(row.relevance)
        return {
            route: {
                **counts,
                "discovered_relevance_rate": (
                    counts["relevant"] / counts["judged"]
                    if counts["judged"] else float("nan")
                ),
            }
            for route, counts in tally.items()
        }

    def audit(self) -> dict:
        atoms = self.judge.load()
        if atoms.empty:
            return {"error": "no judged atoms yet"}
        atoms = atoms.astype({"query_id": str, "doc_id": str, "relevance": int})
        datasets = sorted(atoms["dataset"].unique())

        human = self.sources.human_store(datasets)
        merged = QrelStore.concat([human, self.judge.as_qrelstore()])

        moved = []
        for dataset in datasets:
            before, after = human.lookup(dataset), merged.lookup(dataset)
            ranks = self.sources.rankings(dataset)
            for query_id in atoms[atoms["dataset"] == dataset]["query_id"].unique():
                order_map = ranks.get(query_id)
                if not order_map:
                    continue
                assess = self.objective.assess_order
                scores_before = {r: assess(o, before.get(query_id, {}))[0] for r, o in order_map.items()}
                scores_after = {r: assess(o, after.get(query_id, {}))[0] for r, o in order_map.items()}
                reg_before, reg_after = regime(scores_before), regime(scores_after)
                moved.append({
                    "dataset": dataset, "query_id": query_id,
                    "regime_before": reg_before, "regime_after": reg_after,
                    "resolved": reg_before == "all_tied" and reg_after != "all_tied",
                })
        rescored = pd.DataFrame(moved)

        relevant = atoms[atoms["relevance"] == 1]
        rows_total = atoms[["dataset", "query_id"]].drop_duplicates().shape[0]
        rows_with_hole = relevant[["dataset", "query_id"]].drop_duplicates().shape[0]
        return {
            "atoms": int(len(atoms)),
            "rows_judged": rows_total,
            "qrels_hole_rate_pairs": float((atoms["relevance"] == 1).mean()),
            "qrels_hole_rate_rows": rows_with_hole / rows_total if rows_total else float("nan"),
            "discovered_relevance_by_route": self._discovered_per_route(atoms),
            "tie_conversion_rate": (
                float(rescored["resolved"].mean()) if not rescored.empty else float("nan")
            ),
            "regime_after_counts": (
                rescored["regime_after"].value_counts().to_dict() if not rescored.empty else {}
            ),
        }


