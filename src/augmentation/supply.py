"""The supply index (d43c): full per-lane surface artifacts.

One idempotent regex pass per lane corpus writes
`data/<lane>/surfaces.parquet` — (doc_id, bank, floor, surface) — in
composition floor-key units, so demand (the order sheet) and supply read
against each other. Serves all three consumers: Inject's rung-1 pair
joins, the d40(g) inversion lookup, and the rung-assignment readout that
prices every hungry id floor BEFORE any LLM spend.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Final

import pandas as pd
from tqdm.auto import tqdm

from augmentation.config import AugmentationPaths
from composition.floors import identifier_floor_key
from query_taxonomy.features import FeatureExtractor
from query_taxonomy.taxonomy import FeatureGroup

_COLUMNS: Final[tuple[str, ...]] = (
    "doc_id",   # corpus row it was read from — becomes Inject's grounding doc
    "bank",     # bank that claimed the span — becomes Inject's SpanTarget
    "floor",    # floor the bank rolls up to (id:medical) — the demand unit
    "surface",  # the span text itself, woven into the child verbatim
)


@lru_cache(maxsize=1)
def lane_dirs() -> dict[str, str]:
    """Composition dataset key -> lane data dir name (e.g. 'beir-nfcorpus'
    -> 'nfcorpus'). Lazy: the LANES import pulls the retrieval stack."""
    from hybrid_search_rrf_dataset.lanes import LANES

    return {key: lane.source.name for key, lane in LANES.items()}


class SupplyIndex:
    """Owns the per-lane `surfaces.parquet` artifacts."""

    def __init__(
        self,
        paths: AugmentationPaths | None = None,
        extractor: FeatureExtractor | None = None,
    ) -> None:
        self._paths = paths or AugmentationPaths()
        self._extractor = extractor or FeatureExtractor()
        self._verdicts: dict[tuple[str, str], bool] = {}
        """Memoized claimability per (bank, surface) — instance-scoped because
        the verdict depends on the injected extractor, and per PAIR rather
        than per frame so floors drawing the same bank share the work."""

    def path(self, lane: str) -> Path:
        return self._paths.lane_surfaces(lane)

    def lanes_on_disk(self) -> list[str]:
        return sorted(
            p.parent.name for p in self._paths.data_dir.glob("*/corpus.parquet")
        )

    def build(self, lane: str, *, force: bool = False) -> pd.DataFrame:
        """Scan one lane corpus (full text, no truncation) and persist its
        surfaces. Idempotent — the scan is the expensive leg."""
        target = self.path(lane)
        if target.exists() and not force:
            return pd.read_parquet(target)
        corpus = pd.read_parquet(self._paths.lane_corpus(lane))
        texts = corpus["text"].fillna("")
        if "title" in corpus.columns:
            texts = corpus["title"].fillna("") + "\n" + texts

        rows: list[dict[str, str]] = []
        for doc_id, text in tqdm(
            zip(corpus["doc_id"].astype(str), texts, strict=True),
            total=len(corpus),
            desc=f"surfaces:{lane}",
        ):
            found = self._extractor.resolve(
                str(text), groups=[FeatureGroup.STRUCTURED_IDENTIFIERS]
            ).spans.get(FeatureGroup.STRUCTURED_IDENTIFIERS, {})
            for bank, spans in found.items():
                floor = identifier_floor_key(bank)
                rows.extend(
                    {
                        "doc_id": doc_id,
                        "bank": bank,
                        "floor": floor,
                        "surface": span.text,
                    }
                    for span in spans
                )
        surfaces = pd.DataFrame(rows, columns=_COLUMNS).drop_duplicates()
        surfaces.to_parquet(target, index=False)
        print(f"{lane}: {len(surfaces):,} surfaces from {len(corpus):,} docs -> {target}")
        return surfaces

    def build_all(self, *, force: bool = False) -> None:
        for lane in self.lanes_on_disk():
            self.build(lane, force=force)

    def load(self, lane: str) -> pd.DataFrame:
        if not self.path(lane).exists():
            return pd.DataFrame(columns=_COLUMNS)
        return pd.read_parquet(self.path(lane))

    def _claims(self, bank: str, surface: str) -> bool:
        """Whether the surface ALONE still resolves to the bank that mined it
        and to no other identifier floor — the two ways a mined row fails the
        target it was selected for."""
        if (bank, surface) in self._verdicts:
            return self._verdicts[bank, surface]
        found = self._extractor.resolve(
            surface, groups=[FeatureGroup.STRUCTURED_IDENTIFIERS]
        ).spans.get(FeatureGroup.STRUCTURED_IDENTIFIERS, {})
        claimed = {name for name, spans in found.items() if spans}
        verdict = bank in claimed and {
            identifier_floor_key(b) for b in claimed
        } == {identifier_floor_key(bank)}
        self._verdicts[bank, surface] = verdict
        return verdict

    def claimable(self, surfaces: pd.DataFrame, lane: str = "") -> pd.DataFrame:
        """The mined rows that can still serve the target they get selected
        for. A bank repair invalidates every artifact mined with the old bank
        and nothing else in the pipeline notices, so the rate is printed: a
        lane losing most of one bank is a re-mine signal, not bad luck.

        Called on a demand's OWN slice, never the whole lane — resolving every
        mined surface costs seconds per lane, and a floor only ever draws from
        the banks it named."""
        if surfaces.empty:
            return surfaces
        kept = surfaces[
            [
                self._claims(str(bank), str(surface))
                for bank, surface in zip(surfaces["bank"], surfaces["surface"])
            ]
        ]
        dropped = len(surfaces) - len(kept)
        if not dropped:
            return kept
        mined = surfaces.groupby("bank").size()
        share = kept.groupby("bank").size().reindex(mined.index, fill_value=0) / mined
        rotten = sorted(share[share < 0.5].index)
        note = f" — mostly {', '.join(rotten)}" if rotten else ""
        print(
            f"[{lane}] supply: dropped {dropped:,}/{len(surfaces):,} rows "
            f"({dropped / len(surfaces):.1%}) their bank no longer claims{note}"
        )
        return kept

    def readout(
        self,
        selection: pd.DataFrame,
        sheet_path: Path | str | None = None,
    ) -> pd.DataFrame:
        """Rung capacities per hungry id floor and lane (d43c/f):
        `direct_parents` = rung-1 pairs (parent lacks the floor, its gold
        doc carries a surface); `inversion_docs` = rung-2 lookup (any lane
        doc carrying one). A floor at 0/0 everywhere belongs to the
        synthetic rung."""
        sheet = pd.read_parquet(sheet_path or self._paths.order_sheet)
        hungry = [
            floor
            for floor in sheet.loc[sheet["missing"] > 0, "floor"]
            if floor.startswith("id:")
        ]
        rows: list[dict[str, object]] = []
        for key, lane in lane_dirs().items():
            surfaces = self.load(lane)
            qrels_path = self._paths.lane_qrels(lane)
            if surfaces.empty or not qrels_path.exists():
                continue
            qrels = pd.read_parquet(qrels_path)
            qrels = qrels[qrels["relevance"] >= 1].astype(
                {"query_id": str, "doc_id": str}
            )
            lane_selection = selection[
                (selection["dataset"] == key) & selection["checkable"]
            ]
            for floor in hungry:
                floor_docs = set(surfaces.loc[surfaces["floor"] == floor, "doc_id"])
                if not floor_docs:
                    continue
                judged = qrels[qrels["doc_id"].isin(floor_docs)]
                lacks = ~lane_selection["floors"].map(lambda fs: floor in fs)
                parents = set(lane_selection.loc[lacks, "query_id"].astype(str))
                rows.append({
                    "floor": floor,
                    "lane": key,
                    "inversion_docs": len(floor_docs),
                    "direct_parents": len(
                        set(judged["query_id"]) & parents
                    ),
                })
        return pd.DataFrame(
            rows, columns=["floor", "lane", "inversion_docs", "direct_parents"]
        )
