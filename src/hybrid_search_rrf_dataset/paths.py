"""The per-lane artifact layout, derived from one data root.

Lives in a package whose `__init__` is empty on purpose: a caller that only
wants a path must not pay for the LLM engine's imports.
"""

from __future__ import annotations

from pathlib import Path
from typing import ClassVar

from pydantic import BaseModel, ConfigDict, Field

_DATA_ROOT = Path(__file__).resolve().parent.parent / "data"


class LanePaths(BaseModel):
    """Every artifact a lane owns — its queries, its answer key, its corpus and
    the retrieval results labelling produced from them. The shared owner every
    package reads instead of respelling `data_dir / lane / "*.parquet"`."""

    model_config = ConfigDict(frozen=True)

    ORACLE_SUFFIX: ClassVar[str] = "_oracle"

    data_dir: Path = Field(
        default=_DATA_ROOT,
        description="Data root holding the lane dirs, composition and pool.",
    )

    def lane_dir(self, lane: str) -> Path:
        """The lane's own directory — for a caller that hands the whole dir on
        rather than one artifact."""
        return self.data_dir / lane

    def lane_queries(self, lane: str) -> Path:
        return self.lane_dir(lane) / "queries.parquet"

    def lane_qrels(self, lane: str) -> Path:
        return self.lane_dir(lane) / "qrels.parquet"

    def lane_corpus(self, lane: str) -> Path:
        return self.lane_dir(lane) / "corpus.parquet"

    def lane_corpus_index(self, lane: str) -> Path:
        return self.lane_dir(lane) / "corpus_index.parquet"

    def lane_surfaces(self, lane: str) -> Path:
        return self.lane_dir(lane) / "surfaces.parquet"

    def lane_excluded(self, lane: str) -> Path:
        return self.lane_dir(lane) / "excluded.parquet"

    def lanes_with(self, artifact: str) -> list[str]:
        """Lanes carrying this per-lane artifact on disk. Takes a `lane_*` path's
        `.name`, so the filename keeps exactly one spelling."""
        return sorted(p.parent.name for p in self.data_dir.glob(f"*/{artifact}"))

    def lanes_with_queries(self) -> list[str]:
        return self.lanes_with(self.lane_queries("_").name)

    def lanes_with_qrels(self) -> list[str]:
        return self.lanes_with(self.lane_qrels("_").name)

    def lanes_with_corpus(self) -> list[str]:
        return self.lanes_with(self.lane_corpus("_").name)

    # ---- retrieval results ---------------------------------------------
    # `under` selects the labelling run that produced them; the default is the
    # standing `route_labels` cache. One spelling of ORACLE_SUFFIX, repo-wide.
    def oracle_root(self, under: Path | None = None) -> Path:
        return self.data_dir / "route_labels" if under is None else under

    def oracle_lane_dir(self, lane: str, *, under: Path | None = None) -> Path:
        return self.oracle_root(under) / f"{lane}{self.ORACLE_SUFFIX}"

    def oracle_rows(self, lane: str, *, under: Path | None = None) -> Path:
        """A lane's persisted per-route ranked doc lists — what scoring
        re-derives labels from without re-running retrieval."""
        return self.oracle_lane_dir(lane, under=under) / "rows.parquet"

    def oracle_lanes(self, *, under: Path | None = None) -> list[str]:
        return sorted(
            d.name.removesuffix(self.ORACLE_SUFFIX)
            for d in self.oracle_root(under).glob(f"*{self.ORACLE_SUFFIX}")
            if (d / "rows.parquet").exists()
        )
