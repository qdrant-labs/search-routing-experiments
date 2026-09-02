"""Run-scoped labeling contract for an immutable Rung A planned set."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

import pandas as pd


class LabelRunError(RuntimeError):
    """The plan or run directory violates a labeling integrity condition."""


class LabelRegime(StrEnum):
    """The corpus snapshot against which a planned row must be measured."""

    NATURAL = "natural"
    SUPPLEMENTED = "supplemented"
    CONSTRUCTED = "constructed"


@dataclass(frozen=True)
class Completion:
    planned: int
    labelled: int
    missing: int
    published: bool


class RungLabelRun:
    """Validate, partition, own, and complete one immutable labeling run.

    Historical label membership is deliberately absent. Resume is scoped to
    this run directory and its exact plan fingerprint.
    """

    REQUIRED_PLAN_COLUMNS = frozenset({
        "row_id",
        "dataset",
        "query_id",
        "query",
        "provenance",
        "operator",
        "home_lane",
        "answer_covered",
        "answer_manifest_id",
        "content_fp",
    })
    REQUIRED_LABEL_COLUMNS = frozenset({
        "dataset",
        "query_id",
        "route",
        "shape",
        "score_dense_only",
        "score_pure_rrf",
        "score_sparse_only",
    })
    SCHEMA_VERSION = 1

    def __init__(self, plan: pd.DataFrame, *, out_dir: Path) -> None:
        self.plan = plan.copy()
        self.out_dir = Path(out_dir)
        self._validate_plan()
        self.plan = self.plan.astype({"dataset": str, "query_id": str})
        self.plan_fp = _plan_fingerprint(self.plan)

    @classmethod
    def from_path(cls, plan_path: Path, *, out_dir: Path | None = None) -> RungLabelRun:
        path = Path(plan_path)
        plan = pd.read_parquet(path)
        destination = Path(out_dir) if out_dir is not None else path.parent / "labeling"
        return cls(plan, out_dir=destination)

    def partitions(self) -> dict[LabelRegime, pd.DataFrame]:
        provenance = self.plan["provenance"].fillna("").astype(str)
        operator = self.plan["operator"].fillna("").astype(str)
        constructed = operator == "synthesize"
        natural = (provenance == "natural") & ~constructed
        supplemented = ~(natural | constructed)
        masks = {
            LabelRegime.NATURAL: natural,
            LabelRegime.SUPPLEMENTED: supplemented,
            LabelRegime.CONSTRUCTED: constructed,
        }
        return {
            regime: self.plan.loc[mask].copy().reset_index(drop=True)
            for regime, mask in masks.items()
        }

    def remaining_partitions(self) -> dict[LabelRegime, pd.DataFrame]:
        """Planned rows not yet checkpointed by this same run and regime."""
        self._assert_owner_if_present()
        remaining: dict[LabelRegime, pd.DataFrame] = {}
        for regime, partition in self.partitions().items():
            path = self.regime_dir(regime) / "labels.parquet"
            if not path.exists():
                remaining[regime] = partition
                continue
            labels = pd.read_parquet(path, columns=["dataset", "query_id"]).astype(
                {"dataset": str, "query_id": str}
            )
            keys = list(zip(labels["dataset"], labels["query_id"]))
            if len(keys) != len(set(keys)):
                raise LabelRunError(f"{path} contains duplicate label identities")
            expected = set(zip(partition["dataset"], partition["query_id"]))
            unexpected = set(keys) - expected
            if unexpected:
                raise LabelRunError(
                    f"{path} contains {len(unexpected)} rows outside its {regime} partition"
                )
            seen = set(keys)
            row_keys = list(zip(partition["dataset"], partition["query_id"]))
            remaining[regime] = partition.loc[
                [key not in seen for key in row_keys]
            ].reset_index(drop=True)
        return remaining

    def initialize(self, *, plan_path: Path) -> None:
        """Claim the output directory for this exact plan, or resume it."""
        self._assert_owner_if_present()
        manifest = {
            "schema_version": self.SCHEMA_VERSION,
            "plan_fp": self.plan_fp,
            "plan_path": str(plan_path),
            "planned_rows": len(self.plan),
        }
        path = self.out_dir / "run.json"
        if path.exists():
            return
        self.out_dir.mkdir(parents=True, exist_ok=True)
        _atomic_text(path, json.dumps(manifest, indent=2, sort_keys=True) + "\n")

    def regime_dir(self, regime: LabelRegime) -> Path:
        return self.out_dir / regime.value

    def publish_if_complete(self) -> Completion:
        """Publish one exact, plan-ordered label artifact only at completeness."""
        self._assert_owner_if_present(require=True)
        frames: list[pd.DataFrame] = []
        for regime in LabelRegime:
            path = self.regime_dir(regime) / "labels.parquet"
            if path.exists():
                frame = pd.read_parquet(path)
                missing_columns = self.REQUIRED_LABEL_COLUMNS - set(frame.columns)
                if missing_columns:
                    raise LabelRunError(
                        f"{path} missing label columns {sorted(missing_columns)}"
                    )
                frames.append(frame.astype({"dataset": str, "query_id": str}))

        labels = (
            pd.concat(frames, ignore_index=True)
            if frames
            else pd.DataFrame(columns=sorted(self.REQUIRED_LABEL_COLUMNS))
        )
        label_keys = list(zip(labels["dataset"], labels["query_id"]))
        if len(label_keys) != len(set(label_keys)):
            raise LabelRunError("label shards contain duplicate (dataset, query_id) rows")

        plan_keys = list(zip(self.plan["dataset"], self.plan["query_id"]))
        expected = set(plan_keys)
        unexpected = set(label_keys) - expected
        if unexpected:
            raise LabelRunError(
                f"label shards contain {len(unexpected)} rows outside the frozen plan"
            )
        missing = expected - set(label_keys)
        if missing:
            return Completion(
                planned=len(self.plan),
                labelled=len(label_keys),
                missing=len(missing),
                published=False,
            )

        order = self.plan[["dataset", "query_id"]].assign(_plan_rank=range(len(self.plan)))
        completed = (
            order.merge(labels, on=["dataset", "query_id"], how="left", validate="one_to_one")
            .sort_values("_plan_rank", kind="stable")
            .drop(columns="_plan_rank")
        )
        _atomic_parquet(completed, self.out_dir / "labels.parquet")
        completion = Completion(
            planned=len(self.plan),
            labelled=len(completed),
            missing=0,
            published=True,
        )
        _atomic_text(
            self.out_dir / "completion.json",
            json.dumps(
                {
                    "plan_fp": self.plan_fp,
                    "planned": completion.planned,
                    "labelled": completion.labelled,
                    "missing": completion.missing,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
        )
        return completion

    def _validate_plan(self) -> None:
        missing = self.REQUIRED_PLAN_COLUMNS - set(self.plan.columns)
        if missing:
            raise LabelRunError(f"planned set missing columns {sorted(missing)}")
        keys = list(
            zip(self.plan["dataset"].astype(str), self.plan["query_id"].astype(str))
        )
        if len(keys) != len(set(keys)):
            raise LabelRunError("planned set contains duplicate (dataset, query_id) rows")
        if self.plan["row_id"].astype(str).duplicated().any():
            raise LabelRunError("planned set contains duplicate row_id values")
        covered = self.plan["answer_covered"].fillna(False).eq(True)
        manifests = self.plan["answer_manifest_id"].fillna("").astype(str)
        invalid = ~(covered & manifests.ne(""))
        if invalid.any():
            raise LabelRunError(
                f"planned set contains {int(invalid.sum())} rows without answer coverage"
            )

    def _assert_owner_if_present(self, *, require: bool = False) -> None:
        """Accept checkpoints only when this exact plan owns the directory."""
        manifest_path = self.out_dir / "run.json"
        if manifest_path.exists():
            existing = json.loads(manifest_path.read_text())
            if existing.get("schema_version") != self.SCHEMA_VERSION:
                raise LabelRunError(
                    f"{self.out_dir} has unsupported label-run schema "
                    f"{existing.get('schema_version')!r}"
                )
            if existing.get("plan_fp") != self.plan_fp:
                raise LabelRunError(
                    f"{self.out_dir} belongs to a different plan "
                    f"({existing.get('plan_fp')} != {self.plan_fp})"
                )
            return
        if self.out_dir.exists() and any(self.out_dir.iterdir()):
            raise LabelRunError(
                f"{self.out_dir} contains checkpoints but is not owned by run.json"
            )
        if require:
            raise LabelRunError(f"{self.out_dir} is not an initialized label run")


def _plan_fingerprint(plan: pd.DataFrame) -> str:
    rows = sorted(
        (
            str(row.row_id),
            str(row.dataset),
            str(row.query_id),
            str(row.content_fp),
            str(row.answer_manifest_id),
        )
        for row in plan.itertuples(index=False)
    )
    payload = json.dumps(rows, separators=(",", ":"), ensure_ascii=False).encode()
    return hashlib.sha256(payload).hexdigest()


def _atomic_text(path: Path, payload: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    try:
        tmp.write_text(payload)
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def _atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    try:
        frame.to_parquet(tmp, index=False)
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)
