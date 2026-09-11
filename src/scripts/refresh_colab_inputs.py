"""Mirror the union_230k deepdive notebook's inputs into their own small DVC out.

`src/data` is one 10 GB directory out, and DVC cannot check a single file out of
a directory out — so Colab either pulls everything or nothing. This publishes the
small analysis inputs needed by the Colab notebooks as a separate DVC out.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

DATA = Path(__file__).resolve().parents[1] / "data"
COLAB_OUT = Path(__file__).resolve().parents[1] / "data_colab"

INPUTS = (
    "legb_pilot/os_distill_relabel/cascade_labels.parquet",
    "rungs/100k-v2/planned_set.parquet",
    "rungs/100k-v2/labeling/labels.parquet",
    "route_labels/labels.parquet",
    "v3/labels.parquet",
    "v3/augmented/labels.parquet",
    "v3/dataset_v3.parquet",
    "feature_table/catalog.parquet",
    "home-depot/queries.parquet",
    "home-depot/qrels.parquet",
)


def refresh(data: Path = DATA, out: Path = COLAB_OUT, *, push: bool = True) -> None:
    """Rebuild the out from `data`, then hand it to DVC."""
    if out.exists():
        shutil.rmtree(out)
    for relative in INPUTS:
        destination = out / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(data / relative, destination)
        print(f"{destination.stat().st_size / 1e6:6.1f} MB  {relative}")
    subprocess.run(["dvc", "add", str(out)], check=True)
    if push:
        subprocess.run(["dvc", "push", f"{out}.dvc"], check=True)


if __name__ == "__main__":
    refresh()
