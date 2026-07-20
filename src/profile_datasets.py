"""First full multi-group profile — over the GROUNDED retrieval snapshots
(queries that survived qrels-consistent materialization), so profiles align
with the exact query sets the demo/NDCG framework runs on.

Loads each RetrievalDataset snapshot (materializing it first if missing),
runs all engines (regex + GLiNER2 + spaCy) over its queries, and persists
one CorpusFeatures JSON per dataset plus its summary() report to
src/data/profiles/.

    poetry run python src/profile_datasets.py

Explore in the notebook:

    from query_taxonomy.features import CorpusFeatures
    profile = CorpusFeatures.model_validate_json(
        (PROFILES_DIR / "trec-dl-2022.json").read_text()
    )
"""

from pathlib import Path

import pandas as pd
from tqdm.auto import tqdm

from hybrid_search_rrf_dataset.retrieval import (
    NFCorpus,
    RetrievalDataset,
    SciFact,
    TrecDL2022,
)
from query_taxonomy.features import FeatureExtractor

DATA_DIR = Path(__file__).resolve().parent / "data"
PROFILES_DIR = DATA_DIR / "profiles"

DATASETS: tuple[RetrievalDataset, ...] = (
    TrecDL2022(30000),
    NFCorpus(),
    SciFact(),
)


def snapshot_queries(dataset: RetrievalDataset) -> pd.DataFrame:
    """Queries from the saved snapshot; materialize + save on first use."""
    queries_path = DATA_DIR / dataset.name / "queries.parquet"
    if not queries_path.exists():
        if hasattr(dataset, "materialize"):
            dataset.materialize()
        dataset.save(DATA_DIR)
        if not queries_path.exists():
            # non-materializing datasets save straight from iterators
            return dataset.queries()
    return pd.read_parquet(queries_path)


def main() -> None:
    extractor = FeatureExtractor(engines=None)
    PROFILES_DIR.mkdir(parents=True, exist_ok=True)

    for dataset in DATASETS:
        texts = snapshot_queries(dataset)["text"].tolist()
        progress = tqdm(
            texts, desc=f"profile {dataset.name} (n={len(texts)})", unit="query"
        )
        profile = extractor.extract(progress)
        (PROFILES_DIR / f"{dataset.name}.json").write_text(
            profile.model_dump_json()
        )
        report = profile.summary()
        (PROFILES_DIR / f"{dataset.name}.summary.txt").write_text(report + "\n")
        print(f"\n{'=' * 72}\n{dataset.name}\n{'=' * 72}")
        print(report)

    print(f"\nprofiles persisted -> {PROFILES_DIR}")


if __name__ == "__main__":
    main()
