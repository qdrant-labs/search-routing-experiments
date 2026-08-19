"""Fill `route_labels/autofusion_cache.parquet` for every labelled query — mode C
of `notebooks/autofusion_run.ipynb`, moved out of the kernel because ~46k HTTP
calls over hours has to survive a closed laptop lid. Resumable: a cached
`(dataset, query_id)` never hits the network again, so a Ctrl-C costs at most one
chunk. `--plan` prints coverage and spends nothing.

    poetry run python src/scripts/autofusion_fill.py --plan
    poetry run python src/scripts/autofusion_fill.py --limit 10
    poetry run python src/scripts/autofusion_fill.py --max-chars 1500
    poetry run python src/scripts/autofusion_fill.py 2>&1 | tee autofusion_fill.log
"""

import argparse
import os
import time
from pathlib import Path
from urllib.parse import urlparse

import pandas as pd
from dotenv import load_dotenv
from tqdm.auto import tqdm

from hybrid_search_rrf_dataset.golden import LLMScoreClient
from hybrid_search_rrf_dataset.router import LABELS_PATH, AutoFusionRouter


def _permanent(error: Exception) -> bool:
    """A 4xx is a verdict on the request itself — no retry will change it."""
    response = getattr(error, "response", None)
    return response is not None and 400 <= response.status_code < 500


def _rejection(error: Exception, text: str) -> str:
    """What the endpoint actually said, which `raise_for_status` discards."""
    body = error.response.text.strip()[:300] or "<empty body>"
    return (
        f"classify refused a {len(text):,}-char query with "
        f"{error.response.status_code}: {body}\nquery starts: {text[:120]!r}"
    )


class GuardedScoreClient:
    """Retries transient classify failures and rejects out-of-range scores.

    Closes both gaps outside `router.py`, which already takes an injected
    client: `LLMScoreClient` has no retry, and `_production_route` bands any
    integer without complaint, so an out-of-range score would be served as
    sparse for the rest of the project.
    """

    def __init__(
        self,
        client: LLMScoreClient | None = None,
        attempts: int = 3,
        backoff: float = 2.0,
        max_chars: int | None = None,
    ) -> None:
        self._client = client or LLMScoreClient()
        self._attempts = attempts
        self._backoff = backoff
        self._max_chars = max_chars

    def score(self, query: str) -> int:
        text = query[: self._max_chars] if self._max_chars else query
        for attempt in range(1, self._attempts + 1):
            try:
                score = self._client.score(text)
            except Exception as error:
                if _permanent(error):
                    error.add_note(_rejection(error, text))
                    raise
                if attempt == self._attempts:
                    raise
                time.sleep(self._backoff * attempt)
                continue
            if not 0 <= score <= LLMScoreClient.SCORE_MAX:
                raise ValueError(
                    f"classify returned {score}, outside 0-{LLMScoreClient.SCORE_MAX}"
                )
            return score


def _remaining(
    labels: pd.DataFrame, cache_path: Path = AutoFusionRouter.CACHE_PATH
) -> pd.DataFrame:
    """Rows with no cached score yet, keyed as the router caches them."""
    if not cache_path.exists():
        return labels
    cache = pd.read_parquet(cache_path, columns=["dataset", "query_id"])
    done = set(
        zip(cache["dataset"].astype(str), cache["query_id"].astype(str), strict=True)
    )
    keys = zip(labels["dataset"].astype(str), labels["query_id"].astype(str), strict=True)
    return labels[[key not in done for key in keys]]


def _fill(
    router: AutoFusionRouter,
    chunk: pd.DataFrame,
    desc: str,
    cache_path: Path = AutoFusionRouter.CACHE_PATH,
) -> list[pd.Series]:
    """Score `chunk`, stepping over queries the endpoint permanently refuses.

    The refused row is whatever the cache still lacks once the call has raised,
    which `route_batch` guarantees by saving in a `finally` — so a poison query
    costs its own row and nothing else.
    """
    refused: list[pd.Series] = []
    while len(chunk):
        try:
            router.route_batch(chunk, desc=desc)
            return refused
        except Exception as error:
            if not _permanent(error):
                raise
            unscored = _remaining(chunk, cache_path)
            if unscored.empty:
                raise
            row = unscored.iloc[0]
            refused.append(row)
            print(f"refused {row['dataset']}/{row['query_id']}: {error}")
            chunk = unscored.iloc[1:]
    return refused


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Score every labelled query with the auto-fusion classifier."
    )
    parser.add_argument(
        "--chunk", type=int, default=250,
        help="queries per cache save; a killed run replays at most this many (default: 250)",
    )
    parser.add_argument(
        "--limit", type=int,
        help="score only this many uncached queries — how to measure the call rate first",
    )
    parser.add_argument(
        "--max-chars", type=int,
        help="truncate queries to this length before sending (default: the client's 4096)",
    )
    parser.add_argument(
        "--plan", action="store_true",
        help="print coverage and stop — no calls, no spend",
    )
    args = parser.parse_args()

    load_dotenv(".env") or load_dotenv("../.env")
    labels = pd.read_parquet(LABELS_PATH)
    todo = _remaining(labels)
    print(f"cache   {AutoFusionRouter.CACHE_PATH}")
    print(f"labels  {len(labels):,}")
    print(f"cached  {len(labels) - len(todo):,}")
    print(f"todo    {len(todo):,}")
    if args.plan:
        return

    if args.limit is not None:
        todo = todo.head(args.limit)
        print(f"limited {len(todo):,}")
    if todo.empty:
        print("nothing to do")
        return

    router = AutoFusionRouter(client=GuardedScoreClient(max_chars=args.max_chars))
    print(f"host    {urlparse(os.environ['QDRANT_LLM_FUSION_URL']).netloc}")
    print(f"started {pd.Timestamp.now():%Y-%m-%d %H:%M}")

    started = time.perf_counter()
    refused: list[pd.Series] = []
    for start in tqdm(range(0, len(todo), args.chunk), desc="chunks"):
        chunk = todo.iloc[start : start + args.chunk]
        refused += _fill(router, chunk, desc=f"rows {start:,}")
    rate = (time.perf_counter() - started) / len(todo)

    print(f"{rate:.2f}s/call over {len(todo):,} queries")
    print(f"cache now {len(pd.read_parquet(AutoFusionRouter.CACHE_PATH)):,} rows")
    if refused:
        print(f"\n{len(refused):,} queries the endpoint refused, still unscored:")
        for row in refused:
            print(f"  {row['dataset']}/{row['query_id']}  {len(str(row['query'])):,} chars")


if __name__ == "__main__":
    main()
