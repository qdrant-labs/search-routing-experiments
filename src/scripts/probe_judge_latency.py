"""Time ONE judge call and report what it cost — the reasoning-suppression check.

Reads ~2-5s if `reasoning_effort` is honoured, ~60s if luna reasons anyway.
`tokens` far above ~150 means hidden reasoning tokens are still being generated.

    poetry run python src/scripts/probe_judge_latency.py
"""

from __future__ import annotations

import time

from relevance_judge import RelevanceJudge, RelevanceJudgeConfig

QUERY = "what is a tort?"
DOC = "A tort is a civil wrong that causes harm to another person."


def main() -> None:
    config = RelevanceJudgeConfig()
    judge = RelevanceJudge(config)
    print(f"model            : {judge.model}")
    print(f"reasoning_effort : {config.reasoning_effort!r}")
    print(f"request_timeout  : {config.request_timeout_s}s")
    print("calling...", flush=True)

    start = time.monotonic()
    verdict = judge.judge_one(QUERY, DOC)
    elapsed = time.monotonic() - start
    relevant, reason, spend = verdict.relevant, verdict.reason, verdict.spend

    print(f"\nelapsed  : {elapsed:.1f}s")
    print(f"verdict  : {relevant}")
    print(f"reason   : {reason!r}")
    for name, value in verdict.fields.items():
        print(f"  {name:8s}: {value!r}")
    print(f"tokens   : {spend.tokens}")
    print(f"dropped  : {dict(judge.dropped) or 'none'}")

    if relevant is None:
        print("\n-> NO VERDICT. Check `dropped` above: an error name means the "
              "provider rejected or timed out the call (a bad reasoning_effort "
              "value lands here); 'unparsed' means it answered unreadably.")
    elif elapsed < 15:
        workers = config.llm_workers
        print(f"\n-> Suppression WORKS. At {workers} workers a 3,600-pair gate is "
              f"~{3600 * elapsed / workers / 60:.0f} min.")
    else:
        print(f"\n-> Still slow: luna is reasoning despite effort="
              f"{config.reasoning_effort!r}. {spend.tokens} tokens for a yes/no "
              "points at hidden reasoning. Next lever is a non-reasoning model "
              "(costs a gate re-validation) or the :batch variant.")


if __name__ == "__main__":
    main()
