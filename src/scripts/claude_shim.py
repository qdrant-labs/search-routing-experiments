"""An OpenAI-compatible endpoint served by `claude -p`, so verdict calls spend
a Claude Code subscription instead of API credits. Single-shot text only — the
weaver's tool-loop needs tool calls this cannot return.

    poetry run python src/scripts/claude_shim.py
    JUDGE_MODEL=openai/sonnet poetry run python src/scripts/run_v3_generation.py --llm-coherence
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Final

HOST: Final = "127.0.0.1"
PORT: Final = int(os.getenv("CLAUDE_SHIM_PORT", "8787"))
FALLBACK_MODEL: Final = os.getenv("CLAUDE_SHIM_MODEL", "sonnet")
PROCESSES: Final = int(os.getenv("CLAUDE_SHIM_PROCESSES", "4"))
TIMEOUT_S: Final = int(os.getenv("CLAUDE_SHIM_TIMEOUT", "300"))

WORKDIR: Final = Path(os.getenv("TMPDIR", "/tmp")) / "claude-shim"
"""Outside the repo: a discovered CLAUDE.md would join every verdict prompt."""

SUBSCRIPTION_ENV: Final = {
    key: value
    for key, value in os.environ.items()
    if key not in ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN")
}
"""Either key outranks the OAuth profile, silently restoring API billing."""

DENIED_TOOLS: Final = (
    "Bash Read Write Edit Glob Grep WebFetch WebSearch Task TodoWrite NotebookEdit"
)


def split_roles(messages: list[dict]) -> tuple[str, str]:
    """The (system, user) halves of a chat request, each role's turns joined."""
    halves: dict[bool, list[str]] = {True: [], False: []}
    for message in messages:
        halves[message.get("role") == "system"].append(str(message.get("content", "")))
    return "\n\n".join(halves[True]), "\n\n".join(halves[False])


def as_chat_completion(envelope: dict, model: str) -> dict:
    """The CLI's result envelope in the shape litellm's openai provider reads."""
    usage = envelope.get("usage") or {}
    prompt_tokens = int(usage.get("input_tokens") or 0) + int(
        usage.get("cache_read_input_tokens") or 0
    )
    completion_tokens = int(usage.get("output_tokens") or 0)
    return {
        "id": f"chatcmpl-{envelope.get('session_id', 'shim')}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "finish_reason": "stop",
                "message": {
                    "role": "assistant",
                    "content": envelope.get("result") or "",
                },
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }


class ClaudeCLI:
    """One `claude -p` process per request, capped in flight: each carries a
    whole agent runtime, so the cap is memory rather than sockets."""

    def __init__(self, processes: int = PROCESSES) -> None:
        self._slots = threading.Semaphore(processes)

    def complete(self, messages: list[dict], model: str) -> dict:
        system, prompt = split_roles(messages)
        command = [
            "claude",
            "-p",
            prompt,
            "--output-format",
            "json",
            "--model",
            model,
            "--strict-mcp-config",
            "--disable-slash-commands",
            "--disallowed-tools",
            DENIED_TOOLS,
        ]
        if system:
            command += ["--system-prompt", system]
        with self._slots:
            done = subprocess.run(
                command,
                capture_output=True,
                text=True,
                cwd=WORKDIR,
                env=SUBSCRIPTION_ENV,
                timeout=TIMEOUT_S,
            )
        if done.returncode != 0:
            raise RuntimeError(done.stderr.strip()[-500:] or "claude exited nonzero")
        return json.loads(done.stdout)


class Handler(BaseHTTPRequestHandler):
    cli = ClaudeCLI()

    def do_POST(self) -> None:
        length = int(self.headers.get("content-length") or 0)
        body = json.loads(self.rfile.read(length) or b"{}")
        model = str(body.get("model") or FALLBACK_MODEL).rsplit("/", 1)[-1]
        try:
            envelope = self.cli.complete(body["messages"], model)
        except Exception as exc:
            print(f"claude_shim: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
            self._reply(500, {"error": {"message": str(exc), "type": "claude_cli_error"}})
            return
        self._reply(200, as_chat_completion(envelope, model))

    def _reply(self, status: int, payload: dict) -> None:
        raw = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def log_message(self, *_args) -> None:
        """Access lines would bury the crank's progress bar."""


def _selftest() -> None:
    system, prompt = split_roles(
        [
            {"role": "system", "content": "audit this"},
            {"role": "user", "content": "Query: foo"},
            {"role": "assistant", "content": "yes"},
        ]
    )
    assert system == "audit this", system
    assert prompt == "Query: foo\n\nyes", prompt

    payload = as_chat_completion(
        {
            "result": "yes - the doc answers it",
            "session_id": "abc",
            "usage": {"input_tokens": 10, "cache_read_input_tokens": 2, "output_tokens": 5},
        },
        "sonnet",
    )
    assert payload["choices"][0]["message"]["content"] == "yes - the doc answers it"
    assert payload["usage"]["total_tokens"] == 17, payload["usage"]
    assert as_chat_completion({}, "sonnet")["usage"]["total_tokens"] == 0
    print("selftest ok")


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        _selftest()
        raise SystemExit(0)
    WORKDIR.mkdir(parents=True, exist_ok=True)
    print(
        f"claude_shim http://{HOST}:{PORT}/v1 "
        f"model={FALLBACK_MODEL} processes={PROCESSES}"
    )
    ThreadingHTTPServer((HOST, PORT), Handler).serve_forever()
