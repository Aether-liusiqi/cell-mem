#!/usr/bin/env python3
"""Cell-mem session recording hook script.

This script is installed into the platform's hook directory by
``cell-mem --hooks install`` and invoked by the agent platform
(Codex CLI or Claude Code) on SessionStart and PostToolUse events.

It receives event JSON on stdin, normalizes it, and delivers it to
cell-mem for asynchronous storage.

**Primary delivery: JSONL file** — appends one JSON line to a shared
queue file. This works inside Codex's sandbox (no network required).
cell-mem's IngestServer monitors the file and processes new lines.

**Fallback: HTTP POST** — for non-sandbox environments (Claude Code),
retries up to 5 times to cover MCP subprocess startup delay.

**Zero cell-mem dependency** — uses only Python stdlib.
**Never blocks** — any failure is silently swallowed.
"""

from __future__ import annotations

import json
import os
import sys
import time
import traceback
from pathlib import Path
from urllib import request as urlrequest
from urllib.error import URLError

# ---------------------------------------------------------------------------
# Configuration (overridden by environment)
# ---------------------------------------------------------------------------

INGEST_PORT = os.environ.get("CELL_MEM_INGEST_PORT", "8766")
INGEST_URL = f"http://127.0.0.1:{INGEST_PORT}/ingest"
INGEST_FILE = os.environ.get(
    "CELL_MEM_INGEST_FILE",
    str(Path.home() / ".cell_mem" / "ingest_queue.jsonl"),
)
HTTP_TIMEOUT = 2      # seconds per attempt
MAX_CONTENT_LENGTH = 10000
MAX_RETRIES = 3       # fewer retries — file delivery is instant
RETRY_DELAY = 0.5


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------


def _read_stdin() -> dict:
    """Read event JSON from stdin. Returns empty dict on any failure."""
    if sys.stdin.isatty():
        return {}

    try:
        raw = sys.stdin.read()
        if not raw or not raw.strip():
            return {}
        return json.loads(raw)
    except (json.JSONDecodeError, UnicodeDecodeError, OSError):
        return {}


def _normalize(event: dict) -> dict:
    """Convert platform-specific event JSON to a normalized save payload."""
    content = json.dumps(event, ensure_ascii=False, default=str)
    content = content[:MAX_CONTENT_LENGTH]

    session_id = (
        event.get("session_id")
        or event.get("sessionId")
        or event.get("session", {}).get("id")
        or ""
    )
    event_type = (
        event.get("event")
        or event.get("type")
        or event.get("hook_event_name")
        or event.get("event_type")
        or "unknown"
    )
    tool_name = (
        event.get("tool_name")
        or event.get("toolName")
        or event.get("tool", {}).get("name")
        or ""
    )

    tags = ["hook", f"event:{event_type}"]
    if tool_name:
        tags.append(f"tool:{tool_name}")

    metadata = {
        "source": "cell_mem_session_hook",
        "event_type": event_type,
    }
    if tool_name:
        metadata["tool_name"] = tool_name

    return {
        "content": content,
        "memory_type": "episodic",
        "tags": tags,
        "session_id": session_id,
        "metadata": metadata,
    }


# ---------------------------------------------------------------------------
# File delivery (primary — works inside sandbox)
# ---------------------------------------------------------------------------


def _write_file(payload: dict) -> bool:
    """Append payload as one JSON line to the ingest queue file.

    This is the primary delivery mechanism. It works in sandboxed
    environments where network access is blocked (Codex CLI).
    cell-mem's IngestServer monitors this file and processes new lines.
    """
    try:
        queue_path = Path(INGEST_FILE)
        queue_path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(payload, ensure_ascii=False) + "\n"
        with open(queue_path, "a", encoding="utf-8") as f:
            f.write(line)
        return True
    except (OSError, IOError):
        return False


# ---------------------------------------------------------------------------
# HTTP delivery (fallback — for non-sandbox environments)
# ---------------------------------------------------------------------------


def _post_http(payload: dict) -> bool:
    """POST payload to ingest endpoint with retries."""
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")

    for attempt in range(MAX_RETRIES):
        try:
            req = urlrequest.Request(
                INGEST_URL,
                data=data,
                headers={
                    "Content-Type": "application/json",
                    "User-Agent": "cell-mem-hook/1.0",
                },
            )
            resp = urlrequest.urlopen(req, timeout=HTTP_TIMEOUT)
            if 200 <= resp.status < 300:
                return True
        except (URLError, OSError, ConnectionRefusedError):
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAY)

    return False


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    """Entry point. Always returns 0 (never block the agent)."""
    try:
        event = _read_stdin()
        if not event:
            print("OK (no stdin data)")
            return 0

        payload = _normalize(event)

        # File delivery first (sandbox-safe, no network needed)
        if _write_file(payload):
            print("OK")
        elif _post_http(payload):
            print("OK")
        else:
            print("OK (delivery unavailable)")
    except Exception:
        traceback.print_exc(file=sys.stderr)
        print("OK (hook error)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
