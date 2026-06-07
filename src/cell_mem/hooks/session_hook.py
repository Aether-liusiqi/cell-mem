#!/usr/bin/env python3
"""Cell-mem session recording hook script.

This script is installed into the platform's hook directory by
``cell-mem --hooks install`` and invoked by the agent platform
(Codex CLI or Claude Code) on SessionStart and PostToolUse events.

It receives event JSON on stdin, normalizes it, and HTTP-POSTs
to cell-mem's ingest endpoint for asynchronous storage.

**Zero cell-mem dependency** — uses only Python stdlib. This ensures
the script works in any subprocess environment the platform spawns.

**Never blocks** — any failure after retry exhaustion is silently swallowed.
A lost log entry is acceptable; blocking the agent is not.

**Retry logic** — the ingest endpoint may not be ready yet when the hook
fires (MCP subprocess scheduling takes ~3s). We retry up to 5 times with
a 0.8s delay between attempts to cover this gap.
"""

from __future__ import annotations

import json
import os
import sys
import time
import traceback
from urllib import request as urlrequest
from urllib.error import URLError

# ---------------------------------------------------------------------------
# Configuration (overridden by environment)
# ---------------------------------------------------------------------------

INGEST_PORT = os.environ.get("CELL_MEM_INGEST_PORT", "8766")
INGEST_URL = f"http://127.0.0.1:{INGEST_PORT}/ingest"
HTTP_TIMEOUT = 2  # seconds per attempt
MAX_CONTENT_LENGTH = 10000  # safety cap on event JSON size
MAX_RETRIES = 5       # total attempts before giving up
RETRY_DELAY = 0.8     # seconds between retries


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
    """Convert platform-specific event JSON to a normalized save payload.

    Strategy: save the entire event JSON as content (zero information loss)
    while extracting known fields as structured metadata where possible.
    """
    # Serialize full event as content
    content = json.dumps(event, ensure_ascii=False, default=str)
    content = content[:MAX_CONTENT_LENGTH]

    # Try to extract structured fields for better searchability
    session_id = (
        event.get("session_id")
        or event.get("sessionId")
        or event.get("session", {}).get("id")
        or ""
    )
    event_type = (
        event.get("event")
        or event.get("type")
        or event.get("event_type")
        or "unknown"
    )
    tool_name = (
        event.get("tool_name")
        or event.get("toolName")
        or event.get("tool", {}).get("name")
        or ""
    )

    # Build tags for filtering
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
# HTTP delivery
# ---------------------------------------------------------------------------


def _post(payload: dict) -> bool:
    """POST payload to ingest endpoint with retries.

    Retries cover the MCP subprocess scheduling gap: the hook may fire
    before cell-mem's ingest endpoint has started listening (3s delay).
    """
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
        success = _post(payload)

        if success:
            print("OK")
        else:
            # Silent failure after retries — ingest may genuinely be down
            print("OK (ingest unavailable after retries)")
    except Exception:
        # Last-resort catch: NEVER let an unhandled exception reach the platform
        traceback.print_exc(file=sys.stderr)
        print("OK (hook error)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
