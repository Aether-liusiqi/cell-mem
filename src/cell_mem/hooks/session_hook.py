#!/usr/bin/env python3
"""Cell-mem session recording hook script.

Installed to the platform hook directory by ``cell-mem --hooks install``
and invoked by Codex CLI / Claude Code on SessionStart and PostToolUse.

**Primary delivery: direct JSONL write** — appends one normalized event
to the session JSONL file (``{SESSIONS_DIR}/<encoded-cwd>/<sessionId>.jsonl``).
No queue file, no network — crash-safe, zero-latency recording.

**Fallback: HTTP POST** — if file write fails (sandbox restriction),
falls back to HTTP POST to the cell-mem ingest endpoint.

**Session ID management** — reads/writes ``{SESSIONS_DIR}/_current_session``
pointer file. Creates a new session if none exists or stale (>24h).

Zero cell_mem dependency — pure Python stdlib.
Never blocks the agent — all errors → exit 0.
"""

from __future__ import annotations

import json
import os
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from urllib import request as urlrequest
from urllib.error import URLError

# ---------------------------------------------------------------------------
# Configuration (overridden by environment)
# ---------------------------------------------------------------------------

SESSIONS_DIR = Path(
    os.environ.get(
        "CELL_MEM_SESSIONS_DIR",
        str(Path.home() / ".cell_mem" / "sessions"),
    )
).expanduser()

INGEST_PORT = os.environ.get("CELL_MEM_INGEST_PORT", "8766")
INGEST_URL = f"http://127.0.0.1:{INGEST_PORT}/ingest"

POINTER_FILE = SESSIONS_DIR / "_current_session"
SESSION_MAX_AGE_SEC = 86400  # 24 hours
MAX_CONTENT_LENGTH = 10000
HTTP_TIMEOUT = 2
MAX_RETRIES = 3
RETRY_DELAY = 0.5


# ---------------------------------------------------------------------------
# Session ID management
# ---------------------------------------------------------------------------


def _get_or_create_session_id(cwd: str) -> str:
    """Get existing session ID from pointer file, or create a new one.

    Creates a new session if:
    - Pointer file doesn't exist
    - Pointer is older than 24 hours
    - CWD has changed (different project)
    """
    if POINTER_FILE.exists():
        try:
            data = json.loads(POINTER_FILE.read_text(encoding="utf-8"))
            sid = data.get("session_id", "")
            ptr_cwd = data.get("cwd", "")
            updated_at = data.get("updated_at", "")

            if sid and updated_at:
                try:
                    updated = datetime.fromisoformat(updated_at)
                    age = (datetime.now(timezone.utc) - updated).total_seconds()
                except ValueError:
                    age = 0

                if age < SESSION_MAX_AGE_SEC and (not cwd or cwd == ptr_cwd):
                    return sid
        except (json.JSONDecodeError, OSError):
            pass

    # Create new session
    sid = str(uuid.uuid4())
    POINTER_FILE.parent.mkdir(parents=True, exist_ok=True)
    _write_pointer(sid, cwd)
    return sid


def _write_pointer(session_id: str, cwd: str) -> None:
    """Persist session_id → pointer file."""
    try:
        POINTER_FILE.write_text(json.dumps({
            "session_id": session_id,
            "cwd": cwd,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }, ensure_ascii=False), encoding="utf-8")
    except OSError:
        pass


# ---------------------------------------------------------------------------
# Event normalization
# ---------------------------------------------------------------------------


def _read_stdin() -> dict:
    """Read event JSON from stdin. Returns {} on any failure."""
    if sys.stdin.isatty():
        return {}
    try:
        raw = sys.stdin.read()
        if not raw or not raw.strip():
            return {}
        return json.loads(raw)
    except (json.JSONDecodeError, UnicodeDecodeError, OSError):
        return {}


def _normalize(event: dict, session_id: str) -> dict | None:
    """Convert platform event → cell-mem canonical event.

    Returns a dict with at least {"type": "...", "session_id": "..."}
    or None if the event has no extractable content.
    """
    # Detect platform
    if "hook_event_name" in event:
        return _normalize_codex(event, session_id)
    elif "message" in event:
        return _normalize_claude(event, session_id)
    else:
        return _normalize_generic(event, session_id)


def _normalize_codex(event: dict, session_id: str) -> dict | None:
    """Codex CLI hook event → cell-mem turn."""
    event_name = event.get("hook_event_name") or event.get("type") or "unknown"

    if event_name == "SessionStart":
        return {
            "type": "session_start",
            "session_id": session_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "cwd": event.get("cwd", ""),
            "platform": "codex",
            "agent_model": event.get("model", ""),
        }

    # PostToolUse and everything else → turn
    tool_name = event.get("tool_name") or event.get("toolName") or ""
    tool_input = event.get("tool_input") or event.get("toolInput") or {}

    parts = []
    if tool_name:
        parts.append(f"[Tool: {tool_name}]")
        try:
            parts.append(json.dumps(tool_input, ensure_ascii=False)[:500])
        except Exception:
            parts.append(str(tool_input)[:500])

    # Include tool result if present
    tool_result = event.get("tool_result") or event.get("result") or ""
    if tool_result:
        parts.append(str(tool_result)[:2000])

    # Include model/usage if present
    model = event.get("model", "")
    usage = event.get("usage")

    return {
        "type": "turn",
        "session_id": session_id,
        "turn_id": str(uuid.uuid4()),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "user_message": "",
        "assistant_message": "\n".join(parts),
        "tool_calls": [{"name": tool_name, "input": tool_input}] if tool_name else [],
        "model": model,
        "stop_reason": event.get("stop_reason", ""),
        "usage": usage or {},
    }


def _normalize_claude(event: dict, session_id: str) -> dict | None:
    """Claude Code hook event → cell-mem turn."""
    message = event.get("message", {})
    content = message.get("content", "")

    if isinstance(content, list):
        text_parts = []
        tool_calls = []
        for block in content:
            if isinstance(block, dict):
                if block.get("type") == "text":
                    text_parts.append(str(block.get("text", "")))
                elif block.get("type") == "tool_use":
                    tool_calls.append({
                        "name": block.get("name", ""),
                        "input": block.get("input", {}),
                    })
        assistant_message = "\n".join(text_parts)
    else:
        assistant_message = str(content)
        tool_calls = []

    role = message.get("role", "")
    model = message.get("model", "")
    usage = message.get("usage", {})

    return {
        "type": "turn",
        "session_id": session_id,
        "turn_id": str(uuid.uuid4()),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "user_message": assistant_message[:5000] if role == "user" else "",
        "assistant_message": assistant_message[:5000] if role == "assistant" else "",
        "tool_calls": tool_calls,
        "model": model,
        "stop_reason": message.get("stop_reason", ""),
        "usage": usage,
    }


def _normalize_generic(event: dict, session_id: str) -> dict:
    """Best-effort normalization for unknown event formats."""
    return {
        "type": "turn",
        "session_id": session_id,
        "turn_id": str(uuid.uuid4()),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "assistant_message": json.dumps(event, ensure_ascii=False, default=str)[:5000],
        "stop_reason": "unknown_event",
    }


# ---------------------------------------------------------------------------
# File delivery (primary — direct JSONL write)
# ---------------------------------------------------------------------------


def _write_session_file(record: dict, cwd: str) -> bool:
    """Append one normalized event as a JSON line to the session file.

    Returns True on success, False if file write failed.
    """
    try:
        # Encode cwd for path-safe directory name.
        # First normalize: Codex may double-escape backslashes during event
        # serialization (JSON → string → JSON), producing "C:\\Users" in the
        # parsed string instead of "C:\Users". Normalize to single backslash.
        cwd_norm = cwd.replace("\\\\", "\\")
        encoded = cwd_norm.replace(":", "--").replace("\\", "-").replace("/", "-")
        session_id = record.get("session_id", "unknown")

        session_file = SESSIONS_DIR / encoded / f"{session_id}.jsonl"
        session_file.parent.mkdir(parents=True, exist_ok=True)

        line = json.dumps(record, ensure_ascii=False, default=str) + "\n"
        with open(session_file, "a", encoding="utf-8") as f:
            f.write(line)
        return True
    except (OSError, IOError):
        return False


# ---------------------------------------------------------------------------
# HTTP delivery (fallback — for restricted environments)
# ---------------------------------------------------------------------------


def _post_http(record: dict) -> bool:
    """POST normalized event to ingest endpoint with retries."""
    data = json.dumps(record, ensure_ascii=False).encode("utf-8")

    for attempt in range(MAX_RETRIES):
        try:
            req = urlrequest.Request(
                INGEST_URL,
                data=data,
                headers={
                    "Content-Type": "application/json",
                    "User-Agent": "cell-mem-hook/2.0",
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
    """Entry point. Always returns 0 — never block the agent."""
    try:
        event = _read_stdin()
        if not event:
            print("OK (no stdin)")
            return 0

        cwd = event.get("cwd") or os.getcwd()
        session_id = _get_or_create_session_id(cwd)
        record = _normalize(event, session_id)

        if record is None:
            print("OK (no content)")
            return 0

        # File delivery first (sandbox-safe, no network needed)
        if _write_session_file(record, cwd):
            print("OK")
        elif _post_http(record):
            print("OK")
        else:
            print("OK (delivery unavailable)")
    except Exception:
        import traceback
        traceback.print_exc(file=sys.stderr)
        print("OK (hook error)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
