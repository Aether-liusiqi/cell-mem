"""Unified event schema for cell-mem session recording.

Defines 4 event types (not 11 — we only keep what matters for recall)
and normalization functions that convert Codex/Claude platform events
into cell-mem's canonical format.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any, Dict, Optional

# ---------------------------------------------------------------------------
# Event type constants
# ---------------------------------------------------------------------------

SESSION_START = "session_start"
SESSION_END = "session_end"
TURN = "turn"
CONTEXT = "context"

ALL_TYPES = {SESSION_START, SESSION_END, TURN, CONTEXT}

# ---------------------------------------------------------------------------
# Canonical event builders
# ---------------------------------------------------------------------------


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _uid() -> str:
    return str(uuid.uuid4())


def make_session_start(
    session_id: str,
    cwd: str = "",
    platform: str = "unknown",
    agent_model: str = "",
) -> dict:
    return {
        "type": SESSION_START,
        "session_id": session_id,
        "timestamp": _now(),
        "cwd": cwd,
        "platform": platform,
        "agent_model": agent_model,
    }


def make_session_end(session_id: str, duration_sec: float = 0.0) -> dict:
    return {
        "type": SESSION_END,
        "session_id": session_id,
        "timestamp": _now(),
        "duration_sec": duration_sec,
    }


def make_turn(
    session_id: str,
    user_message: str = "",
    assistant_message: str = "",
    tool_calls: list | None = None,
    model: str = "",
    stop_reason: str = "",
    usage: dict | None = None,
) -> dict:
    return {
        "type": TURN,
        "session_id": session_id,
        "turn_id": _uid(),
        "timestamp": _now(),
        "user_message": user_message,
        "assistant_message": assistant_message,
        "tool_calls": tool_calls or [],
        "model": model,
        "stop_reason": stop_reason,
        "usage": usage or {},
    }


def make_context(
    session_id: str,
    context_type: str,
    content: str,
) -> dict:
    return {
        "type": CONTEXT,
        "session_id": session_id,
        "timestamp": _now(),
        "context_type": context_type,
        "content": content,
    }


# ---------------------------------------------------------------------------
# Platform normalizers
# ---------------------------------------------------------------------------


def normalize_codex_event(raw: dict, session_id: str) -> Optional[dict]:
    """Convert a Codex CLI hook event to a cell-mem canonical event.

    Codex sends events via hooks.json with these fields:
      - hook_event_name: "SessionStart" | "PostToolUse"
      - session_id: UUID string
      - cwd: working directory
      - For PostToolUse: tool_name, tool_input, tool_result (optional)
      - For PostToolUse with agent model: model, usage (optional)
    """
    event_type = raw.get("hook_event_name") or raw.get("type") or ""

    if event_type == "SessionStart":
        return make_session_start(
            session_id=session_id,
            cwd=raw.get("cwd", ""),
            platform="codex",
            agent_model=raw.get("model", ""),
        )

    if event_type == "PostToolUse":
        tool_name = raw.get("tool_name") or raw.get("toolName") or ""
        tool_input = raw.get("tool_input") or raw.get("toolInput") or {}
        tool_result = raw.get("tool_result") or raw.get("result") or ""

        # Build assistant_message from tool interaction context
        parts = []
        if tool_name:
            parts.append(f"[Tool: {tool_name}]")
            try:
                import json
                parts.append(json.dumps(tool_input, ensure_ascii=False)[:500])
            except Exception:
                parts.append(str(tool_input)[:500])
        if tool_result:
            parts.append(str(tool_result)[:2000])

        return make_turn(
            session_id=session_id,
            user_message="",  # Codex doesn't pass user message in hook
            assistant_message="\n".join(parts),
            tool_calls=[{"name": tool_name, "input": tool_input}] if tool_name else [],
            model=raw.get("model", ""),
            stop_reason=raw.get("stop_reason", ""),
            usage=raw.get("usage"),
        )

    # Unknown event type — store as a generic turn for traceability
    return make_turn(
        session_id=session_id,
        assistant_message=str(raw)[:5000],
        stop_reason="unknown_event",
    )


def normalize_claude_event(raw: dict, session_id: str) -> Optional[dict]:
    """Convert a Claude Code hook event to a cell-mem canonical event.

    Claude Code hook events arrive via settings.json hooks with fields
    that vary by event. We extract what's available.
    """
    event_type = raw.get("type") or raw.get("event") or ""

    if event_type in ("session_start", "SessionStart"):
        return make_session_start(
            session_id=session_id,
            cwd=raw.get("cwd", ""),
            platform="claude",
            agent_model=raw.get("model", ""),
        )

    # For Claude Code, the primary content is in assistant/user messages
    message = raw.get("message", {})
    content = message.get("content", "")

    # Content can be a string or a list of content blocks
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

    if role == "user":
        return make_turn(
            session_id=session_id,
            user_message=assistant_message[:5000],
            assistant_message="",
        )
    elif role == "assistant":
        return make_turn(
            session_id=session_id,
            assistant_message=assistant_message[:5000],
            tool_calls=tool_calls,
            model=model,
            stop_reason=message.get("stop_reason", ""),
            usage=usage,
        )

    # Fallback: treat as generic event
    return make_turn(
        session_id=session_id,
        assistant_message=str(raw)[:5000],
    )


def normalize_event(raw: dict, session_id: str, platform: str = "auto") -> Optional[dict]:
    """Auto-detect platform and normalize. Safe to call with any input."""
    if not raw or not isinstance(raw, dict):
        return None

    if platform == "auto":
        # Heuristic: Codex events have hook_event_name, Claude events have message.content
        if "hook_event_name" in raw:
            platform = "codex"
        elif "message" in raw:
            platform = "claude"
        else:
            platform = "unknown"

    if platform == "codex":
        return normalize_codex_event(raw, session_id)
    elif platform == "claude":
        return normalize_claude_event(raw, session_id)
    else:
        # Best-effort: dump as turn
        return make_turn(
            session_id=session_id,
            assistant_message=str(raw)[:5000],
        )
