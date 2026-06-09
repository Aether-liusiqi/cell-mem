"""SessionRecorder — JSONL-based session file management.

Manages session JSONL files in path-encoded project directories.
Pure file I/O — no dependency on MemorySystem or any cell_mem internals.
Thread-safe, crash-safe (append + immediate flush).

Design borrowed from Claude Code's session recording:
  - One JSONL file per session: <base>/<encoded-cwd>/<sessionId>.jsonl
  - Path encoding: C:/Users/Xiaochong → C--Users-Xiaochong
  - Append-only, flush after every write
  - Session ID managed via a pointer file ({base}/_current_session)
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Default sessions directory — can be overridden by CELL_MEM_SESSIONS_DIR env var.
# For Codex sandbox (Windows), set this to a D: drive path to avoid C: drive writes.
_DEFAULT_SESSIONS_DIR = str(Path.home() / ".cell_mem" / "sessions")

# Pointer file name (lives inside the sessions base directory)
_POINTER_FILENAME = "_current_session"

# Session expiry: auto-create new session if pointer file is older than this
_SESSION_MAX_AGE_SEC = 86400  # 24 hours


class SessionRecorder:
    """Manages session JSONL file creation and append-only writing.

    Thread-safe. Provides two modes:
    - **Server mode**: ``start_session()`` + ``write_event()`` — for use inside
      the cell-mem MCP server (ingest endpoint, memory_save tool).
    - **Hook mode**: ``ensure_session()`` + ``write_event()`` — for use in
      the standalone hook script (reads pointer file, zero server dependency).

    Usage (server):
        recorder = SessionRecorder()
        sid = recorder.start_session(cwd="/path/to/project", platform="codex")
        recorder.write_event({"type": "turn", ...})

    Usage (hook script):
        recorder = SessionRecorder()
        sid = recorder.ensure_session(cwd=os.getcwd())
        recorder.write_event({"type": "turn", ...})
    """

    def __init__(self, base_dir: str | None = None):
        """
        Args:
            base_dir: Root directory for all session files.
                      Priority: parameter > CELL_MEM_SESSIONS_DIR env > ~/.cell_mem/sessions
        """
        self._base_dir = Path(
            base_dir
            or os.environ.get("CELL_MEM_SESSIONS_DIR")
            or _DEFAULT_SESSIONS_DIR
        ).expanduser().resolve()
        self._base_dir.mkdir(parents=True, exist_ok=True)

        self._session_id: Optional[str] = None
        self._file: Optional[object] = None  # TextIO wrapper
        self._lock = threading.Lock()
        self._cwd: str = ""
        logger.debug("SessionRecorder base_dir=%s", self._base_dir)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def sessions_dir(self) -> str:
        return str(self._base_dir)

    @property
    def session_id(self) -> Optional[str]:
        return self._session_id

    @property
    def pointer_file(self) -> Path:
        """Path to the _current_session pointer file (inside base_dir)."""
        return self._base_dir / _POINTER_FILENAME

    # ------------------------------------------------------------------
    # Server mode: explicit session lifecycle
    # ------------------------------------------------------------------

    def start_session(
        self,
        cwd: str,
        platform: str = "unknown",
        agent_model: str = "",
    ) -> str:
        """Create a new session, write session_start event, return session_id.

        Also updates the pointer file so hook scripts can discover the session.

        Args:
            cwd: Working directory (encoded for path-safe directory name).
            platform: "codex" | "claude" | "manual"
            agent_model: Model name if known (e.g., "deepseek-v4-flash").
        """
        with self._lock:
            self._session_id = str(uuid.uuid4())
            self._cwd = cwd
            self._open_file()

            from cell_mem.session.schema import make_session_start

            self._write_line(make_session_start(
                session_id=self._session_id,
                cwd=cwd,
                platform=platform,
                agent_model=agent_model,
            ))
            self._update_pointer()
            logger.info("Session started: %s (cwd=%s, platform=%s)",
                        self._session_id[:8], cwd, platform)
            return self._session_id

    # ------------------------------------------------------------------
    # Hook mode: discover or create session from pointer file
    # ------------------------------------------------------------------

    def ensure_session(self, cwd: str) -> str:
        """Get or create the current session ID.

        For hook scripts: reads the pointer file to find the active session.
        Creates a new session if:
        - Pointer file doesn't exist
        - Pointer file is older than 24 hours (stale session)
        - Session JSONL file doesn't exist (orphaned pointer)

        Returns the session_id (existing or newly created).
        """
        with self._lock:
            # Try to reuse existing session from pointer file
            if self._try_reuse_session(cwd):
                assert self._session_id is not None
                return self._session_id

            # Create new session
            self._session_id = str(uuid.uuid4())
            self._cwd = cwd
            self._open_file()

            from cell_mem.session.schema import make_session_start

            self._write_line(make_session_start(
                session_id=self._session_id,
                cwd=cwd,
                platform="auto",
            ))
            self._update_pointer()
            logger.debug("New session (hook mode): %s", self._session_id[:8])
            return self._session_id

    def _try_reuse_session(self, cwd: str) -> bool:
        """Try to reuse the session pointed to by the pointer file.

        Returns True if a valid existing session was found and opened.
        """
        ptr = self.pointer_file
        if not ptr.exists():
            return False

        try:
            data = json.loads(ptr.read_text(encoding="utf-8"))
            sid = data.get("session_id", "")
            ptr_cwd = data.get("cwd", "")
            updated_at = data.get("updated_at", "")
        except (json.JSONDecodeError, OSError):
            return False

        if not sid:
            return False

        # Check staleness
        if updated_at:
            try:
                updated = datetime.fromisoformat(updated_at)
                age = (datetime.now(timezone.utc) - updated).total_seconds()
                if age > _SESSION_MAX_AGE_SEC:
                    logger.debug("Session %s expired (age=%.0fh)", sid[:8], age / 3600)
                    return False
            except ValueError:
                pass

        # Only reuse if CWD matches (don't mix projects)
        if cwd and ptr_cwd and cwd != ptr_cwd:
            logger.debug("CWD changed (%s → %s), creating new session",
                        ptr_cwd, cwd)
            return False

        # Verify the session file exists (pointer could be orphaned)
        session_file = self._session_path(sid)
        if not session_file.exists():
            logger.debug("Session file missing for %s, creating new", sid[:8])
            return False

        self._session_id = sid
        self._cwd = ptr_cwd or cwd
        self._open_file()
        logger.debug("Reusing session: %s", sid[:8])
        return True

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def write_event(self, event: dict) -> bool:
        """Append one event as a JSON line to the current session file.

        Thread-safe. Immediately flushes to disk. Returns False if no
        active session.

        Args:
            event: Dict with at least "type" key. Serialized as JSON.
        """
        if not event:
            return False

        with self._lock:
            if self._session_id is None:
                logger.warning("write_event called with no active session")
                return False
            self._write_line(event)
            # Update pointer timestamp so hook scripts can detect activity
            self._update_pointer()
            return True

    def _write_line(self, data: dict) -> None:
        """Write one JSON line + flush. Must hold self._lock."""
        line = json.dumps(data, ensure_ascii=False, default=str) + "\n"
        if self._file is None:
            self._open_file()
        self._file.write(line)
        self._file.flush()

    # ------------------------------------------------------------------
    # File management
    # ------------------------------------------------------------------

    def _open_file(self) -> None:
        """Open (or reopen) the current session JSONL file for appending."""
        if self._session_id is None:
            raise RuntimeError("Cannot open file: no session_id")

        session_file = self._session_path(self._session_id)
        session_file.parent.mkdir(parents=True, exist_ok=True)

        import io
        # Use buffered text I/O with manual flush control
        self._file = open(session_file, "a", encoding="utf-8")

    def _session_path(self, session_id: str) -> Path:
        """<base>/<encoded-cwd>/<session_id>.jsonl"""
        return self._base_dir / self._encode_path(self._cwd) / f"{session_id}.jsonl"

    def _update_pointer(self) -> None:
        """Write {session_id, cwd, updated_at} to the pointer file."""
        if self._session_id is None:
            return
        try:
            self.pointer_file.write_text(json.dumps({
                "session_id": self._session_id,
                "cwd": self._cwd,
                "updated_at": datetime.now(timezone.utc).isoformat(),
            }, ensure_ascii=False), encoding="utf-8")
        except OSError:
            pass  # Non-fatal — pointer file is best-effort

    # ------------------------------------------------------------------
    # Static helpers
    # ------------------------------------------------------------------

    @staticmethod
    def encode_path(path: str) -> str:
        """Encode a filesystem path for use as a directory name.

        C:\\Users\\Xiaochong → C--Users-Xiaochong
        d:/test → d--test
        """
        return path.replace(":", "--").replace("\\", "-").replace("/", "-")

    _encode_path = encode_path  # instance alias

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Close the current session file."""
        with self._lock:
            if self._file is not None:
                self._file.close()
                self._file = None
                logger.debug("Session file closed")

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
