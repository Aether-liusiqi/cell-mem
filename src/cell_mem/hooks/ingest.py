"""Ingest endpoint for hook scripts — HTTP + file-based.

Runs in daemon threads alongside the MCP server. Hook scripts deliver
events via two channels:

1. **JSONL file** (primary, sandbox-safe): hook appends a JSON line to
   ``~/.cell_mem/ingest_queue.jsonl``. A file-watcher thread reads new
   lines and saves them. Works inside Codex's sandbox (no network needed).

2. **HTTP POST** (fallback): hook POSTs to ``/ingest`` on 127.0.0.1.
   For non-sandbox environments (Claude Code).

Both channels write to episodic memory immediately (embedding=NULL,
filled later by the EmbeddingWorker).

Zero new dependencies — uses stdlib http.server only.
"""

from __future__ import annotations

import json
import logging
import os
import time
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path
from threading import Thread, Lock
from typing import Optional

logger = logging.getLogger(__name__)

# Default path for the shared JSONL queue file
DEFAULT_QUEUE_FILE = str(Path.home() / ".cell_mem" / "ingest_queue.jsonl")
# How often the file watcher polls for new lines (seconds)
WATCH_INTERVAL = 1.0
# Maximum queue file size before truncation (~1 MB)
MAX_QUEUE_BYTES = 1_000_000


class IngestServer:
    """Ingest server with HTTP endpoint + JSONL file watcher.

    Usage:
        ingest = IngestServer(memory_system, port=8766)
        ingest.start()   # daemon threads, non-blocking
    """

    def __init__(
        self,
        memory_system: "MemorySystem",  # noqa: F821
        port: int = 8766,
        queue_file: str | None = None,
    ):
        from cell_mem.memory_system import MemorySystem

        self._ms: MemorySystem = memory_system
        self._port = port
        self._queue_file = queue_file or DEFAULT_QUEUE_FILE
        self._httpd: Optional[HTTPServer] = None
        self._http_thread: Optional[Thread] = None
        self._watch_thread: Optional[Thread] = None
        self._watch_offset = 0  # bytes read so far from the queue file

    @property
    def port(self) -> int:
        return self._port

    @property
    def running(self) -> bool:
        return self._httpd is not None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self) -> bool:
        """Start HTTP server + file watcher in daemon threads.

        Returns:
            True if at least one ingest channel started successfully.
        """
        http_ok = self._start_http()
        self._start_file_watcher()
        return http_ok or True  # file watcher always succeeds

    def shutdown(self) -> None:
        """Shut down all ingest channels."""
        if self._httpd is not None:
            self._httpd.shutdown()
            self._httpd.server_close()
            self._httpd = None
            logger.debug("Ingest HTTP server stopped")

    # ------------------------------------------------------------------
    # Shared save logic
    # ------------------------------------------------------------------

    def _ingest(self, payload: dict) -> bool:
        """Save a normalized payload to episodic memory. Returns True on success."""
        try:
            content = payload.get("content", "")
            if not content:
                return False

            tags = payload.get("tags", [])
            if not isinstance(tags, list):
                tags = []
            if "hook" not in tags:
                tags = list(tags) + ["hook"]

            result = self._ms.save(
                content=str(content)[:10000],
                memory_type=payload.get("memory_type", "episodic"),
                options={
                    "tags": tags,
                    "session_id": payload.get("session_id"),
                    "metadata": payload.get("metadata"),
                },
            )
            return result.get("status") == "ok"
        except Exception:
            logger.debug("Ingest save failed", exc_info=True)
            return False

    # ------------------------------------------------------------------
    # HTTP endpoint
    # ------------------------------------------------------------------

    def _start_http(self) -> bool:
        """Start the HTTP ingest server. Returns True if port bound successfully."""
        ingest = self._ingest  # capture method

        class _Handler(BaseHTTPRequestHandler):
            def log_message(self, format, *args):
                pass

            def do_GET(self) -> None:
                if self.path == "/health":
                    self._json({"status": "ok", "ingest_port": self.server.ingest_port})
                else:
                    self.send_error(404)

            def do_POST(self) -> None:
                if self.path != "/ingest":
                    self.send_error(404)
                    return
                try:
                    length = int(self.headers.get("Content-Length", "0"))
                    if length == 0:
                        self._json({"error": "empty body"}, 400)
                        return
                    body = json.loads(self.rfile.read(length))
                    if ingest(body):
                        self._json({"ok": True})
                    else:
                        self._json({"error": "save failed"}, 500)
                except json.JSONDecodeError:
                    self._json({"error": "invalid json"}, 400)
                except Exception as exc:
                    logger.debug("HTTP ingest error: %s", exc)
                    self._json({"error": str(exc)}, 500)

            def _json(self, data: dict, status: int = 200) -> None:
                payload = json.dumps(data, ensure_ascii=False).encode("utf-8")
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

        try:
            self._httpd = HTTPServer(("127.0.0.1", self._port), _Handler)
        except OSError:
            logger.debug("Ingest port %d unavailable", self._port)
            return False

        self._httpd.ingest_port = self._port
        self._httpd._ms = self._ms  # type: ignore[attr-defined]

        self._http_thread = Thread(target=self._httpd.serve_forever, daemon=True)
        self._http_thread.start()
        logger.info("Ingest HTTP ready on http://127.0.0.1:%d/ingest", self._port)
        return True

    # ------------------------------------------------------------------
    # JSONL file watcher (sandbox-safe primary channel)
    # ------------------------------------------------------------------

    def _start_file_watcher(self) -> None:
        """Start a daemon thread that polls the JSONL queue file for new lines."""
        self._watch_thread = Thread(target=self._watch_loop, daemon=True)
        self._watch_thread.start()
        logger.info("Ingest file watcher ready on %s", self._queue_file)

    def _watch_loop(self) -> None:
        """Continuously poll the queue file for new JSONL lines."""
        queue_path = Path(self._queue_file)

        while True:
            try:
                if not queue_path.exists():
                    time.sleep(WATCH_INTERVAL)
                    continue

                file_size = queue_path.stat().st_size
                if file_size > self._watch_offset:
                    with open(queue_path, "r", encoding="utf-8") as f:
                        f.seek(self._watch_offset)
                        for line in f:
                            line = line.strip()
                            if not line:
                                continue
                            try:
                                payload = json.loads(line)
                                self._ingest(payload)
                            except json.JSONDecodeError:
                                logger.debug("Skipping invalid JSONL line")
                        self._watch_offset = f.tell()

                # Truncate if the file grows too large
                if file_size > MAX_QUEUE_BYTES:
                    self._truncate_queue(queue_path)

            except Exception:
                logger.debug("File watcher error", exc_info=True)

            time.sleep(WATCH_INTERVAL)

    def _truncate_queue(self, queue_path: Path) -> None:
        """Truncate the queue file to prevent unbounded growth."""
        try:
            # Keep last 100KB to avoid losing recent entries
            keep_bytes = 100_000
            file_size = queue_path.stat().st_size
            if file_size <= keep_bytes:
                return
            with open(queue_path, "rb") as f:
                f.seek(file_size - keep_bytes)
                tail = f.read()
            with open(queue_path, "wb") as f:
                f.write(tail)
            self._watch_offset = len(tail)
            logger.debug("Queue file truncated (was %d bytes)", file_size)
        except OSError:
            pass
