"""Ingest endpoint for hook scripts — HTTP server.

Runs as a daemon thread alongside the MCP server. Receives events from
hook scripts via HTTP POST and writes them to session JSONL files through
the SessionRecorder.

SessionProcessor (background daemon) handles the JSONL→episodic bridge
asynchronously — ingest is now a fast, no-processing passthrough.

Only the HTTP channel remains. The file-watcher channel was removed in
v2.1 — hook scripts now write directly to session JSONL files, eliminating
the queue-file intermediary.
"""

from __future__ import annotations

import json
import logging
from http.server import HTTPServer, BaseHTTPRequestHandler
from threading import Thread
from typing import Optional

logger = logging.getLogger(__name__)


class IngestServer:
    """HTTP ingest endpoint — receives events from hook scripts.

    Usage:
        ingest = IngestServer(session_recorder, port=8766)
        ingest.start()   # daemon thread, non-blocking
    """

    def __init__(
        self,
        session_recorder: "SessionRecorder",  # noqa: F821
        port: int = 8766,
    ):
        from cell_mem.session.recorder import SessionRecorder

        self._recorder: SessionRecorder = session_recorder
        self._port = port
        self._httpd: Optional[HTTPServer] = None
        self._http_thread: Optional[Thread] = None

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
        """Start HTTP server in a daemon thread. Returns True on success."""
        recorder = self._recorder  # capture for handler closure

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
                    if self.server.recorder.write_event(body):
                        self._json({"ok": True})
                    else:
                        self._json({"error": "no active session"}, 500)
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
        self._httpd.recorder = recorder  # type: ignore[attr-defined]

        self._http_thread = Thread(target=self._httpd.serve_forever, daemon=True)
        self._http_thread.start()
        logger.info("Ingest HTTP ready on http://127.0.0.1:%d/ingest", self._port)
        return True

    def shutdown(self) -> None:
        """Shut down the HTTP server."""
        if self._httpd is not None:
            self._httpd.shutdown()
            self._httpd.server_close()
            self._httpd = None
            logger.debug("Ingest HTTP server stopped")
