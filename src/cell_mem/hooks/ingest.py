"""Lightweight HTTP ingest endpoint for hook scripts.

Runs in a daemon thread alongside the MCP server. Hook scripts POST events here;
they are saved to episodic memory immediately (embedding=NULL, filled later by
the EmbeddingWorker).

Zero new dependencies — uses stdlib http.server only.
"""

from __future__ import annotations

import json
import logging
from http.server import HTTPServer, BaseHTTPRequestHandler
from threading import Thread
from typing import Optional

logger = logging.getLogger(__name__)


class IngestServer:
    """Minimal HTTP server that accepts POST /ingest and GET /health.

    Bind is restricted to 127.0.0.1 — this is an internal channel between
    hook scripts and the cell-mem server, never exposed to the network.

    Usage:
        ingest = IngestServer(memory_system, port=8766)
        ingest.start()   # daemon thread, non-blocking
    """

    def __init__(
        self,
        memory_system: "MemorySystem",  # noqa: F821
        port: int = 8766,
    ):
        from cell_mem.memory_system import MemorySystem

        self._ms: MemorySystem = memory_system
        self._port = port
        self._httpd: Optional[HTTPServer] = None
        self._thread: Optional[Thread] = None

    @property
    def port(self) -> int:
        return self._port

    @property
    def running(self) -> bool:
        return self._httpd is not None

    def start(self) -> bool:
        """Start the ingest server in a daemon thread.

        Returns:
            True if the server started successfully, False if the port was
            already in use (another cell-mem instance is running).
        """
        ms = self._ms  # capture for handler

        class _Handler(BaseHTTPRequestHandler):
            # Suppress per-request log lines
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
                    content = body.get("content", "")
                    if not content:
                        self._json({"error": "missing content"}, 400)
                        return

                    tags = body.get("tags", [])
                    if not isinstance(tags, list):
                        tags = []
                    # Auto-tag as hook event (avoid duplicates)
                    if "hook" not in tags:
                        tags = list(tags) + ["hook"]

                    result = self.server._ms.save(
                        content=str(content)[:10000],  # safety cap
                        memory_type=body.get("memory_type", "episodic"),
                        options={
                            "tags": tags,
                            "session_id": body.get("session_id"),
                            "metadata": body.get("metadata"),
                        },
                    )
                    if result.get("status") == "ok":
                        obj_id = result["data"]["id"]
                        self._json({"ok": True, "id": obj_id})
                    else:
                        self._json({"error": result.get("error", "save failed")}, 500)
                except json.JSONDecodeError:
                    self._json({"error": "invalid json"}, 400)
                except Exception as exc:
                    logger.debug("Ingest save failed: %s", exc)
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
            logger.debug("Ingest port %d unavailable (in use by another process)", self._port)
            return False

        self._httpd.ingest_port = self._port
        self._httpd._ms = ms  # type: ignore[attr-defined]

        self._thread = Thread(target=self._httpd.serve_forever, daemon=True)
        self._thread.start()

        logger.info("Ingest endpoint ready on http://127.0.0.1:%d/ingest", self._port)
        return True

    def shutdown(self) -> None:
        """Shut down the ingest server."""
        if self._httpd is not None:
            self._httpd.shutdown()
            self._httpd.server_close()
            self._httpd = None
            logger.debug("Ingest server stopped")
