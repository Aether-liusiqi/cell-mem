"""SessionProcessor — background daemon that bridges session JSONL to episodic memory.

Follows the same daemon-thread pattern as EmbeddingWorker:
  - Polling loop (configurable interval)
  - Processes new lines from session JSONL files
  - Chunks TURN events into ~500-char semantic segments
  - Saves to episodic memory (embedding=NULL, EmbeddingWorker handles vectors)
  - Tracks per-file progress in SQLite meta table (crash-safe resume)
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

# Chunk size in characters — balances recall precision with storage efficiency
CHUNK_SIZE = 500
# Poll interval when no new content (seconds)
POLL_INTERVAL = 2.0
# Max content length per chunk (32KB guard — same as MemorySystem.save)
MAX_CONTENT_LENGTH = 32000
# Meta key prefix for storing per-file byte offsets
OFFSET_META_PREFIX = "session_offset:"


class SessionProcessor:
    """Background daemon: scans session JSONL → chunks → episodic memory.

    Usage:
        processor = SessionProcessor(memory_system, "/path/to/sessions")
        processor.start()
        # ... server runs ...
        processor.stop()
    """

    def __init__(
        self,
        memory_system: "MemorySystem",  # noqa: F821
        sessions_dir: str,
        poll_interval: float = POLL_INTERVAL,
    ):
        from cell_mem.memory_system import MemorySystem

        self._ms: MemorySystem = memory_system
        self._sessions_dir = Path(sessions_dir)
        self._poll_interval = poll_interval
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._processed_chunks = 0
        self._processed_files = 0

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Start background processing in a daemon thread."""
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(
            target=self._run, daemon=True, name="cell-mem-session-processor"
        )
        self._thread.start()
        logger.info("SessionProcessor started (poll=%.1fs, chunk=%d chars)",
                     self._poll_interval, CHUNK_SIZE)

    def stop(self) -> None:
        """Signal the processor to stop. Thread exits at next idle check."""
        self._running = False
        logger.info("SessionProcessor stopping (files=%d, chunks=%d)",
                     self._processed_files, self._processed_chunks)

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def _run(self) -> None:
        """Main loop: discover session files → process new content → sleep → repeat."""
        # Wait for embedding model to be ready before first processing
        try:
            self._ms.embed_model.ensure_loaded()
            logger.info("SessionProcessor: embedding model ready")
        except Exception as exc:
            logger.warning("SessionProcessor: embedding model not ready: %s", exc)
            # Continue anyway — save() works without embedding (NULL embed)

        while self._running:
            try:
                files = self._discover_session_files()
                for file_path in files:
                    if not self._running:
                        break
                    try:
                        n = self._process_file(file_path)
                        if n > 0:
                            self._processed_chunks += n
                            self._processed_files += 1
                    except Exception as exc:
                        logger.debug("Error processing %s: %s", file_path.name, exc)
            except Exception as exc:
                logger.debug("SessionProcessor scan error: %s", exc)

            time.sleep(self._poll_interval)

        logger.info("SessionProcessor exited (files=%d, chunks=%d)",
                     self._processed_files, self._processed_chunks)

    # ------------------------------------------------------------------
    # File discovery
    # ------------------------------------------------------------------

    def _discover_session_files(self) -> List[Path]:
        """Find all .jsonl session files in the sessions directory tree.

        Skips empty dirs and non-.jsonl files. Excludes the _current_session pointer.
        """
        if not self._sessions_dir.exists():
            return []

        files = []
        for jsonl_file in self._sessions_dir.rglob("*.jsonl"):
            if jsonl_file.name.startswith("_"):
                continue  # Skip _current_session, _metadata, etc.
            if not self._is_fully_processed(jsonl_file):
                files.append(jsonl_file)

        # Sort by modification time — process oldest first
        try:
            files.sort(key=lambda f: f.stat().st_mtime)
        except OSError:
            pass

        return files

    # ------------------------------------------------------------------
    # Per-file processing
    # ------------------------------------------------------------------

    def _process_file(self, file_path: Path) -> int:
        """Process new lines from a session JSONL file.

        Returns the number of chunks saved to episodic memory.
        """
        # Get the last-processed byte offset
        offset_key = self._offset_key(file_path)
        last_offset = self._get_offset(offset_key)

        # Check if file has new content
        try:
            file_size = file_path.stat().st_size
        except OSError:
            return 0

        if file_size <= last_offset:
            return 0

        # Read new lines
        new_lines = []
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                f.seek(last_offset)
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    new_lines.append(line)
                new_offset = f.tell()
        except OSError as exc:
            logger.debug("Cannot read %s: %s", file_path.name, exc)
            return 0

        if not new_lines:
            return 0

        # Parse and process each line
        chunks_saved = 0
        session_id = self._extract_session_id(file_path)

        for line in new_lines:
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue

            event_type = event.get("type", "")
            if event_type != "turn":
                continue  # Only TURN events contain retrievable content

            chunks = self._chunk_turn(event)
            for chunk in chunks:
                if self._save_chunk(chunk):
                    chunks_saved += 1

        # Update offset only after successful processing
        self._set_offset(offset_key, new_offset)
        if chunks_saved > 0:
            logger.debug("Processed %s: %d new lines → %d chunks",
                         file_path.name, len(new_lines), chunks_saved)

        return chunks_saved

    def _extract_session_id(self, file_path: Path) -> str:
        """Extract session_id from filename: <session_id>.jsonl"""
        return file_path.stem

    # ------------------------------------------------------------------
    # Chunking
    # ------------------------------------------------------------------

    def _chunk_turn(self, turn: dict) -> List[dict]:
        """Split a TURN event into 1-N retrievable chunks.

        Each chunk is a dict ready for ms.save():
          {content, memory_type, options: {tags, session_id, metadata}}

        Strategy:
        - Short turns (< CHUNK_SIZE total) → 1 chunk
        - Long turns → split by paragraph breaks when possible, then by char count
        - Preserve user_message + assistant_message + tool context together
        """
        session_id = turn.get("session_id", "")
        turn_id = turn.get("turn_id", "")
        model = turn.get("model", "")
        tool_calls = turn.get("tool_calls", [])

        # Build full turn text
        parts = []
        user_msg = turn.get("user_message", "").strip()
        asst_msg = turn.get("assistant_message", "").strip()

        if user_msg:
            parts.append(f"User: {user_msg}")
        if asst_msg:
            parts.append(f"Assistant: {asst_msg}")
        if tool_calls:
            tool_names = [t.get("name", "?") for t in tool_calls if isinstance(t, dict)]
            if tool_names:
                parts.append(f"Tools used: {', '.join(tool_names)}")

        full_text = "\n".join(parts)

        # If short enough, return as single chunk
        if len(full_text) <= CHUNK_SIZE:
            return [self._make_chunk_dict(
                content=full_text,
                session_id=session_id,
                turn_id=turn_id,
                chunk_index=0,
                total_chunks=1,
                model=model,
            )]

        # Long text: split into chunks
        chunks = []
        # Try paragraph split first
        paragraphs = full_text.split("\n\n")
        current = ""
        chunk_idx = 0

        for para in paragraphs:
            if len(current) + len(para) + 2 <= CHUNK_SIZE:
                current = f"{current}\n\n{para}" if current else para
            else:
                if current:
                    chunks.append(current)
                    chunk_idx += 1
                # If single paragraph exceeds chunk size, force-split
                if len(para) > CHUNK_SIZE:
                    for i in range(0, len(para), CHUNK_SIZE):
                        chunks.append(para[i:i + CHUNK_SIZE])
                        chunk_idx += 1
                    current = ""
                else:
                    current = para

        if current:
            chunks.append(current)
            chunk_idx += 1

        total = len(chunks)
        return [
            self._make_chunk_dict(
                content=chunk_text,
                session_id=session_id,
                turn_id=turn_id,
                chunk_index=i,
                total_chunks=total,
                model=model,
            )
            for i, chunk_text in enumerate(chunks)
        ]

    @staticmethod
    def _make_chunk_dict(
        content: str,
        session_id: str,
        turn_id: str,
        chunk_index: int,
        total_chunks: int,
        model: str = "",
    ) -> dict:
        """Build a save-ready chunk dict."""
        return {
            "content": content[:MAX_CONTENT_LENGTH],
            "memory_type": "episodic",
            "options": {
                "tags": ["session", "chunk"],
                "session_id": session_id,
                "metadata": {
                    "source": "session_processor",
                    "turn_id": turn_id,
                    "chunk_index": chunk_index,
                    "total_chunks": total_chunks,
                    "model": model,
                    "processed_at": datetime.now(timezone.utc).isoformat(),
                },
            },
        }

    # ------------------------------------------------------------------
    # Save to episodic memory
    # ------------------------------------------------------------------

    def _save_chunk(self, chunk: dict) -> bool:
        """Save a single chunk to episodic memory via MemorySystem.save()."""
        try:
            result = self._ms.save(
                content=chunk["content"],
                memory_type=chunk["memory_type"],
                options=chunk["options"],
            )
            return result.get("status") == "ok"
        except Exception as exc:
            logger.debug("SessionProcessor save failed: %s", exc)
            return False

    # ------------------------------------------------------------------
    # Offset tracking (SQLite meta table, crash-safe resume)
    # ------------------------------------------------------------------

    def _offset_key(self, file_path: Path) -> str:
        """Generate a stable meta key for a session file."""
        # Use relative path from sessions dir for portability
        try:
            rel = file_path.relative_to(self._sessions_dir)
            return f"{OFFSET_META_PREFIX}{rel.as_posix()}"
        except ValueError:
            return f"{OFFSET_META_PREFIX}{file_path.as_posix()}"

    def _get_offset(self, key: str) -> int:
        """Read the last-processed byte offset from meta table."""
        try:
            blob = self._ms.store.get_meta(key)
            if blob:
                return int(blob.decode("utf-8"))
        except Exception:
            pass
        return 0

    def _set_offset(self, key: str, offset: int) -> None:
        """Persist the processed byte offset to meta table."""
        try:
            self._ms.store.set_meta(key, str(offset).encode("utf-8"))
        except Exception:
            pass  # Non-fatal — worst case, we re-process a few lines on restart

    def _is_fully_processed(self, file_path: Path) -> bool:
        """Check if a file has been fully processed."""
        try:
            file_size = file_path.stat().st_size
            offset = self._get_offset(self._offset_key(file_path))
            return offset >= file_size
        except OSError:
            return True  # Skip files we can't read

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    @property
    def processed_chunks(self) -> int:
        return self._processed_chunks

    @property
    def processed_files(self) -> int:
        return self._processed_files

    def pending_files(self) -> int:
        """Number of session files with unprocessed content."""
        try:
            files = self._discover_session_files()
            return len(files)
        except Exception:
            return -1
