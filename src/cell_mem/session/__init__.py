"""Session recording and processing — JSONL-based session storage with async chunk→embed pipeline.

SessionRecorder: manages session JSONL files (create, append, flush).
SessionProcessor: background daemon that bridges JSONL to episodic memory.
"""

from cell_mem.session.recorder import SessionRecorder
from cell_mem.session.processor import SessionProcessor

__all__ = ["SessionRecorder", "SessionProcessor"]
