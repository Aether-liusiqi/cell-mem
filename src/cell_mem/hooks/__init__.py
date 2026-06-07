"""Cell-mem hook system — automatic session recording for agent platforms.

Provides:
- ``IngestServer``: lightweight HTTP endpoint for hook scripts
- ``install()`` / ``uninstall()``: register/remove hooks in Codex/Claude configs
- ``detect_platforms()``: discover installed agent platforms
"""

from cell_mem.hooks.ingest import IngestServer
from cell_mem.hooks.registrar import detect_platforms, install, uninstall

__all__ = [
    "IngestServer",
    "detect_platforms",
    "install",
    "uninstall",
]
