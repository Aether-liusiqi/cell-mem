"""memory_status MCP tool — health and statistics."""

from __future__ import annotations

import logging
from typing import Any, Dict

logger = logging.getLogger(__name__)


def register_status_tool(mcp, memory_system: "MemorySystem") -> None:  # noqa: F821
    """Register memory_status on a FastMCP server instance."""

    @mcp.tool(
        name="memory_status",
        description=(
            "Get health and statistics for the Cell-mem memory system. "
            "Returns layer counts, vector index info, consolidation status, "
            "and overall health indicator."
        ),
    )
    def memory_status() -> Dict[str, Any]:
        return memory_system.status()
