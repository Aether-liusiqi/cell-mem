"""memory_recall MCP tool — retrieve memories from the Cell-mem system."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def register_recall_tool(mcp, memory_system: "MemorySystem") -> None:  # noqa: F821
    """Register memory_recall on a FastMCP server instance."""

    @mcp.tool(
        name="memory_recall",
        description=(
            "Retrieve relevant memories from the Cell-mem system. "
            "Searches across working, episodic, and semantic layers using "
            "vector similarity. Supports direct vector search and "
            "adds keyword search and association graph traversal. "
            "Set options.memory_type to 'episodic', 'semantic', or omit for both."
        ),
    )
    def memory_recall(
        query: str,
        options: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        return memory_system.recall(query, options)
