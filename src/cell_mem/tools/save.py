"""memory_save MCP tool — write memories to the Cell-mem system."""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


def register_save_tool(mcp, memory_system: "MemorySystem") -> None:  # noqa: F821
    """Register memory_save on a FastMCP server instance."""

    @mcp.tool(
        name="memory_save",
        description=(
            "Write a memory to the Cell-mem brain-inspired memory system. "
            "Memories are routed to working, episodic, or semantic layers "
            "based on memory_type. Semantic memories should have higher confidence "
            "(0.5+) for facts and knowledge; episodic for interaction records; "
            "working for transient task context."
        ),
    )
    def memory_save(
        content: str,
        memory_type: str = "episodic",
        options: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        return memory_system.save(content, memory_type, options)
