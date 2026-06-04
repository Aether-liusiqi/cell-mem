"""Phase 2 MCP tool registrations.

Phase 2a: memory_associate is live.
Phase 2b: memory_forget and memory_consolidate are live.
All three are registered separately by their respective helper functions.
"""

from __future__ import annotations

import logging
from typing import Any, Dict

logger = logging.getLogger(__name__)


def register_phase2b_tools(mcp, memory_system: "MemorySystem") -> None:  # noqa: F821
    """Register memory_forget and memory_consolidate tools (Phase 2b live)."""

    @mcp.tool(
        name="memory_forget",
        description=(
            "Forget or expire a memory. Set expire=True to soft-delete "
            "(mark invalidated_at) instead of hard-deleting. Useful for "
            "removing incorrect or outdated memories from episodic and "
            "semantic layers."
        ),
    )
    def memory_forget(
        memory_id: str, reason: str = "", expire: bool = False
    ) -> Dict[str, Any]:
        return memory_system.forget(memory_id, reason, expire)

    @mcp.tool(
        name="memory_consolidate",
        description=(
            "Manually trigger a memory consolidation cycle. Scores all "
            "episodic memories (five-dimension model), identifies forget "
            "candidates below dynamic threshold, archives to cold storage, "
            "and runs DBSCAN pattern detection to auto-create semantic entries. "
            "Returns statistics about the consolidation cycle."
        ),
    )
    def memory_consolidate() -> Dict[str, Any]:
        return memory_system.consolidate()


def register_phase3_tools(mcp, memory_system: "MemorySystem") -> None:  # noqa: F821
    """Register memory_verify and memory_reflect tools (Phase 3)."""
    from cell_mem.tools.verify import register_verify_tool
    from cell_mem.tools.reflect import register_reflect_tool

    register_verify_tool(mcp, memory_system)
    register_reflect_tool(mcp, memory_system)

    import logging
    logging.getLogger(__name__).info("Phase 3 MCP tools: memory_verify, memory_reflect")


def register_phase4_tools(mcp, memory_system: "MemorySystem") -> None:  # noqa: F821
    """Register Phase 4 tools: memory_replay, creative pool, environment check."""
    from cell_mem.tools.replay import register_replay_tool, register_creative_pool_tools

    register_replay_tool(mcp, memory_system)
    register_creative_pool_tools(mcp, memory_system)

    import logging
    logging.getLogger(__name__).info(
        "Phase 4 MCP tools: memory_replay, memory_hypothesis_feedback, "
        "memory_creative_pool, memory_check_environment"
    )


def register_associate_tool(mcp, memory_system: "MemorySystem") -> None:  # noqa: F821
    """Register the real memory_associate tool (Phase 2a)."""

    @mcp.tool(
        name="memory_associate",
        description=(
            "Create a directed association edge between two memory items. "
            "Edge weights range from -1 (inhibitory/contradiction) to 1 (strong association). "
            "Relation types: associated_with, causes, contradicts, is_a, part_of. "
            "These edges enable graph-based memory retrieval via the 'two_pass' strategy."
        ),
    )
    def memory_associate(
        source_id: str,
        target_id: str,
        weight: float = 1.0,
        relation_type: str = "associated_with",
    ) -> Dict[str, Any]:
        return memory_system.associate(source_id, target_id, weight, relation_type)
