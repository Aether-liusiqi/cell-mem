"""MCP tools for user preference pipeline — fully automatic after deployment."""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


def register_preference_tools(mcp, memory_system: "MemorySystem") -> None:  # noqa: F821
    """Register all preference-related MCP tools on a FastMCP server instance."""

    @mcp.tool(
        name="memory_extract_preferences",
        description=(
            "Extract user preferences from recent interaction history. "
            "Normally runs automatically during consolidation cycles. "
            "Use this to force extraction on demand."
        ),
    )
    def memory_extract_preferences(
        limit: int = 50,
    ) -> Dict[str, Any]:
        return memory_system.extract_preferences(limit=limit)

    @mcp.tool(
        name="memory_get_preferences",
        description=(
            "Query detected user preferences. Returns preferences ranked by "
            "confidence, optionally filtered by type or context relevance."
        ),
    )
    def memory_get_preferences(
        context_text: Optional[str] = None,
        min_confidence: float = 0.3,
        preference_type: Optional[str] = None,
    ) -> Dict[str, Any]:
        return memory_system.get_preferences(
            context_text=context_text,
            min_confidence=min_confidence,
            preference_type=preference_type,
        )

    @mcp.tool(
        name="memory_check_preference_conflicts",
        description=(
            "Detect contradictory user preferences. For example, "
            "'prefers short responses' vs 'likes detailed explanations' — "
            "these may indicate context-dependent preferences."
        ),
    )
    def memory_check_preference_conflicts() -> Dict[str, Any]:
        return memory_system.check_preference_conflicts()

    @mcp.tool(
        name="memory_inject_preference",
        description=(
            "Manually add a user preference. Use for explicit user statements "
            "like 'I always prefer X over Y'. The preference enters the pipeline "
            "and will be auto-verified through usage."
        ),
    )
    def memory_inject_preference(
        preference_text: str,
        preference_type: str = "general",
        confidence: float = 0.7,
        trigger_context: Optional[str] = None,
    ) -> Dict[str, Any]:
        return memory_system.inject_preference(
            preference_text=preference_text,
            preference_type=preference_type,
            confidence=confidence,
            trigger_context=trigger_context,
        )

    @mcp.tool(
        name="memory_record_preference_feedback",
        description=(
            "Record feedback on whether a detected preference was accurate. "
            "Confirmed preferences strengthen; rejected ones weaken. "
            "Normally auto-triggered when preference-derived procedural templates are used."
        ),
    )
    def memory_record_preference_feedback(
        preference_id: str,
        confirmed: bool,
    ) -> Dict[str, Any]:
        return memory_system.record_preference_feedback(
            preference_id=preference_id,
            confirmed=confirmed,
        )

    logger.info("Preference MCP tools registered: 5 tools")
