"""MCP tools for generative replay, creative pool feedback, and environment check."""

from __future__ import annotations

import logging
from typing import Any, Dict

logger = logging.getLogger(__name__)


def register_replay_tool(mcp, memory_system: "MemorySystem") -> None:  # noqa: F821
    """Register memory_replay tool."""

    @mcp.tool(
        name="memory_replay",
        description=(
            "Trigger a generative replay cycle. Uses a 5-phase algorithm to "
            "generate creative hypotheses by: (1) biased seed sampling from "
            "semantic memory, (2) weak-edge random walks through the association "
            "graph, (3) cross-domain concept pairing with LLM hypothesis generation, "
            "(4) four-layer noise filtering, and (5) storing hypotheses in the "
            "creative pool with confidence 0.1-0.3. Optional theme_text biases "
            "seed selection toward a topic."
        ),
    )
    def memory_replay(
        theme_text: str | None = None,
    ) -> Dict[str, Any]:
        return memory_system.replay(theme_text)


def register_creative_pool_tools(mcp, memory_system: "MemorySystem") -> None:  # noqa: F821
    """Register creative pool feedback and query tools."""

    @mcp.tool(
        name="memory_hypothesis_feedback",
        description=(
            "Record feedback on a creative hypothesis from the generative replay "
            "engine. Set confirmed=True to boost confidence by +0.3. Hypotheses "
            "reaching confidence >= 0.6 are automatically promoted to semantic "
            "memory. Set confirmed=False to increment the ignore counter; after "
            "3 ignores the topic is suppressed from future generation."
        ),
    )
    def memory_hypothesis_feedback(
        hypothesis_id: str,
        confirmed: bool,
    ) -> Dict[str, Any]:
        return memory_system.record_hypothesis_feedback(hypothesis_id, confirmed)

    @mcp.tool(
        name="memory_creative_pool",
        description=(
            "Query the creative pool of generated hypotheses. Optional status "
            "filter: pending (default), confirmed, rejected, promoted."
        ),
    )
    def memory_creative_pool(
        status: str | None = None,
    ) -> Dict[str, Any]:
        return memory_system.creative_pool_status(status)

    @mcp.tool(
        name="memory_check_environment",
        description=(
            "Compare current environment state against the last snapshot, detect "
            "changes, and auto-verify affected semantic entries whose falsifiable "
            "conditions overlap with changed fields. Entries with met conditions "
            "are automatically expired. Provide environment as dict, e.g.: "
            '{"react_version": "18.2", "os": "linux", "dependencies": "react,next.js"}.'
        ),
    )
    def memory_check_environment(
        environment: dict,
    ) -> Dict[str, Any]:
        return memory_system.check_environment(environment)
