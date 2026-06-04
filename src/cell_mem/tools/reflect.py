"""memory_reflect MCP tool — manual self-reflection trigger.

Triggers failure attribution analysis. Searches relevant episodic memories
and uses LLM to identify root causes of failures.
"""

from __future__ import annotations

from typing import Any, Dict


def register_reflect_tool(mcp, memory_system: "MemorySystem") -> None:  # noqa: F821
    """Register the memory_reflect MCP tool."""

    @mcp.tool(
        name="memory_reflect",
        description=(
            "Trigger self-reflection with one or all four dimensions. "
            "Dimension 1 (failure): effect attribution analysis — why did it fail? "
            "Dimension 2 (strategy): evaluate procedural template effectiveness. "
            "Dimension 3 (gaps): detect missing or retrievable knowledge. "
            "Dimension 4 (process): update templates, adjust weights, create meta-knowledge. "
            "Use dimensions='failure' for simple attribution; "
            "dimensions='all' for full 4-dimension reflection."
        ),
    )
    def memory_reflect(
        task_description: str,
        outcome: str = "failure",
        dimensions: str = "failure",
    ) -> Dict[str, Any]:
        return memory_system.reflect(task_description, outcome, dimensions)
