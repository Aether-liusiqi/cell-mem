"""memory_verify MCP tool — Phase 3: manual falsifiable condition verification.

Checks semantic knowledge entries against current environment values.
If a falsifiable condition is met, the entry is marked as expired.
"""

from __future__ import annotations

from typing import Any, Dict


def register_verify_tool(mcp, memory_system: "MemorySystem") -> None:  # noqa: F821
    """Register the memory_verify MCP tool."""

    @mcp.tool(
        name="memory_verify",
        description=(
            "Verify falsifiable conditions for semantic knowledge entries. "
            "Provide an entry_id to check a single entry, or omit to verify all "
            "entries with falsifiable conditions. "
            "Provide an environment dict describing current project state, e.g.: "
            '{"react_version": "18.2", "os": "linux", "dependencies": "react,next.js"}. '
            "Supported operators: <, <=, >, >=, ==, !=, contains, not_contains, matches. "
            "Entries whose conditions are met will be marked as expired (invalidated_at set)."
        ),
    )
    def memory_verify(
        entry_id: str | None = None,
        environment: dict | None = None,
    ) -> Dict[str, Any]:
        if entry_id:
            return memory_system.verify(entry_id, environment)
        else:
            return memory_system.verify_all(environment or {})
