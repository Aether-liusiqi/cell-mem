"""Falsifiable condition evaluator — manual memory_verify trigger.

Evaluates structured falsifiable conditions stored in semantic memory entries.
Each condition is a dict describing when the knowledge becomes invalid:

    {"field": "runtime_version", "operator": "<", "value": "18"}
    {"field": "dependency_installed", "operator": "contains", "value": "next.js"}
    {"field": "file_pattern", "operator": "matches", "value": "*.tsx"}

The evaluator checks these against an environment snapshot dict provided
by the caller (the MCP tool or the agent).

Supported operators: <, <=, >, >=, ==, !=, contains, not_contains, matches
"""

from __future__ import annotations

import fnmatch
import logging
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

_OPERATORS = {
    "<", "<=", ">", ">=", "==", "!=",
    "contains", "not_contains", "matches",
}

# Safety limits for environment snapshots (prevents OOM from oversized dicts)
_MAX_ENVIRONMENT_KEYS = 200
_MAX_ENVIRONMENT_VALUE_LENGTH = 10_000
_MAX_ENVIRONMENT_DEPTH = 3


def _parse_version(v: Any) -> Optional[Tuple[int, ...]]:
    """Try to parse a value as a semver-like tuple for comparison.

    Returns a tuple of ints if parseable, None otherwise.
    """
    try:
        parts = str(v).split(".")
        return tuple(int(p) for p in parts)
    except (ValueError, TypeError):
        return None


def _compare_versions(actual: Any, expected: Any, operator: str) -> bool:
    """Compare two values with version-aware logic.

    If both values are version-like (e.g., "18.2.1"), uses tuple comparison.
    Otherwise falls back to string comparison.
    """
    actual_str = str(actual)
    expected_str = str(expected)

    # Try version tuple comparison first
    actual_ver = _parse_version(actual)
    expected_ver = _parse_version(expected)

    if actual_ver is not None and expected_ver is not None:
        # Pad to same length for comparison
        max_len = max(len(actual_ver), len(expected_ver))
        a_padded = actual_ver + (0,) * (max_len - len(actual_ver))
        e_padded = expected_ver + (0,) * (max_len - len(expected_ver))

        if operator == "<":
            return a_padded < e_padded
        elif operator == "<=":
            return a_padded <= e_padded
        elif operator == ">":
            return a_padded > e_padded
        elif operator == ">=":
            return a_padded >= e_padded

    # Fall back to string comparison for == and !=, and for non-version values
    if operator == "==":
        return actual_str == expected_str
    elif operator == "!=":
        return actual_str != expected_str
    elif operator == "<":
        return actual_str < expected_str
    elif operator == "<=":
        return actual_str <= expected_str
    elif operator == ">":
        return actual_str > expected_str
    elif operator == ">=":
        return actual_str >= expected_str

    return False


class ConditionEvaluator:
    """Evaluates falsifiable conditions stored in semantic memory.

    Supports environment snapshot capture and auto-verification
    on environment changes.

    Usage:
        evaluator = ConditionEvaluator(sqlite_store=store)
        condition = {"field": "react_version", "operator": "<", "value": "18"}
        environment = {"react_version": "17"}
        is_met, reason = evaluator.evaluate(condition, environment)
        # → (True, "react_version 17 < 18")
    """

    def __init__(
        self,
        sqlite_store: Optional["SqliteStore"] = None,  # noqa: F821
    ):
        self._store = sqlite_store

    def evaluate(
        self, condition: dict, environment: dict
    ) -> Tuple[bool, str]:
        """Evaluate a single condition against the environment.

        Args:
            condition: Dict with "field", "operator", "value" keys.
            environment: Dict of current environment values (size-limited).

        Returns:
            (is_fulfilled, reason_string).
            is_fulfilled = True means the condition is triggered
            (the knowledge may be stale/expired).
        """
        # Safety: reject oversized environment dicts
        if len(environment) > _MAX_ENVIRONMENT_KEYS:
            return False, (
                f"Environment dict exceeds max {_MAX_ENVIRONMENT_KEYS} keys "
                f"({len(environment)} provided)"
            )

        field = condition.get("field", "")
        operator = condition.get("operator", "==")
        expected = condition.get("value")

        if not field:
            return False, "No field specified in condition"

        if operator not in _OPERATORS:
            return False, f"Unknown operator: {operator}"

        if field not in environment:
            return False, f"Field '{field}' not found in environment"

        actual = environment[field]

        try:
            if operator in ("<", "<=", ">", ">=", "==", "!="):
                is_met = _compare_versions(actual, expected, operator)
            elif operator == "contains":
                is_met = str(expected) in str(actual)
            elif operator == "not_contains":
                is_met = str(expected) not in str(actual)
            elif operator == "matches":
                is_met = fnmatch.fnmatch(str(actual), str(expected))
            else:
                return False, f"Unhandled operator: {operator}"

            reason = f"{field} {operator} {expected}: actual={actual} → triggered={is_met}"
            return is_met, reason

        except Exception as exc:
            return False, f"Evaluation error: {exc}"

    def verify_entry(
        self,
        entry: "MemoryObject",  # noqa: F821
        environment: dict,
    ) -> Dict[str, Any]:
        """Check whether a semantic memory entry's falsifiable condition is met.

        Args:
            entry: A MemoryObject with semantic memory data.
            environment: Dict of current environment values.

        Returns:
            {"verified": bool, "condition_met": bool, "reason": str,
             "action": "none"|"expire", "entry_id": str}
        """
        fc = entry.falsifiable_condition

        if fc is None:
            return {
                "verified": True,
                "condition_met": False,
                "reason": "No falsifiable condition on this entry",
                "action": "none",
                "entry_id": entry.id,
            }

        if not isinstance(fc, dict):
            return {
                "verified": True,
                "condition_met": False,
                "reason": "Falsifiable condition is not a dict",
                "action": "none",
                "entry_id": entry.id,
            }

        is_met, reason = self.evaluate(fc, environment)
        action = "expire" if is_met else "none"

        return {
            "verified": True,
            "condition_met": is_met,
            "reason": reason,
            "action": action,
            "entry_id": entry.id,
            "condition": fc,
        }

    def verify_all(
        self,
        semantic: "SemanticMemory",  # noqa: F821
        environment: dict,
    ) -> Dict[str, Any]:
        """Check all semantic entries that have falsifiable conditions.

        Args:
            semantic: SemanticMemory instance.
            environment: Dict of current environment values.

        Returns:
            {"status": "ok", "data": {"total_checked": N, "triggered": [...],
             "still_valid": N, "no_condition": N}}
        """
        from cell_mem.memory.semantic import SemanticMemory

        if not environment:
            return {
                "status": "error",
                "error": "No environment provided for condition verification",
            }

        # Load all semantic entries with falsifiable conditions
        conn = semantic._store.get_connection()
        rows = conn.execute(
            "SELECT * FROM semantic_memory WHERE falsifiable_condition IS NOT NULL"
        ).fetchall()

        from cell_mem.models import MemoryObject

        triggered = []
        still_valid = 0

        for row in rows:
            entry = MemoryObject.from_row(dict(row))
            result = self.verify_entry(entry, environment)
            if result["condition_met"]:
                triggered.append(result)
            else:
                still_valid += 1

        logger.info(
            "Verified %d entries: %d triggered, %d still valid",
            len(rows), len(triggered), still_valid,
        )

        return {
            "status": "ok",
            "data": {
                "total_checked": len(rows),
                "triggered": triggered,
                "still_valid": still_valid,
                "no_condition": semantic.count() - len(rows),
            },
        }

    # ------------------------------------------------------------------
    # Environment Snapshots + Auto-Verify
    # ------------------------------------------------------------------

    def capture_snapshot(self, environment: dict) -> int:
        """Save current environment state as a snapshot. Returns snapshot ID."""
        if self._store is None:
            raise ValueError("No SqliteStore configured for snapshots")

        import json
        from datetime import datetime, timezone

        now = datetime.now(timezone.utc).isoformat()
        snap_json = json.dumps(environment, ensure_ascii=False)

        cursor = self._store.execute(
            "INSERT INTO environment_snapshots (snapshot_json, captured_at) VALUES (?, ?)",
            (snap_json, now),
        )
        self._store.commit()
        snap_id = cursor.lastrowid
        logger.debug("Environment snapshot #%d captured", snap_id)
        return snap_id

    def get_latest_snapshot(self) -> Optional[dict]:
        """Return the most recent snapshot dict, or None."""
        if self._store is None:
            return None

        import json

        row = self._store.fetchone(
            "SELECT snapshot_json FROM environment_snapshots ORDER BY id DESC LIMIT 1"
        )
        if row is None:
            return None
        try:
            return json.loads(row["snapshot_json"])
        except (json.JSONDecodeError, TypeError):
            return None

    def detect_changes(self, current: dict) -> dict:
        """Compare current environment with latest snapshot.

        Returns:
            {"changed_fields": [...], "added_fields": [...],
             "removed_fields": [...], "has_changes": bool}
        """
        prev = self.get_latest_snapshot()
        if prev is None:
            return {
                "changed_fields": list(current.keys()),
                "added_fields": [],
                "removed_fields": [],
                "has_changes": True,
            }

        changed = [k for k in current if k in prev and str(current[k]) != str(prev[k])]
        added = [k for k in current if k not in prev]
        removed = [k for k in prev if k not in current]

        return {
            "changed_fields": changed,
            "added_fields": added,
            "removed_fields": removed,
            "has_changes": bool(changed or added or removed),
        }

    def auto_verify(
        self,
        semantic: "SemanticMemory",  # noqa: F821
        current_env: dict,
    ) -> dict:
        """Full auto-falsifiable pipeline.

        1. Detect changes from last snapshot.
        2. If no changes, return early.
        3. Filter semantic entries whose falsifiable_condition field overlaps.
        4. Run verify on filtered subset.
        5. Capture new snapshot after verification.
        """
        changes = self.detect_changes(current_env)
        if not changes["has_changes"]:
            return {
                "status": "ok",
                "data": {
                    "triggered": False,
                    "note": "No environment changes detected",
                    "changes": changes,
                },
            }

        # Build set of changed field names
        changed_fields = set(changes["changed_fields"] + changes["added_fields"] + changes["removed_fields"])

        # Filter semantic entries whose condition field overlaps
        conn = semantic._store.get_connection()
        rows = conn.execute(
            "SELECT * FROM semantic_memory WHERE falsifiable_condition IS NOT NULL"
        ).fetchall()

        from cell_mem.models import MemoryObject
        import json

        triggered = []
        for row in rows:
            entry = MemoryObject.from_row(dict(row))
            fc = entry.falsifiable_condition
            if isinstance(fc, dict) and fc.get("field", "") in changed_fields:
                result = self.verify_entry(entry, current_env)
                if result["condition_met"]:
                    # Expire the entry
                    from datetime import datetime, timezone
                    now = datetime.now(timezone.utc).isoformat()
                    conn.execute(
                        "UPDATE semantic_memory SET invalidated_at = ? WHERE id = ?",
                        (now, entry.id),
                    )
                    conn.commit()
                    triggered.append(result)

        # Capture new snapshot
        self.capture_snapshot(current_env)

        logger.info(
            "Auto-verify: %d changes, %d entries checked, %d triggered",
            len(changed_fields), len(rows), len(triggered),
        )

        return {
            "status": "ok",
            "data": {
                "triggered": True,
                "changes_detected": changes,
                "entries_checked": len(rows),
                "triggered_entries": triggered,
            },
        }
