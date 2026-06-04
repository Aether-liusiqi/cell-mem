"""Unified data models for Cell-mem memory system.

MemoryObject is the canonical representation across all four memory layers.
Fields are intentionally permissive — validation is done at the layer level.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class MemoryType(str, Enum):
    WORKING = "working"
    EPISODIC = "episodic"
    SEMANTIC = "semantic"
    PROCEDURAL = "procedural"


class LifecycleStage(str, Enum):
    PLASTIC = "plastic"
    SEMI_STABLE = "semi_stable"
    LOCKED = "locked"


class EdgeType(str, Enum):
    """Association edge types for the memory graph (Kuzu-compatible)."""
    ASSOCIATED_WITH = "associated_with"
    CAUSES = "causes"
    CONTRADICTS = "contradicts"
    IS_A = "is_a"
    PART_OF = "part_of"


class NodeType(str, Enum):
    """Node types for the memory graph (Kuzu-compatible)."""
    MEMORY_CONCEPT = "memory_concept"
    ENTITY = "entity"
    SESSION = "session"
    PROJECT = "project"


class MemoryObject(BaseModel):
    """Unified memory object used across all four memory layers.

    Not all fields apply to every memory_type. Layer-level code
    is responsible for validating required-fields-by-type.
    """

    id: str = Field(default_factory=lambda: uuid.uuid4().hex[:16])
    content: str
    memory_type: MemoryType = MemoryType.EPISODIC

    # --- Working memory fields ---
    attention_score: float = 1.0
    base_priority: float = 1.0
    relevance: float = 1.0
    last_accessed_at: Optional[str] = None
    was_referenced: bool = False
    task_completed: bool = False

    # --- Episodic memory fields ---
    valence: float = 0.0
    consolidation_score: float = 0.0
    was_in_wm: bool = False

    # --- Semantic memory fields ---
    confidence: float = 0.0
    lifecycle: LifecycleStage = LifecycleStage.PLASTIC
    falsifiable_condition: Optional[Dict[str, Any]] = None

    # --- Common metadata ---
    session_id: Optional[str] = None
    task_id: Optional[str] = None
    created_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    event_at: Optional[str] = None
    valid_until: Optional[str] = None
    invalidated_at: Optional[str] = None
    tags: List[str] = Field(default_factory=list)
    source_references: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @property
    def created_datetime(self) -> datetime:
        return datetime.fromisoformat(self.created_at)

    @property
    def age_hours(self) -> float:
        delta = datetime.now(timezone.utc) - self.created_datetime
        return delta.total_seconds() / 3600.0

    def to_row_dict(self, exclude: set | None = None) -> Dict[str, Any]:
        """Convert to dict suitable for SQLite row insertion.
        Excludes fields stored as separate BLOBs (embedding, projection_vector)
        and any fields named in `exclude`.
        """
        d = self.model_dump(exclude=exclude or set())
        d["tags_json"] = _to_json(d.pop("tags", []))
        d["source_refs_json"] = _to_json(d.pop("source_references", []))
        d["metadata_json"] = _to_json(d.pop("metadata", {}))
        if d.get("falsifiable_condition") is not None:
            d["falsifiable_condition"] = _to_json(d["falsifiable_condition"])
        return d

    @classmethod
    def from_row(cls, row: Dict[str, Any]) -> "MemoryObject":
        """Create MemoryObject from a SQLite row dict."""
        row = dict(row)
        row["tags"] = _from_json(row.pop("tags_json", "[]"), [])
        row["source_references"] = _from_json(row.pop("source_refs_json", "[]"), [])
        row["metadata"] = _from_json(row.pop("metadata_json", "{}"), {})
        fc = row.get("falsifiable_condition")
        if isinstance(fc, str):
            row["falsifiable_condition"] = _from_json(fc, None)
        return cls(**row)


class StatusReport(BaseModel):
    """Output schema for memory_status MCP tool."""

    layers: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    vector_index: Dict[str, Any] = Field(default_factory=dict)
    consolidation: Dict[str, Any] = Field(default_factory=dict)
    graph: Dict[str, Any] = Field(default_factory=dict)
    llm: Dict[str, Any] = Field(default_factory=dict)
    creative_pool: Dict[str, Any] = Field(default_factory=dict)
    reflection: Dict[str, Any] = Field(default_factory=dict)
    health: str = "healthy"


def _to_json(obj: Any) -> str:
    import json

    return json.dumps(obj, ensure_ascii=False, default=str)


def _from_json(s: str, default: Any) -> Any:
    import json

    try:
        return json.loads(s)
    except (json.JSONDecodeError, TypeError):
        return default
