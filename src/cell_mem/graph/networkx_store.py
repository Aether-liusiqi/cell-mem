"""NetworkX-based graph store backed by SQLite persistence (Phase 2a).

Replaces StubGraphStore with a real DiGraph, loading from graph_nodes
and graph_edges tables on init and writing through to SQLite on mutation.

Cold edge decay is computed lazily on read: edges not traversed for > 7 days
have their weight multiplied by exp(-days/30).
"""

from __future__ import annotations

import logging
import math
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

import networkx as nx

from cell_mem.graph.store import GraphStore

logger = logging.getLogger(__name__)

_COLD_DECAY_THRESHOLD_DAYS = 7.0
_COLD_DECAY_HALFLIFE_DAYS = 30.0


class NetworkXGraphStore(GraphStore):
    """Real graph store: NetworkX DiGraph + SQLite adjacency tables.

    On initialization, loads all edges and nodes from SQLite into
    a NetworkX DiGraph. Mutations are immediately flushed to SQLite.

    Cold edge decay: edges with last_traversed_at older than 7 days
    have effective weight = w * exp(-days/30).
    """

    def __init__(self, sqlite_store: "SqliteStore") -> None:  # noqa: F821
        from cell_mem.storage.sqlite_store import SqliteStore

        self._store: SqliteStore = sqlite_store
        self._graph = nx.DiGraph()
        self._load_from_db()
        logger.info(
            "NetworkXGraphStore ready: %d nodes, %d edges",
            self._graph.number_of_nodes(),
            self._graph.number_of_edges(),
        )

    # ------------------------------------------------------------------
    # GraphStore ABC implementation
    # ------------------------------------------------------------------

    def add_edge(
        self,
        source: str,
        target: str,
        weight: float,
        relation_type: str,
    ) -> None:
        """Add or update a directed edge. Auto-creates nodes if missing.

        Args:
            source: Source node ID.
            target: Target node ID.
            weight: Edge weight in [-1, 1].
            relation_type: One of EdgeType values.
        """
        # Validate
        if not (-1.0 <= weight <= 1.0):
            raise ValueError(f"Edge weight {weight} out of range [-1, 1]")

        valid_types = {"associated_with", "causes", "contradicts", "is_a", "part_of"}
        if relation_type not in valid_types:
            raise ValueError(
                f"Invalid relation_type '{relation_type}'. "
                f"Must be one of: {sorted(valid_types)}"
            )

        now = datetime.now(timezone.utc).isoformat()

        # Auto-create nodes if missing
        for node_id, label in [(source, source), (target, target)]:
            if node_id not in self._graph:
                self._ensure_node(node_id, "memory_concept", label, now)

        # Persist to SQLite (INSERT OR REPLACE for idempotent updates)
        self._store.execute(
            """INSERT OR REPLACE INTO graph_edges
               (source_id, target_id, weight, relation_type, created_at, last_traversed_at)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (source, target, weight, relation_type, now, None),
        )
        self._store.commit()

        # Update NetworkX graph
        self._graph.add_edge(
            source, target,
            weight=weight,
            relation_type=relation_type,
            created_at=now,
            last_traversed_at=None,
        )

        logger.debug("Edge: %s -[%s %.2f]-> %s", source[:12], relation_type, weight, target[:12])

    def get_neighbors(self, node_id: str) -> List[Tuple[str, float, str]]:
        """Return (target_id, effective_weight, relation_type) for all outgoing edges.

        Applies cold edge decay lazily on read.
        """
        if node_id not in self._graph:
            return []

        results: List[Tuple[str, float, str]] = []
        for _, target, data in self._graph.out_edges(node_id, data=True):
            effective_weight = self._apply_cold_decay(
                data.get("weight", 0.0),
                data.get("last_traversed_at"),
            )
            results.append((target, effective_weight, data.get("relation_type", "")))
        return results

    def remove_edge(self, source: str, target: str) -> None:
        """Remove an edge. No-op if edge does not exist."""
        if self._graph.has_edge(source, target):
            self._graph.remove_edge(source, target)

        self._store.execute(
            "DELETE FROM graph_edges WHERE source_id = ? AND target_id = ?",
            (source, target),
        )
        self._store.commit()
        logger.debug("Edge removed: %s -> %s", source[:12], target[:12])

    def node_count(self) -> int:
        return self._graph.number_of_nodes()

    def edge_count(self) -> int:
        return self._graph.number_of_edges()

    # ------------------------------------------------------------------
    # Extended API (not in ABC, used by activation spreading)
    # ------------------------------------------------------------------

    def add_node(
        self,
        node_id: str,
        node_type: str = "memory_concept",
        label: str = "",
    ) -> None:
        """Explicitly add a node (no-op if exists)."""
        now = datetime.now(timezone.utc).isoformat()
        self._ensure_node(node_id, node_type, label or node_id, now)

    def get_out_edges(
        self, node_id: str
    ) -> List[Tuple[str, str, float, str]]:
        """Return (source, target, effective_weight, relation_type) for outgoing edges."""
        if node_id not in self._graph:
            return []

        results: List[Tuple[str, str, float, str]] = []
        for source, target, data in self._graph.out_edges(node_id, data=True):
            effective_weight = self._apply_cold_decay(
                data.get("weight", 0.0),
                data.get("last_traversed_at"),
            )
            results.append((source, target, effective_weight, data.get("relation_type", "")))
        return results

    def has_node(self, node_id: str) -> bool:
        return node_id in self._graph

    def mark_traversed(self, node_id: str) -> None:
        """Update last_traversed_at for all incoming edges to this node."""
        now = datetime.now(timezone.utc).isoformat()
        # Update in-memory
        for source, target, data in self._graph.in_edges(node_id, data=True):
            data["last_traversed_at"] = now
        # Persist to SQLite
        self._store.execute(
            "UPDATE graph_edges SET last_traversed_at = ? WHERE target_id = ?",
            (now, node_id),
        )
        self._store.commit()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_from_db(self) -> None:
        """Rebuild NetworkX graph from SQLite tables."""
        # Load nodes
        node_rows = self._store.fetchall("SELECT id, node_type, label, created_at FROM graph_nodes")
        for row in node_rows:
            self._graph.add_node(
                row["id"],
                node_type=row["node_type"],
                label=row["label"],
                created_at=row["created_at"],
            )

        # Load edges
        edge_rows = self._store.fetchall(
            """SELECT source_id, target_id, weight, relation_type, created_at, last_traversed_at
               FROM graph_edges"""
        )
        for row in edge_rows:
            if row["source_id"] not in self._graph:
                self._graph.add_node(row["source_id"], node_type="memory_concept", label=row["source_id"])
            if row["target_id"] not in self._graph:
                self._graph.add_node(row["target_id"], node_type="memory_concept", label=row["target_id"])
            self._graph.add_edge(
                row["source_id"], row["target_id"],
                weight=row["weight"],
                relation_type=row["relation_type"],
                created_at=row["created_at"],
                last_traversed_at=row["last_traversed_at"],
            )

    def _ensure_node(
        self, node_id: str, node_type: str, label: str, created_at: str
    ) -> None:
        """Create a node in SQLite and NetworkX if it doesn't exist."""
        if node_id in self._graph:
            return
        self._store.execute(
            "INSERT OR IGNORE INTO graph_nodes (id, node_type, label, created_at) VALUES (?, ?, ?, ?)",
            (node_id, node_type, label, created_at),
        )
        self._store.commit()
        self._graph.add_node(
            node_id, node_type=node_type, label=label, created_at=created_at
        )

    @staticmethod
    def _apply_cold_decay(
        weight: float, last_traversed_at: Optional[str]
    ) -> float:
        """Apply cold edge decay if last traversal was > 7 days ago.

        w *= exp(-days_since / 30)
        """
        if last_traversed_at is None:
            return weight
        try:
            last = datetime.fromisoformat(last_traversed_at)
            days = (datetime.now(timezone.utc) - last).total_seconds() / 86400.0
            if days > _COLD_DECAY_THRESHOLD_DAYS:
                decay = math.exp(-(days - _COLD_DECAY_THRESHOLD_DAYS) / _COLD_DECAY_HALFLIFE_DAYS)
                return weight * decay
        except (ValueError, TypeError):
            pass
        return weight
