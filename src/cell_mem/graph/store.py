"""Graph store abstraction and Phase 1 no-op stub.

Phase 2 replaces StubGraphStore with NetworkX (WAL-persisted) and
eventually Kuzu (embedded Cypher engine).
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import List, Tuple

logger = logging.getLogger(__name__)


class GraphStore(ABC):
    """Abstract graph store for memory association edges."""

    @abstractmethod
    def add_edge(
        self, source: str, target: str, weight: float, relation_type: str
    ) -> None: ...

    @abstractmethod
    def get_neighbors(self, node_id: str) -> List[Tuple[str, float, str]]: ...

    @abstractmethod
    def get_out_edges(
        self, node_id: str
    ) -> List[Tuple[str, str, float, str]]:
        """Return (source, target, effective_weight, relation_type) for outgoing edges.

        Used by spread_activation() for BFS graph traversal.
        """
        ...

    @abstractmethod
    def remove_edge(self, source: str, target: str) -> None: ...

    @abstractmethod
    def node_count(self) -> int: ...

    @abstractmethod
    def edge_count(self) -> int: ...


class StubGraphStore(GraphStore):
    """Phase 1 no-op graph store. All operations are logged and return empty."""

    def __init__(self):
        logger.info("StubGraphStore ready (Phase 1 — no graph operations)")

    def add_edge(
        self, source: str, target: str, weight: float, relation_type: str
    ) -> None:
        logger.debug(
            "Graph stub: add_edge(%s -> %s, w=%.2f, %s)",
            source[:8], target[:8], weight, relation_type,
        )

    def get_neighbors(self, node_id: str) -> List[Tuple[str, float, str]]:
        return []

    def get_out_edges(self, node_id: str) -> List[Tuple[str, str, float, str]]:
        return []

    def remove_edge(self, source: str, target: str) -> None:
        logger.debug("Graph stub: remove_edge(%s, %s)", source[:8], target[:8])

    def node_count(self) -> int:
        return 0

    def edge_count(self) -> int:
        return 0
