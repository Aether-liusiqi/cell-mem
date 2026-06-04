"""Search engine coordinating vector search across memory layers.

Phase 1: vector-only (direct strategy).
Phase 2a: FTS5 keyword + RRF fusion + association graph BFS (two_pass).
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from cell_mem.models import MemoryObject, MemoryType

logger = logging.getLogger(__name__)

# Reciprocal Rank Fusion constant
RRF_K = 60


class SearchEngine:
    """Coordinates search across episodic and semantic memory layers.

    Phase 2a strategies:
    - "direct": vector similarity only (Phase 1)
    - "rrf_fusion": vector + FTS5 keyword fused via RRF
    - "two_pass": RRF fusion + BFS activation spreading on graph
    """

    DEFAULT_TOP_K = 10

    def __init__(
        self,
        sqlite_store: "SqliteStore",  # noqa: F821
        episodic_vs: "VectorStore",  # noqa: F821
        semantic_vs: "VectorStore",  # noqa: F821
        embed_model: "EmbeddingModel",  # noqa: F821
        graph_store: Optional["GraphStore"] = None,  # noqa: F821
    ):
        from cell_mem.embedding.local import EmbeddingModel
        from cell_mem.graph.store import GraphStore
        from cell_mem.storage.sqlite_store import SqliteStore
        from cell_mem.storage.vector_store import VectorStore

        self._store: SqliteStore = sqlite_store
        self._episodic_vs: VectorStore = episodic_vs
        self._semantic_vs: VectorStore = semantic_vs
        self._embed: EmbeddingModel = embed_model
        self._graph: Optional[GraphStore] = graph_store

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def search(
        self,
        query: str,
        options: dict | None = None,
    ) -> List[MemoryObject]:
        """Search across memory layers (convenience — embeds query internally).

        Prefer search_by_vector() when the caller has already embedded the query.
        """
        query_embedding = self._embed.embed_query(query)
        return self.search_by_vector(query_embedding, options, query_text=query)

    def search_by_vector(
        self,
        query_embedding: np.ndarray,
        options: dict | None = None,
        query_text: str | None = None,
    ) -> List[MemoryObject]:
        """Search across memory layers using a pre-computed embedding vector.

        Args:
            query_embedding: Pre-computed 384d query vector.
            options: {
                memory_type: "episodic" | "semantic" | None (all),
                top_k: int (default 10),
                strategy: "direct" | "rrf_fusion" | "two_pass" (default "direct"),
                min_confidence: float (default 0.0),
            }
            query_text: Raw query text (required for FTS5 keyword search in
                        rrf_fusion and two_pass strategies).

        Returns:
            List of MemoryObject sorted by relevance.
        """
        opts = self._normalize_options(options or {})
        memory_type = opts.get("memory_type")
        top_k = opts.get("top_k", self.DEFAULT_TOP_K)
        min_confidence = opts.get("min_confidence", 0.0)
        strategy = opts.get("strategy", "direct")

        # Use query_text from options if not passed explicitly
        if query_text is None:
            query_text = opts.get("query_text", "")

        # --- Strategy dispatch ---
        if strategy == "rrf_fusion" and query_text:
            return self._rrf_fusion(query_text, query_embedding, top_k, min_confidence, memory_type)
        elif strategy == "two_pass" and query_text:
            return self._two_pass_search(query_text, query_embedding, top_k, min_confidence, memory_type)
        elif strategy not in ("direct",):
            logger.warning(
                "Unknown strategy '%s' or missing query_text, falling back to direct", strategy
            )
            strategy = "direct"

        # --- Direct strategy (Phase 1 fallback) ---
        results: List[MemoryObject] = []

        if memory_type in (None, "episodic"):
            results.extend(
                self._search_episodic(query_embedding, top_k, min_confidence)
            )

        if memory_type in (None, "semantic"):
            results.extend(
                self._search_semantic(query_embedding, top_k, min_confidence)
            )

        # Sort by distance ascending (cosine distance: 0 = identical)
        results.sort(key=lambda m: m.metadata.get("_search_distance", 1.0))

        return results[:top_k]

    # ------------------------------------------------------------------
    # Phase 2a strategies
    # ------------------------------------------------------------------

    def _rrf_fusion(
        self,
        query: str,
        query_embedding: np.ndarray,
        top_k: int,
        min_confidence: float,
        memory_type: str | None,
    ) -> List[MemoryObject]:
        """Reciprocal Rank Fusion: combine vector + FTS5 keyword results.

        RRF score = sum(1 / (K + rank_i)) across ranked lists, K=60.
        """
        # Get expanded retrieval pool for fusion
        fetch_k = max(top_k * 3, 30)

        # Ranked list 1: vector search
        vec_results = self._ranked_vector_search(query_embedding, fetch_k, min_confidence, memory_type)

        # Ranked list 2: FTS5 keyword search
        fts_results = self._ranked_fts5_search(query, fetch_k, min_confidence, memory_type)

        # RRF fusion
        rrf_scores: Dict[str, float] = {}
        obj_map: Dict[str, MemoryObject] = {}

        for rank, obj in enumerate(vec_results, start=1):
            obj_map[obj.id] = obj
            rrf_scores[obj.id] = rrf_scores.get(obj.id, 0.0) + 1.0 / (RRF_K + rank)

        for rank, obj in enumerate(fts_results, start=1):
            if obj.id not in obj_map:
                obj_map[obj.id] = obj
            rrf_scores[obj.id] = rrf_scores.get(obj.id, 0.0) + 1.0 / (RRF_K + rank)

        # Sort by RRF score descending
        sorted_ids = sorted(rrf_scores, key=rrf_scores.get, reverse=True)
        results = [obj_map[oid] for oid in sorted_ids[:top_k]]

        # Store RRF score in metadata for potential downstream use
        for obj in results:
            obj.metadata["_rrf_score"] = rrf_scores.get(obj.id, 0.0)

        return results

    def _two_pass_search(
        self,
        query: str,
        query_embedding: np.ndarray,
        top_k: int,
        min_confidence: float,
        memory_type: str | None,
    ) -> List[MemoryObject]:
        """Two-pass retrieval: RRF fusion + BFS activation spreading.

        Pass 1: RRF fusion (vector + FTS5) → initial candidate set.
        Pass 2: BFS activation spreading from candidate node IDs on graph.
        Final score = RRF_score + 0.5 * activation_score → rerank.
        """
        from cell_mem.graph.activation import spread_activation

        # Pass 1: RRF fusion (get more results for graph traversal)
        pass1_k = max(top_k * 2, 20)
        pass1_results = self._rrf_fusion(query, query_embedding, pass1_k, min_confidence, memory_type)

        # Pass 2: Activation spreading
        if self._graph and pass1_results:
            seed_ids = [obj.id for obj in pass1_results]
            activations = spread_activation(
                seed_ids, self._graph, max_hops=2, decay=0.3
            )

            # Build final scores: RRF + 0.5 * activation
            final_scores: Dict[str, float] = {}
            obj_map: Dict[str, MemoryObject] = {}

            for obj in pass1_results:
                obj_map[obj.id] = obj
                final_scores[obj.id] = obj.metadata.get("_rrf_score", 0.0)

            # Add graph-discovered nodes
            for node_id, activation in activations.items():
                if node_id in obj_map:
                    # Boost existing result
                    final_scores[node_id] += 0.5 * activation
                else:
                    # New node from graph — fetch it
                    new_obj = self._fetch_by_id(node_id)
                    if new_obj and new_obj.confidence >= min_confidence:
                        obj_map[node_id] = new_obj
                        final_scores[node_id] = 0.5 * activation

            # Sort by final score descending
            sorted_ids = sorted(final_scores, key=final_scores.get, reverse=True)
            results = [obj_map[oid] for oid in sorted_ids[:top_k]]
        else:
            # No graph store or no results — fall back to RRF-only
            results = pass1_results[:top_k]

        return results

    # ------------------------------------------------------------------
    # Ranked retrieval helpers (return ranked lists for RRF)
    # ------------------------------------------------------------------

    def _ranked_vector_search(
        self,
        query_embedding: np.ndarray,
        top_k: int,
        min_confidence: float,
        memory_type: str | None,
    ) -> List[MemoryObject]:
        """Return vector search results in ranked order (closest first)."""
        results: List[MemoryObject] = []

        if memory_type in (None, "episodic"):
            results.extend(self._search_episodic(query_embedding, top_k, min_confidence))

        if memory_type in (None, "semantic"):
            results.extend(self._search_semantic(query_embedding, top_k, min_confidence))

        results.sort(key=lambda m: m.metadata.get("_search_distance", 1.0))
        return results[:top_k]

    def _ranked_fts5_search(
        self,
        query: str,
        top_k: int,
        min_confidence: float,
        memory_type: str | None,
    ) -> List[MemoryObject]:
        """Return FTS5 keyword search results in ranked order."""
        results: List[MemoryObject] = []

        if memory_type in (None, "episodic"):
            try:
                rows = self._store.fetchall(
                    """SELECT rowid FROM episodic_fts WHERE episodic_fts MATCH ?
                       ORDER BY rank LIMIT ?""",
                    (query, top_k),
                )
                for row in rows:
                    ep_row = self._store.fetchone(
                        "SELECT * FROM episodic_memory WHERE rowid = ?", (row["rowid"],)
                    )
                    if ep_row is None:
                        continue
                    obj = MemoryObject.from_row(dict(ep_row))
                    if obj.confidence < min_confidence:
                        continue
                    results.append(obj)
            except Exception:
                pass  # FTS5 query syntax error → skip

        if memory_type in (None, "semantic"):
            try:
                rows = self._store.fetchall(
                    """SELECT rowid FROM semantic_fts WHERE semantic_fts MATCH ?
                       ORDER BY rank LIMIT ?""",
                    (query, top_k),
                )
                for row in rows:
                    sem_row = self._store.fetchone(
                        "SELECT * FROM semantic_memory WHERE rowid = ?", (row["rowid"],)
                    )
                    if sem_row is None:
                        continue
                    obj = MemoryObject.from_row(dict(sem_row))
                    obj.memory_type = MemoryType.SEMANTIC
                    if obj.confidence < min_confidence:
                        continue
                    results.append(obj)
            except Exception:
                pass  # FTS5 query syntax error → skip

        return results[:top_k]

    # ------------------------------------------------------------------
    # Internal search methods
    # ------------------------------------------------------------------

    def _search_episodic(
        self,
        query_embedding: np.ndarray,
        top_k: int,
        min_confidence: float,
    ) -> List[MemoryObject]:
        hits = self._episodic_vs.search(query_embedding, top_k)
        results = []
        for ep_id, distance in hits:
            row = self._store.fetchone(
                "SELECT * FROM episodic_memory WHERE id = ?", (ep_id,)
            )
            if row is None:
                continue
            obj = MemoryObject.from_row(dict(row))
            if obj.confidence < min_confidence:
                continue
            obj.metadata["_search_distance"] = distance
            results.append(obj)
        return results

    def _search_semantic(
        self,
        query_embedding: np.ndarray,
        top_k: int,
        min_confidence: float,
    ) -> List[MemoryObject]:
        if self._semantic_vs is None:
            return []
        hits = self._semantic_vs.search(query_embedding, top_k)
        results = []
        for sem_id, distance in hits:
            row = self._store.fetchone(
                "SELECT * FROM semantic_memory WHERE id = ?", (sem_id,)
            )
            if row is None:
                continue
            obj = MemoryObject.from_row(dict(row))
            obj.memory_type = MemoryType.SEMANTIC
            if obj.confidence < min_confidence:
                continue
            obj.metadata["_search_distance"] = distance
            results.append(obj)
        return results

    def _fetch_by_id(self, memory_id: str) -> Optional[MemoryObject]:
        """Try to fetch a memory by ID from either episodic or semantic layer."""
        row = self._store.fetchone(
            "SELECT * FROM episodic_memory WHERE id = ?", (memory_id,)
        )
        if row:
            return MemoryObject.from_row(dict(row))

        row = self._store.fetchone(
            "SELECT * FROM semantic_memory WHERE id = ?", (memory_id,)
        )
        if row:
            obj = MemoryObject.from_row(dict(row))
            obj.memory_type = MemoryType.SEMANTIC
            return obj

        return None

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize_options(options: dict) -> dict:
        """Fill defaults for search options."""
        defaults: Dict[str, Any] = {
            "memory_type": None,
            "top_k": SearchEngine.DEFAULT_TOP_K,
            "strategy": "direct",
            "min_confidence": 0.0,
            "query_text": "",
        }
        merged = {**defaults, **options}
        # Validate memory_type
        mt = merged.get("memory_type")
        if mt is not None and mt not in ("episodic", "semantic"):
            logger.warning("Invalid memory_type '%s', searching all layers", mt)
            merged["memory_type"] = None
        # Validate strategy
        strategy = merged.get("strategy")
        if strategy not in ("direct", "rrf_fusion", "two_pass"):
            logger.warning("Unknown strategy '%s', using direct", strategy)
            merged["strategy"] = "direct"
        return merged
