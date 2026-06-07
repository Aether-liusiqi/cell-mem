"""Episodic memory layer — hippocampus-like storage of concrete experiences.

Each interaction is stored as a full record with 384d embedding, 2048d sparse
projection vector (pattern separation), and four timestamps.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import numpy as np

from cell_mem.models import LifecycleStage, MemoryObject, MemoryType

logger = logging.getLogger(__name__)


class EpisodicMemory:
    """Stores concrete interaction records with pattern separation.

    Each record: content + 384d embedding + 2048d sparse projection vector +
    valence + consolidation score + timestamps.
    """

    _table = "episodic_memory"

    def __init__(
        self,
        sqlite_store: "SqliteStore",  # noqa: F821
        vector_store: "VectorStore",  # noqa: F821
        embed_model: "EmbeddingModel",  # noqa: F821
        projection: "ProjectionMatrix",  # noqa: F821
    ):
        from cell_mem.embedding.local import EmbeddingModel, ProjectionMatrix
        from cell_mem.storage.sqlite_store import SqliteStore
        from cell_mem.storage.vector_store import VectorStore

        self._store: SqliteStore = sqlite_store
        self._vs: VectorStore = vector_store
        self._embed: EmbeddingModel = embed_model
        self._proj: ProjectionMatrix = projection
        logger.info("EpisodicMemory ready")

    # ------------------------------------------------------------------
    # CRUD
    # ------------------------------------------------------------------

    def store(
        self,
        content: str,
        session_id: str | None = None,
        task_id: str | None = None,
        valence: float = 0.0,
        was_in_wm: bool = False,
        metadata: dict | None = None,
    ) -> MemoryObject:
        """Store episode immediately. Embedding is computed async in background.

        Content is FTS5-indexed and instantly searchable. Embedding and
        projection vectors are filled in later by EmbeddingWorker.
        """
        now = datetime.now(timezone.utc).isoformat()

        obj = MemoryObject(
            content=content,
            memory_type=MemoryType.EPISODIC,
            session_id=session_id,
            task_id=task_id,
            valence=valence,
            was_in_wm=was_in_wm,
            created_at=now,
            event_at=now,
            metadata=metadata or {},
        )
        row = obj.to_row_dict()

        self._store.execute(
            f"""INSERT INTO {self._table} (
                id, content, embedding, projection_vector,
                confidence, valence, consolidation_score, was_in_wm,
                session_id, task_id, created_at, event_at,
                valid_until, invalidated_at,
                tags_json, source_refs_json, metadata_json
            ) VALUES (?, ?, NULL, NULL, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                obj.id,
                row["content"],
                row["confidence"],
                row["valence"],
                row["consolidation_score"],
                row["was_in_wm"],
                row.get("session_id"),
                row.get("task_id"),
                row["created_at"],
                row.get("event_at"),
                row.get("valid_until"),
                row.get("invalidated_at"),
                row.get("tags_json"),
                row.get("source_refs_json"),
                row.get("metadata_json"),
            ),
        )
        self._store.commit()

        logger.debug("Episodic stored: %s (embedding pending)", obj.id[:8])
        return obj

    def get(self, episode_id: str) -> Optional[MemoryObject]:
        row = self._store.fetchone(
            f"SELECT * FROM {self._table} WHERE id = ?", (episode_id,)
        )
        if row is None:
            return None
        return MemoryObject.from_row(dict(row))

    def delete(self, episode_id: str) -> bool:
        self._store.execute(
            f"DELETE FROM {self._table} WHERE id = ?", (episode_id,)
        )
        self._store.commit()
        self._vs.delete(episode_id)
        return True

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def recall_by_vector(
        self, query_embedding: np.ndarray, top_k: int = 10
    ) -> List[MemoryObject]:
        """Vector similarity search using vec_distance_cosine."""
        hits = self._vs.search(query_embedding, top_k)
        results = []
        for ep_id, distance in hits:
            row = self._store.fetchone(
                f"SELECT * FROM {self._table} WHERE id = ?", (ep_id,)
            )
            if row is None:
                continue
            obj = MemoryObject.from_row(dict(row))
            obj.metadata["_search_distance"] = distance
            results.append(obj)
        return results

    def recall_by_keyword(self, query: str, limit: int = 10) -> List[MemoryObject]:
        """FTS5 keyword search. Results are deduplicated by ID."""
        results = []
        seen: set = set()
        try:
            rows = self._store.fetchall(
                "SELECT rowid FROM episodic_fts WHERE episodic_fts MATCH ? "
                "ORDER BY rank LIMIT ?",
                (query, limit),
            )
            for row in rows:
                # FTS content-sync tables use episodic_memory.rowid, NOT id.
                # Must join through rowid to get the actual UUID id.
                ep_row = self._store.fetchone(
                    f"SELECT * FROM {self._table} WHERE rowid = ?",
                    (row["rowid"],),
                )
                if ep_row is None:
                    continue
                ep = MemoryObject.from_row(dict(ep_row))
                if ep.id not in seen:
                    seen.add(ep.id)
                    results.append(ep)
        except Exception as exc:
            logger.warning("FTS search failed: %s", exc)
        return results

    def recall_combined(
        self, query: str, query_embedding: np.ndarray, top_k: int = 10
    ) -> List[MemoryObject]:
        """Vector + FTS5 + RRF fusion search."""
        return self.recall_by_vector(query_embedding, top_k)

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def count(self) -> int:
        row = self._store.fetchone(
            f"SELECT COUNT(*) as cnt FROM {self._table}"
        )
        return row["cnt"] if row else 0

    def oldest_at(self) -> Optional[str]:
        row = self._store.fetchone(
            f"SELECT MIN(created_at) as oldest FROM {self._table}"
        )
        return row["oldest"] if row else None

    def count_by_confidence(self) -> Dict[str, int]:
        """Returns counts by lifecycle stage."""
        plastic = self._store.fetchone(
            f"SELECT COUNT(*) as cnt FROM {self._table} WHERE confidence < 0.5"
        )
        semi = self._store.fetchone(
            f"SELECT COUNT(*) as cnt FROM {self._table} WHERE confidence >= 0.5 AND confidence < 0.8"
        )
        locked = self._store.fetchone(
            f"SELECT COUNT(*) as cnt FROM {self._table} WHERE confidence >= 0.8"
        )
        return {
            "plastic": plastic["cnt"] if plastic else 0,
            "semi_stable": semi["cnt"] if semi else 0,
            "locked": locked["cnt"] if locked else 0,
        }

    # ------------------------------------------------------------------
    # Update, batch load, similarity search
    # ------------------------------------------------------------------

    def update(
        self,
        episode_id: str,
        consolidation_score: float | None = None,
        metadata: dict | None = None,
        valence: float | None = None,
    ) -> Optional[MemoryObject]:
        """Update consolidation_score, valence, and/or metadata for an episode.

        Returns the updated MemoryObject, or None if not found.
        """
        existing = self.get(episode_id)
        if existing is None:
            return None

        import json

        new_score = (
            consolidation_score if consolidation_score is not None
            else existing.consolidation_score
        )
        new_valence = valence if valence is not None else existing.valence
        new_metadata = metadata if metadata is not None else existing.metadata
        metadata_json = json.dumps(new_metadata, ensure_ascii=False)

        self._store.execute(
            f"UPDATE {self._table} SET consolidation_score = ?, valence = ?, "
            f"metadata_json = ? WHERE id = ?",
            (new_score, new_valence, metadata_json, episode_id),
        )
        self._store.commit()
        return self.get(episode_id)

    def load_all(
        self, limit: int = 1000, offset: int = 0
    ) -> List[MemoryObject]:
        """Load episodes in batches — used by consolidation sweep."""
        rows = self._store.fetchall(
            f"SELECT * FROM {self._table} ORDER BY created_at DESC LIMIT ? OFFSET ?",
            (limit, offset),
        )
        return [MemoryObject.from_row(dict(r)) for r in rows]

    def load_all_projection_vectors(self) -> Dict[str, np.ndarray]:
        """Load (id → 2048d projection_vector) for all episodes that have one.

        Used by PatternDetector for DBSCAN clustering.
        """
        rows = self._store.fetchall(
            f"SELECT id, projection_vector FROM {self._table} "
            f"WHERE projection_vector IS NOT NULL"
        )
        result = {}
        for row in rows:
            blob = row["projection_vector"]
            if blob is not None:
                result[row["id"]] = np.frombuffer(blob, dtype=np.float32).copy()
        return result

    def count_similar_fts(self, content: str) -> int:
        """Count episodes with similar content using FTS5 search.

        Extracts 3-5 longest words as search terms. Used for S_repetition.
        """
        import re

        words = re.findall(r'\w{4,}', content.lower())[:5]
        if not words:
            return 0
        query = " OR ".join(words)
        try:
            row = self._store.fetchone(
                "SELECT COUNT(*) as cnt FROM episodic_fts WHERE episodic_fts MATCH ?",
                (query,),
            )
            return row["cnt"] if row else 0
        except Exception:
            return 0

    def find_most_similar(
        self, query_embedding: np.ndarray, exclude_id: str | None = None
    ) -> float:
        """Find max cosine similarity to any existing episode.

        Returns 0.0 if no similar episodes found. Used for S_novelty.
        """
        hits = self._vs.search(query_embedding, top_k=5)
        max_sim = 0.0
        for ep_id, distance in hits:
            if ep_id == exclude_id:
                continue
            sim = 1.0 - distance  # cosine_distance → similarity
            if sim > max_sim:
                max_sim = sim
        return max_sim

    def batch_get(self, ids: List[str]) -> List[MemoryObject]:
        """Fetch multiple episodes by ID. Used by consolidation."""
        results = []
        for ep_id in ids:
            obj = self.get(ep_id)
            if obj:
                results.append(obj)
        return results
