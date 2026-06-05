"""Semantic memory layer — neocortex-like abstract knowledge.

Stores facts, patterns, user preferences, and project knowledge detached
from specific episodes. Knowledge has confidence-driven lifecycle stages:
plastic (<0.5) → semi-stable (0.5-0.8) → locked (>0.8).
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from cell_mem.models import LifecycleStage, MemoryObject, MemoryType

logger = logging.getLogger(__name__)


class SemanticMemory:
    """Abstract knowledge storage with confidence-driven lifecycle.

    Knowledge entries are indexed by 384d embeddings for vector search.
    Each entry has a falsifiable condition (optional structured JSON).
    """

    _table = "semantic_memory"

    def __init__(
        self,
        sqlite_store: "SqliteStore",  # noqa: F821
        vector_store: "VectorStore",  # noqa: F821
        embed_model: "EmbeddingModel",  # noqa: F821
    ):
        from cell_mem.embedding.local import EmbeddingModel
        from cell_mem.storage.sqlite_store import SqliteStore
        from cell_mem.storage.vector_store import VectorStore

        self._store: SqliteStore = sqlite_store
        self._vs: VectorStore = vector_store
        self._embed: EmbeddingModel = embed_model
        logger.info("SemanticMemory ready")

    # ------------------------------------------------------------------
    # CRUD
    # ------------------------------------------------------------------

    def add(
        self,
        content: str,
        confidence: float = 0.0,
        falsifiable_condition: dict | None = None,
        source_references: List[str] | None = None,
        metadata: dict | None = None,
    ) -> MemoryObject:
        """Create a semantic knowledge entry. Embedding computed async in background.

        Content is FTS5-indexed and instantly searchable. Embedding vector
        is filled in later by EmbeddingWorker.
        """
        now = datetime.now(timezone.utc).isoformat()
        lifecycle = self._confidence_to_lifecycle(confidence)

        fc_json = json.dumps(falsifiable_condition, ensure_ascii=False) if falsifiable_condition else None

        obj = MemoryObject(
            content=content,
            memory_type=MemoryType.SEMANTIC,
            confidence=confidence,
            lifecycle=lifecycle,
            falsifiable_condition=falsifiable_condition,
            source_references=source_references or [],
            created_at=now,
            metadata=metadata or {},
        )
        row = obj.to_row_dict()

        self._store.execute(
            f"""INSERT INTO {self._table} (
                id, content, embedding, confidence, lifecycle,
                falsifiable_condition, invalidated_at, source_refs_json,
                created_at, updated_at, tags_json, metadata_json
            ) VALUES (?, ?, NULL, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                obj.id,
                row["content"],
                row["confidence"],
                row["lifecycle"],
                fc_json,
                row.get("invalidated_at"),
                row["source_refs_json"],
                row["created_at"],
                row["created_at"],  # updated_at = created_at on insert
                row["tags_json"],
                row["metadata_json"],
            ),
        )
        self._store.commit()

        logger.debug("Semantic added: %s (confidence=%.2f, embedding pending)", obj.id[:8], confidence)
        return obj

    def update(
        self,
        knowledge_id: str,
        content: str | None = None,
        confidence: float | None = None,
        falsifiable_condition: dict | None = None,
    ) -> Optional[MemoryObject]:
        """Update an existing knowledge entry. Re-embed if content changes."""
        existing = self.get(knowledge_id)
        if existing is None:
            return None

        now = datetime.now(timezone.utc).isoformat()
        new_content = content or existing.content
        new_confidence = confidence if confidence is not None else existing.confidence
        lifecycle = self._confidence_to_lifecycle(new_confidence)

        if content is not None and content != existing.content:
            # Content changed — set embedding to NULL, worker will re-compute
            self._store.execute(
                f"UPDATE {self._table} SET content = ?, embedding = NULL WHERE id = ?",
                (content, knowledge_id),
            )

        fc_json = (
            json.dumps(falsifiable_condition, ensure_ascii=False)
            if falsifiable_condition is not None
            else json.dumps(existing.falsifiable_condition, ensure_ascii=False)
            if existing.falsifiable_condition
            else None
        )

        self._store.execute(
            f"UPDATE {self._table} SET confidence = ?, lifecycle = ?, "
            f"falsifiable_condition = ?, updated_at = ? WHERE id = ?",
            (new_confidence, lifecycle.value, fc_json, now, knowledge_id),
        )
        self._store.commit()

        logger.debug("Semantic updated: %s (confidence=%.2f)", knowledge_id, new_confidence)
        return self.get(knowledge_id)

    def get(self, knowledge_id: str) -> Optional[MemoryObject]:
        row = self._store.fetchone(
            f"SELECT * FROM {self._table} WHERE id = ?", (knowledge_id,)
        )
        if row is None:
            return None
        return MemoryObject.from_row(dict(row))

    def delete(self, knowledge_id: str) -> bool:
        self._store.execute(
            f"DELETE FROM {self._table} WHERE id = ?", (knowledge_id,)
        )
        self._store.commit()
        self._vs.delete(knowledge_id)
        return True

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(
        self, query_embedding: np.ndarray, top_k: int = 10
    ) -> List[MemoryObject]:
        """Vector similarity search."""
        hits = self._vs.search(query_embedding, top_k)
        results = []
        for sem_id, distance in hits:
            row = self._store.fetchone(
                f"SELECT * FROM {self._table} WHERE id = ?", (sem_id,)
            )
            if row is None:
                continue
            obj = MemoryObject.from_row(dict(row))
            obj.memory_type = MemoryType.SEMANTIC
            obj.metadata["_search_distance"] = distance
            results.append(obj)
        return results

    # ------------------------------------------------------------------
    # Seed config import
    # ------------------------------------------------------------------

    def import_seed_config(self, config_path: str) -> int:
        """Import seed knowledge from a JSON file.

        Expected format:
        {
          "seed_knowledge": [
            {
              "content": "...",
              "confidence": 0.9,
              "falsifiable_condition": {...},
              "source": "manual"
            }
          ]
        }

        Returns count of imported entries.
        """
        path = Path(config_path)
        if not path.exists():
            logger.warning("Seed config not found: %s, starting empty", config_path)
            return 0

        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            logger.error("Invalid seed config JSON: %s", exc)
            raise ValueError(f"Invalid seed config JSON: {exc}") from exc

        entries = data.get("seed_knowledge", [])
        imported = 0
        for entry in entries:
            try:
                self.add(
                    content=entry["content"],
                    confidence=entry.get("confidence", 0.5),
                    falsifiable_condition=entry.get("falsifiable_condition"),
                    source_references=entry.get("source_references", []),
                    metadata={"source": entry.get("source", "seed_config")},
                )
                imported += 1
            except Exception as exc:
                logger.warning("Failed to import seed entry '%s': %s",
                               entry.get("content", "")[:50], exc)

        logger.info("Imported %d/%d seed knowledge entries", imported, len(entries))
        return imported

    # ------------------------------------------------------------------
    # Bulk access
    # ------------------------------------------------------------------

    def load_all(self, limit: int = 200, offset: int = 0) -> List[MemoryObject]:
        """Load semantic entries in batches."""
        rows = self._store.fetchall(
            f"SELECT * FROM {self._table} ORDER BY created_at DESC LIMIT ? OFFSET ?",
            (limit, offset),
        )
        return [MemoryObject.from_row(dict(r)) for r in rows]

    def recall_by_vector(
        self, query_embedding: np.ndarray, top_k: int = 10
    ) -> List[MemoryObject]:
        """Vector search returning MemoryObjects (reuses search())."""
        return self.search(query_embedding, top_k)

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def count(self) -> int:
        row = self._store.fetchone(
            f"SELECT COUNT(*) as cnt FROM {self._table}"
        )
        return row["cnt"] if row else 0

    def avg_confidence(self) -> float:
        row = self._store.fetchone(
            f"SELECT AVG(confidence) as avg FROM {self._table}"
        )
        if row and row["avg"] is not None:
            return row["avg"]
        return 0.0

    def count_with_falsifiable(self) -> int:
        row = self._store.fetchone(
            f"SELECT COUNT(*) as cnt FROM {self._table} WHERE falsifiable_condition IS NOT NULL"
        )
        return row["cnt"] if row else 0

    def count_expired(self) -> int:
        row = self._store.fetchone(
            f"SELECT COUNT(*) as cnt FROM {self._table} WHERE invalidated_at IS NOT NULL"
        )
        return row["cnt"] if row else 0

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _confidence_to_lifecycle(confidence: float) -> LifecycleStage:
        if confidence >= 0.8:
            return LifecycleStage.LOCKED
        elif confidence >= 0.5:
            return LifecycleStage.SEMI_STABLE
        else:
            return LifecycleStage.PLASTIC
