"""Vector storage abstraction and implementations.

SqliteVecStore uses sqlite-vec's cosine distance for similarity search.
ChromaStore provides a fallback when sqlite-vec isn't available.
"""

from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class VectorStore(ABC):
    """Abstract vector store for embedding search."""

    @abstractmethod
    def insert(
        self, id: str, vector: np.ndarray, metadata: dict | None = None
    ) -> None: ...

    @abstractmethod
    def search(
        self, query_vector: np.ndarray, top_k: int = 10
    ) -> List[Tuple[str, float]]: ...

    @abstractmethod
    def delete(self, id: str) -> None: ...

    @abstractmethod
    def count(self) -> int: ...


class SqliteVecStore(VectorStore):
    """Vector store backed by sqlite-vec + SQLite table.

    Bound to a specific table in the SqliteStore. Each memory layer
    (episodic, semantic) gets its own SqliteVecStore instance wrapping
    its respective table.

    The table must have columns: id TEXT PK, embedding BLOB.
    """

    def __init__(
        self,
        store: "SqliteStore",  # noqa: F821
        table_name: str,
        dimension: int,
    ):
        from cell_mem.storage.sqlite_store import SqliteStore

        self._store: SqliteStore = store
        self._table = table_name
        self._dim = dimension

        if not self._store.vec_available:
            raise RuntimeError(
                "sqlite-vec extension not available; use ChromaStore as fallback"
            )
        logger.info(
            "SqliteVecStore ready: table=%s dim=%d", self._table, self._dim
        )

    # ------------------------------------------------------------------
    # VectorStore interface
    # ------------------------------------------------------------------

    def insert(
        self, id: str, vector: np.ndarray, metadata: dict | None = None
    ) -> None:
        """Store embedding vector in an existing row.

        IMPORTANT: The row must already exist in the target table (created by
        the memory layer via INSERT). This method only UPDATEs the embedding
        column — it does NOT create rows. This prevents overwriting content
        and other columns that the memory layer manages.
        """
        import sqlite_vec

        vec = np.asarray(vector, dtype=np.float32).ravel()
        if vec.shape[0] != self._dim:
            raise ValueError(
                f"Vector dimension mismatch: expected {self._dim}, got {vec.shape[0]}"
            )
        blob = sqlite_vec.serialize_float32(vec.tolist())

        # UPDATE only — row must already exist (created by EpisodicMemory/SemanticMemory)
        self._store.execute(
            f"UPDATE {self._table} SET embedding = ? WHERE id = ?",
            (blob, id),
        )
        self._store.commit()

    def search(
        self, query_vector: np.ndarray, top_k: int = 10
    ) -> List[Tuple[str, float]]:
        import sqlite_vec

        vec = np.asarray(query_vector, dtype=np.float32).ravel()
        blob = sqlite_vec.serialize_float32(vec.tolist())

        rows = self._store.fetchall(
            f"SELECT id, vec_distance_cosine(embedding, ?) AS dist "
            f"FROM {self._table} "
            f"WHERE embedding IS NOT NULL "
            f"ORDER BY dist "
            f"LIMIT ?",
            (blob, top_k),
        )
        return [(row["id"], row["dist"]) for row in rows]

    def delete(self, id: str) -> None:
        self._store.execute(
            f"DELETE FROM {self._table} WHERE id = ?", (id,)
        )
        self._store.commit()

    def count(self) -> int:
        row = self._store.fetchone(
            f"SELECT COUNT(*) as cnt FROM {self._table}"
        )
        return row["cnt"] if row else 0


class ChromaStore(VectorStore):
    """ChromaDB fallback for when sqlite-vec is unavailable.

    Uses Chroma's PersistentClient with cosine distance.
    """

    def __init__(self, persist_dir: str, collection_name: str, dimension: int):
        import chromadb
        from chromadb.config import Settings

        self._client = chromadb.PersistentClient(
            path=str(Path(persist_dir).resolve()),
            settings=Settings(anonymized_telemetry=False),
        )
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        self._dim = dimension
        logger.info(
            "ChromaStore ready: collection=%s dim=%d", collection_name, dimension
        )

    def insert(
        self, id: str, vector: np.ndarray, metadata: dict | None = None
    ) -> None:
        vec = np.asarray(vector, dtype=np.float32).ravel()
        self._collection.upsert(
            ids=[id],
            embeddings=[vec.tolist()],
            metadatas=[metadata or {}],
        )

    def search(
        self, query_vector: np.ndarray, top_k: int = 10
    ) -> List[Tuple[str, float]]:
        vec = np.asarray(query_vector, dtype=np.float32).ravel()
        results = self._collection.query(
            query_embeddings=[vec.tolist()],
            n_results=top_k,
            include=["distances"],
        )
        ids = results.get("ids", [[]])[0]
        distances = results.get("distances", [[]])[0]
        return list(zip(ids, distances))

    def delete(self, id: str) -> None:
        self._collection.delete(ids=[id])

    def count(self) -> int:
        return self._collection.count()
