"""Background embedding worker — async vectorization daemon.

Decouples slow embedding computation from real-time save/recall.
Writes content to SQLite immediately (FTS5-searchable); computes
embeddings lazily in a background thread without blocking any ops.

Analogy: hippocampal fast encoding (save) vs. cortical slow consolidation
         (background embedding + vector indexing).
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# Batch size for processing pending records
_BATCH_SIZE = 30
# Poll interval when no pending records
_IDLE_SLEEP_SEC = 2.0


class EmbeddingWorker:
    """Background daemon thread for async embedding computation.

    Usage:
        worker = EmbeddingWorker(store, embed_model, projection,
                                 episodic_vs, semantic_vs)
        worker.start()   # called after MCP server is ready
        # ... save/recall work normally ...
        worker.stop()    # during shutdown
    """

    def __init__(
        self,
        store: "SqliteStore",           # noqa: F821
        embed_model: "EmbeddingModel",   # noqa: F821
        projection: "ProjectionMatrix",  # noqa: F821
        episodic_vs,                     # SqliteVecStore or ChromaStore
        semantic_vs,                     # SqliteVecStore or ChromaStore
    ):
        self._store = store
        self._embed = embed_model
        self._proj = projection
        self._episodic_vs = episodic_vs
        self._semantic_vs = semantic_vs
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._processed = 0

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Start background processing in a daemon thread."""
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True, name="cell-mem-embed")
        self._thread.start()
        logger.info("EmbeddingWorker started (background daemon)")

    def stop(self) -> None:
        """Signal the worker to stop. Thread exits at next idle check."""
        self._running = False
        logger.info("EmbeddingWorker stopping (processed %d records)", self._processed)

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def _run(self) -> None:
        """Main loop: load model, then poll + process pending records forever."""
        # Phase 1: load the embedding model (the slow part — done in background)
        try:
            logger.info("EmbeddingWorker loading model in background...")
            self._embed.ensure_loaded()
            logger.info("EmbeddingWorker model ready (dim=%d)", self._embed.DIM)
        except Exception as exc:
            logger.error("EmbeddingWorker failed to load model: %s", exc)
            return  # Cannot work without model

        # Phase 2: process pending records continuously
        while self._running:
            batch = self._get_pending(limit=_BATCH_SIZE)
            if not batch:
                time.sleep(_IDLE_SLEEP_SEC)
                continue

            try:
                n = self._process_batch(batch)
                self._processed += n
            except Exception as exc:
                logger.warning("EmbeddingWorker batch failed: %s", exc)
                time.sleep(0.5)  # Brief pause on error

        logger.info("EmbeddingWorker exited (total %d processed)", self._processed)

    # ------------------------------------------------------------------
    # Pending record queries
    # ------------------------------------------------------------------

    def _get_pending(self, limit: int = _BATCH_SIZE) -> list:
        """Fetch records with NULL embeddings from both episodic and semantic tables."""
        episodic = self._store.fetchall(
            """SELECT id, content, 'episodic' AS src
               FROM episodic_memory
               WHERE embedding IS NULL
               ORDER BY created_at ASC
               LIMIT ?""",
            (limit,),
        )
        semantic = self._store.fetchall(
            """SELECT id, content, 'semantic' AS src
               FROM semantic_memory
               WHERE embedding IS NULL
               ORDER BY created_at ASC
               LIMIT ?""",
            (limit,),
        )
        # Interleave for fairness
        combined = []
        e_rows = list(episodic or [])
        s_rows = list(semantic or [])
        max_len = max(len(e_rows), len(s_rows))
        for i in range(max_len):
            if i < len(e_rows):
                combined.append(dict(e_rows[i]))
            if i < len(s_rows):
                combined.append(dict(s_rows[i]))
        return combined[:limit]

    # ------------------------------------------------------------------
    # Batch processing
    # ------------------------------------------------------------------

    def _process_batch(self, records: list) -> int:
        """Compute embeddings for a batch, update tables + vector stores."""
        import sqlite_vec

        texts = [r["content"] for r in records]
        embeddings = self._embed.embed_batch(texts)  # (N, 384)

        count = 0
        for i, rec in enumerate(records):
            vec = embeddings[i] if len(embeddings.shape) == 2 else embeddings
            try:
                rec_id = rec["id"]
                src = rec["src"]
                emb_blob = sqlite_vec.serialize_float32(
                    np.asarray(vec, dtype=np.float32).ravel().tolist()
                )

                if src == "episodic":
                    self._process_episodic(rec_id, np.asarray(vec, dtype=np.float32), emb_blob)
                else:
                    self._process_semantic(rec_id, np.asarray(vec, dtype=np.float32), emb_blob)

                count += 1
            except Exception as exc:
                logger.debug("EmbeddingWorker failed record %s: %s",
                           rec.get("id", "?")[:8], exc)

        if count:
            self._store.commit()
        return count

    def _process_episodic(self, rec_id: str, vec: np.ndarray, emb_blob: bytes) -> None:
        """Update one episodic record with embedding + projection + vector index."""
        import sqlite_vec

        proj_vec = self._proj.project(vec)
        proj_blob = sqlite_vec.serialize_float32(proj_vec.tolist())

        self._store.execute(
            """UPDATE episodic_memory
               SET embedding = ?, projection_vector = ?
               WHERE id = ?""",
            (emb_blob, proj_blob, rec_id),
        )

        # Index in vector store for fast similarity search
        try:
            self._episodic_vs.insert(rec_id, vec)
        except Exception as exc:
            logger.debug("EmbeddingWorker episodic vs insert failed %s: %s",
                       rec_id[:8], exc)

    def _process_semantic(self, rec_id: str, vec: np.ndarray, emb_blob: bytes) -> None:
        """Update one semantic record with embedding + vector index."""
        self._store.execute(
            "UPDATE semantic_memory SET embedding = ? WHERE id = ?",
            (emb_blob, rec_id),
        )

        try:
            self._semantic_vs.insert(rec_id, vec)
        except Exception as exc:
            logger.debug("EmbeddingWorker semantic vs insert failed %s: %s",
                       rec_id[:8], exc)

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    @property
    def processed_count(self) -> int:
        return self._processed

    @property
    def is_ready(self) -> bool:
        """Whether the embedding model is loaded and processing has begun."""
        return self._embed.loaded

    def pending_count(self) -> int:
        """Number of records still waiting for embedding."""
        try:
            ep = self._store.fetchone(
                "SELECT COUNT(*) as cnt FROM episodic_memory WHERE embedding IS NULL"
            )
            sm = self._store.fetchone(
                "SELECT COUNT(*) as cnt FROM semantic_memory WHERE embedding IS NULL"
            )
            return (ep["cnt"] if ep else 0) + (sm["cnt"] if sm else 0)
        except Exception:
            return -1
