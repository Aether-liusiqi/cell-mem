"""Working memory layer — prefrontal-cortex-like transient storage.

Key mechanisms:
- Attention decay: score = base_priority × relevance × exp(-Δt / τ), τ=300s
- Proactive refresh: every 60s, scan endangered items and recompute relevance
- Session-end four-way routing: keep / downgrade / store / discard
- Cross-session continuity: decay timers persist across sessions
"""

from __future__ import annotations

import logging
import math
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import numpy as np

from cell_mem.models import MemoryObject, MemoryType

logger = logging.getLogger(__name__)


class WorkingMemory:
    """Transient storage simulating prefrontal cortex persistent activity.

    Capacity: ~50 items. Items decay with attention half-life τ=300s.
    At session end, items are routed four ways based on attention score
    and whether they were actively referenced.
    """

    TAU = 300.0  # attention half-life in seconds (5 minutes)
    DECAY_THRESHOLD = 0.15  # below this → endangered
    MAX_ITEMS = 50
    REFRESH_INTERVAL = 60.0  # seconds between proactive refresh scans
    STALE_TIMEOUT = 900.0  # 15 minutes without access → eviction candidate
    SESSION_END_THRESHOLD = 0.3  # above this → retain or downgrade

    def __init__(
        self,
        sqlite_store: "SqliteStore",  # noqa: F821
    ):
        from cell_mem.storage.sqlite_store import SqliteStore

        self._store: SqliteStore = sqlite_store
        self._last_refresh_time: float = time.time()

        # Load surviving items from previous sessions
        self._load_survivors()

        logger.info("WorkingMemory initialized (τ=%.0fs, max=%d)", self.TAU, self.MAX_ITEMS)

    # ------------------------------------------------------------------
    # CRUD
    # ------------------------------------------------------------------

    def add(
        self,
        content: str,
        session_id: str,
        base_priority: float = 1.0,
        metadata: dict | None = None,
    ) -> MemoryObject:
        """Add an item to working memory. Evict lowest-attention item if at capacity."""
        self._evict_if_full()

        now = datetime.now(timezone.utc).isoformat()
        obj = MemoryObject(
            content=content,
            memory_type=MemoryType.WORKING,
            session_id=session_id,
            base_priority=base_priority,
            last_accessed_at=now,
            created_at=now,
            metadata=metadata or {},
        )

        # Only include fields that exist in working_memory table
        _WM_COLS = {
            "id", "content", "attention_score", "base_priority", "relevance",
            "last_accessed_at", "was_referenced", "task_completed",
            "session_id", "created_at", "tags_json", "metadata_json",
        }
        row = obj.to_row_dict(exclude={
            "memory_type", "valence", "consolidation_score", "was_in_wm",
            "confidence", "lifecycle", "falsifiable_condition",
            "task_id", "event_at", "valid_until", "invalidated_at",
            "tags", "source_references", "metadata",
        })
        row = {k: v for k, v in row.items() if k in _WM_COLS}
        row["task_completed"] = int(row.get("task_completed", False))
        row["was_referenced"] = int(row.get("was_referenced", False))
        columns = ", ".join(row.keys())
        placeholders = ", ".join("?" for _ in row)
        self._store.execute(
            f"INSERT INTO {self._table} ({columns}) VALUES ({placeholders})",
            tuple(row.values()),
        )
        self._store.commit()

        logger.debug("WM add: %s (total: %d)", obj.id, self.count())
        return obj

    def access(self, item_id: str) -> bool:
        """Mark item as referenced — resets decay timer. Returns False if not found."""
        now = datetime.now(timezone.utc).isoformat()
        self._store.execute(
            f"UPDATE {self._table} SET last_accessed_at = ?, was_referenced = 1 "
            f"WHERE id = ?",
            (now, item_id),
        )
        self._store.commit()
        affected = self._store.get_connection().total_changes
        return affected > 0

    def preactivate(self, item_id: str) -> bool:
        """Associative preactivation — halves decay without marking referenced."""
        row = self._store.fetchone(
            f"SELECT last_accessed_at FROM {self._table} WHERE id = ?", (item_id,)
        )
        if row is None:
            return False

        # Halve the effective elapsed time by moving last_accessed_at forward
        last = datetime.fromisoformat(row["last_accessed_at"])
        now = datetime.now(timezone.utc)
        midpoint = last + (now - last) / 2
        self._store.execute(
            f"UPDATE {self._table} SET last_accessed_at = ? WHERE id = ?",
            (midpoint.isoformat(), item_id),
        )
        self._store.commit()
        return True

    def remove(self, item_id: str) -> bool:
        """Remove an item from working memory."""
        self._store.execute(
            f"DELETE FROM {self._table} WHERE id = ?", (item_id,)
        )
        self._store.commit()
        return True

    # ------------------------------------------------------------------
    # Attention computation
    # ------------------------------------------------------------------

    def compute_attention(
        self,
        item_id: str,
        current_time: datetime | None = None,
        relevance: float = 1.0,
    ) -> Optional[float]:
        """AttentionScore = base_priority × relevance × exp(-Δt / τ)."""
        row = self._store.fetchone(
            f"SELECT base_priority, last_accessed_at FROM {self._table} WHERE id = ?",
            (item_id,),
        )
        if row is None:
            return None

        now = current_time or datetime.now(timezone.utc)
        last = datetime.fromisoformat(row["last_accessed_at"])
        delta_t = (now - last).total_seconds()

        score = row["base_priority"] * relevance * math.exp(-delta_t / self.TAU)
        return score

    def get_attention(self, item_id: str) -> Optional[float]:
        """Get current attention score + update the stored value."""
        score = self.compute_attention(item_id)
        if score is not None:
            self._store.execute(
                f"UPDATE {self._table} SET attention_score = ?, relevance = ? "
                f"WHERE id = ?",
                (score, 1.0, item_id),
            )
            self._store.commit()
        return score

    # ------------------------------------------------------------------
    # Proactive refresh
    # ------------------------------------------------------------------

    def scan_endangered(self) -> List[str]:
        """Return IDs of items with attention below the decay threshold."""
        now = datetime.now(timezone.utc)
        rows = self._store.fetchall(
            f"SELECT id, base_priority, last_accessed_at FROM {self._table}"
        )
        endangered = []
        for row in rows:
            last = datetime.fromisoformat(row["last_accessed_at"])
            delta_t = (now - last).total_seconds()
            score = row["base_priority"] * math.exp(-delta_t / self.TAU)
            if score < self.DECAY_THRESHOLD:
                endangered.append(row["id"])
        return endangered

    def proactive_refresh(
        self,
        context_embedding: np.ndarray,
        embedding_model: "EmbeddingModel",  # noqa: F821
    ) -> int:
        """Scan endangered items; refresh those still relevant to current context.

        A background-thread-free alternative: called at the start of memory_recall
        if > REFRESH_INTERVAL seconds have elapsed since the last refresh.

        Returns count of refreshed items.
        """
        from cell_mem.embedding.local import EmbeddingModel

        now = time.time()
        if now - self._last_refresh_time < self.REFRESH_INTERVAL:
            return 0
        self._last_refresh_time = now

        endangered = self.scan_endangered()
        if not endangered:
            return 0

        refreshed = 0
        current_time = datetime.now(timezone.utc)

        for item_id in endangered:
            row = self._store.fetchone(
                f"SELECT content FROM {self._table} WHERE id = ?", (item_id,)
            )
            if row is None:
                continue
            item_embedding = embedding_model.embed(row["content"])
            similarity = float(
                np.dot(context_embedding, item_embedding)
                / (np.linalg.norm(context_embedding) * np.linalg.norm(item_embedding) + 1e-8)
            )
            if similarity > 0.5:
                # Reset decay timer
                self._store.execute(
                    f"UPDATE {self._table} SET last_accessed_at = ?, relevance = ? "
                    f"WHERE id = ?",
                    (current_time.isoformat(), similarity, item_id),
                )
                refreshed += 1

        if refreshed:
            self._store.commit()
            logger.debug("Proactive refresh: %d/%d items saved", refreshed, len(endangered))

        return refreshed

    def should_refresh(self) -> bool:
        """Check if enough time has elapsed for a proactive refresh cycle."""
        return (time.time() - self._last_refresh_time) >= self.REFRESH_INTERVAL

    # ------------------------------------------------------------------
    # Session-end routing
    # ------------------------------------------------------------------

    def session_end_routing(self, session_id: str) -> Dict[str, List[MemoryObject]]:
        """Four-way routing of working memory items at session end.

        Returns dict with keys: keep_in_wm, downgrade_ep, store_ep, discard.
        """
        result: Dict[str, List[MemoryObject]] = {
            "keep_in_wm": [],
            "downgrade_ep": [],
            "store_ep": [],
            "discard": [],
        }

        now = datetime.now(timezone.utc)
        rows = self._store.fetchall(
            f"SELECT * FROM {self._table} WHERE session_id = ?", (session_id,)
        )

        for row in rows:
            obj = MemoryObject.from_row(dict(row))
            score = self.compute_attention(obj.id, current_time=now)

            if score is None:
                result["discard"].append(obj)
                self.remove(obj.id)
                continue

            if score > self.SESSION_END_THRESHOLD and not obj.task_completed:
                # Keep in working memory for next session
                result["keep_in_wm"].append(obj)
                # Note: decay timer NOT reset — cross-session continuity

            elif score > self.SESSION_END_THRESHOLD and obj.task_completed:
                # Downgrade to episodic with was_in_wm flag
                obj.memory_type = MemoryType.EPISODIC
                obj.was_in_wm = True
                result["downgrade_ep"].append(obj)
                self.remove(obj.id)

            elif obj.was_referenced:
                # Store complete record in episodic
                obj.memory_type = MemoryType.EPISODIC
                result["store_ep"].append(obj)
                self.remove(obj.id)

            else:
                # Pure context noise — discard
                result["discard"].append(obj)
                self.remove(obj.id)

        logger.info(
            "Session-end routing: keep=%d downgrade=%d store=%d discard=%d",
            len(result["keep_in_wm"]),
            len(result["downgrade_ep"]),
            len(result["store_ep"]),
            len(result["discard"]),
        )
        return result

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def get(self, item_id: str) -> Optional[MemoryObject]:
        row = self._store.fetchone(
            f"SELECT * FROM {self._table} WHERE id = ?", (item_id,)
        )
        if row is None:
            return None
        return MemoryObject.from_row(dict(row))

    def list_active(self) -> List[MemoryObject]:
        """All items currently in working memory."""
        rows = self._store.fetchall(f"SELECT * FROM {self._table}")
        return [MemoryObject.from_row(dict(r)) for r in rows]

    def count(self) -> int:
        row = self._store.fetchone(f"SELECT COUNT(*) as cnt FROM {self._table}")
        return row["cnt"] if row else 0

    def avg_attention(self) -> float:
        row = self._store.fetchone(
            f"SELECT AVG(attention_score) as avg FROM {self._table}"
        )
        if row and row["avg"] is not None:
            return row["avg"]
        return 0.0

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    _table = "working_memory"

    def _load_survivors(self) -> None:
        """Load items kept from previous session. Their decay timers are honored."""
        count = self.count()
        if count > 0:
            logger.info("Loaded %d surviving working memory items", count)

    def _evict_if_full(self) -> None:
        """If at capacity, evict the item with lowest attention score."""
        count = self.count()
        if count < self.MAX_ITEMS:
            return

        rows = self._store.fetchall(
            f"SELECT id FROM {self._table} ORDER BY attention_score ASC LIMIT 1"
        )
        if rows:
            victim = rows[0]["id"]
            self.remove(victim)
            logger.debug("WM evicted: %s (at capacity %d)", victim, self.MAX_ITEMS)
