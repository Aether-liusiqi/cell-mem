"""Procedural memory layer — basal-ganglia-like skill/strategy storage.

Procedural memory with two key differentiators from
existing solutions (LangMem, AgentMemory):

1. Condition-triggered: Templates are activated by cosine similarity
   between current task context and stored condition_embedding, not by
   explicit keyword search.

2. Reinforcement-learning update: Each execution outcome (success/failure)
   adjusts activation_weight via multiplicative factors. Successful
   templates strengthen; failing ones decay toward suppression.

Storage: SQLite procedural_memory table with FTS5 on template_content
and trigger_condition for explicit search.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from cell_mem.models import LifecycleStage, MemoryObject, MemoryType

logger = logging.getLogger(__name__)

# RL update factors
RL_SUCCESS_FACTOR = 1.05  # Multiply weight on success
RL_FAILURE_FACTOR = 0.85  # Multiply weight on failure
RL_WEIGHT_MAX = 1.0
RL_WEIGHT_MIN = 0.1
RL_DECAY_THRESHOLD = 0.25  # Below this, template is "decaying" (candidate for review)


class ProceduralMemory:
    """Basal-ganglia-like procedural memory layer.

    Constructor-injected dependencies:
    - sqlite_store: SqliteStore
    - embed_model: EmbeddingModel (for condition_embedding generation)
    """

    _table = "procedural_memory"

    def __init__(
        self,
        sqlite_store: "SqliteStore",  # noqa: F821
        embed_model: "EmbeddingModel",  # noqa: F821
    ):
        from cell_mem.embedding.local import EmbeddingModel
        from cell_mem.storage.sqlite_store import SqliteStore

        self._store: SqliteStore = sqlite_store
        self._embed: EmbeddingModel = embed_model
        logger.info("ProceduralMemory ready")

    # ------------------------------------------------------------------
    # CRUD
    # ------------------------------------------------------------------

    def store(
        self,
        template_content: str,
        trigger_condition: str | None = None,
        task_type: str | None = None,
        source_episode_ids: List[str] | None = None,
        metadata: dict | None = None,
    ) -> MemoryObject:
        """Create a procedural template.

        If trigger_condition is provided, it is embedded and stored as
        condition_embedding for cosine-similarity auto-triggering.
        If None, the template is only accessible via explicit keyword search.
        """
        import sqlite_vec

        now = datetime.now(timezone.utc).isoformat()

        # Embed trigger condition if provided
        cond_blob = None
        if trigger_condition:
            cond_vec = self._embed.embed(trigger_condition)
            cond_blob = sqlite_vec.serialize_float32(cond_vec.tolist())

        obj = MemoryObject(
            content=template_content,
            memory_type=MemoryType.PROCEDURAL,
            lifecycle=LifecycleStage.PLASTIC,
            created_at=now,
            metadata=metadata or {},
        )
        row = obj.to_row_dict()

        source_json = json.dumps(source_episode_ids or [], ensure_ascii=False)

        self._store.execute(
            f"""INSERT INTO {self._table} (
                id, template_content, trigger_condition, condition_embedding,
                activation_weight, success_count, failure_count,
                last_triggered_at, last_outcome_at, lifecycle,
                task_type, source_episode_ids,
                created_at, updated_at, tags_json, metadata_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                obj.id,
                template_content,
                trigger_condition,
                cond_blob,
                0.5,  # Initial activation_weight
                0,    # success_count
                0,    # failure_count
                None,  # last_triggered_at
                None,  # last_outcome_at
                "plastic",
                task_type,
                source_json,
                now,
                now,
                row["tags_json"],
                row["metadata_json"],
            ),
        )
        self._store.commit()

        logger.debug(
            "Procedural stored: %s (task_type=%s, has_condition=%s)",
            obj.id[:8], task_type, trigger_condition is not None,
        )
        return obj

    def get(self, proc_id: str) -> Optional[MemoryObject]:
        """Retrieve a procedural template by ID."""
        row = self._store.fetchone(
            f"SELECT * FROM {self._table} WHERE id = ?", (proc_id,)
        )
        if row is None:
            return None
        return self._row_to_obj(dict(row))

    def update(
        self,
        proc_id: str,
        template_content: str | None = None,
        trigger_condition: str | None = None,
        activation_weight: float | None = None,
        metadata: dict | None = None,
    ) -> Optional[MemoryObject]:
        """Update a procedural template.

        If trigger_condition changes, re-embeds and updates condition_embedding.
        """
        existing = self.get(proc_id)
        if existing is None:
            return None

        now = datetime.now(timezone.utc).isoformat()

        # Re-embed if trigger condition changed
        if trigger_condition is not None:
            import sqlite_vec

            cond_vec = self._embed.embed(trigger_condition)
            cond_blob = sqlite_vec.serialize_float32(cond_vec.tolist())
            self._store.execute(
                f"UPDATE {self._table} SET trigger_condition = ?, condition_embedding = ? WHERE id = ?",
                (trigger_condition, cond_blob, proc_id),
            )

        if template_content is not None:
            self._store.execute(
                f"UPDATE {self._table} SET template_content = ? WHERE id = ?",
                (template_content, proc_id),
            )

        if activation_weight is not None:
            weight = max(RL_WEIGHT_MIN, min(RL_WEIGHT_MAX, activation_weight))
            lifecycle = self._weight_to_lifecycle(weight)
            self._store.execute(
                f"UPDATE {self._table} SET activation_weight = ?, lifecycle = ? WHERE id = ?",
                (weight, lifecycle, proc_id),
            )

        if metadata is not None:
            meta_json = json.dumps(metadata, ensure_ascii=False)
            self._store.execute(
                f"UPDATE {self._table} SET metadata_json = ? WHERE id = ?",
                (meta_json, proc_id),
            )

        self._store.execute(
            f"UPDATE {self._table} SET updated_at = ? WHERE id = ?",
            (now, proc_id),
        )
        self._store.commit()

        return self.get(proc_id)

    def delete(self, proc_id: str) -> bool:
        """Delete a procedural template."""
        self._store.execute(
            f"DELETE FROM {self._table} WHERE id = ?", (proc_id,)
        )
        self._store.commit()
        return True

    # ------------------------------------------------------------------
    # Condition-triggered retrieval (the key differentiator)
    # ------------------------------------------------------------------

    def match_by_context(
        self,
        context_embedding: np.ndarray,
        threshold: float = 0.65,
        top_k: int = 5,
        explore_ratio: float = 0.2,
    ) -> List[Tuple[MemoryObject, float]]:
        """Cosine-similarity match between context and stored condition embeddings.

        Only returns templates with:
        - Non-NULL condition_embedding
        - activation_weight > RL_WEIGHT_MIN (not suppressed)
        - cosine similarity >= threshold

        Args:
            context_embedding: 384d embedding of the current task/situation.
            threshold: Minimum cosine similarity to trigger (default 0.65).
            top_k: Maximum number of matches to return.

        Returns:
            List of (MemoryObject, similarity_score) sorted by similarity desc.
        """
        # Load all rows with non-NULL condition_embedding and active weight
        rows = self._store.fetchall(
            f"""SELECT * FROM {self._table}
                WHERE condition_embedding IS NOT NULL
                  AND activation_weight > ?
                ORDER BY activation_weight DESC""",
            (RL_WEIGHT_MIN,),
        )

        if not rows:
            return []

        context_vec = np.asarray(context_embedding, dtype=np.float32)
        context_norm = np.linalg.norm(context_vec)
        if context_norm == 0:
            return []

        matches = []
        for row in rows:
            d = dict(row)
            try:
                cond_blob = d["condition_embedding"]
                cond_vec = np.frombuffer(cond_blob, dtype=np.float32)
                cond_norm = np.linalg.norm(cond_vec)
                if cond_norm == 0:
                    continue

                similarity = float(
                    np.dot(context_vec, cond_vec) / (context_norm * cond_norm)
                )

                if similarity >= threshold:
                    obj = self._row_to_obj(d)
                    matches.append((obj, similarity))
            except Exception as exc:
                logger.debug("Failed to compute similarity for %s: %s", d.get("id", "?"), exc)
                continue

        # Sort by similarity descending, take top_k
        matches.sort(key=lambda x: x[1], reverse=True)
        top_matches = matches[:top_k]

        # Explore/Exploit epsilon-greedy
        if len(top_matches) > 1 and explore_ratio > 0:
            import random
            if random.random() < explore_ratio:
                # Explore: pick from non-best matches, weighted by (1 - activation_weight)
                rest = top_matches[1:]
                weights = [1.0 - m.metadata.get("activation_weight", 0.5) for m, _ in rest]
                weights = [max(w, 0.01) for w in weights]
                total = sum(weights)
                probs = [w / total for w in weights]
                idx = random.choices(range(len(rest)), weights=probs, k=1)[0]
                explore_pick = rest[idx]
                # Mark as explore pick in metadata
                if not explore_pick[0].metadata.get("_explore_pick"):
                    explore_pick[0].metadata["_explore_pick"] = True
                # Return [explore_pick, best_match, ...rest excluding explore]
                result = [explore_pick, top_matches[0]]
                for i, m in enumerate(rest):
                    if i != idx:
                        result.append(m)
                return result

        return top_matches

    def match_by_keyword(self, query: str, limit: int = 5) -> List[MemoryObject]:
        """FTS5 keyword search on template_content and trigger_condition.

        Uses OR semantics: splits the query into tokens and joins with OR
        to avoid FTS5's default implicit AND which fails on multi-word queries
        where not every token appears in every document.
        """
        # Tokenize and build OR query: "token1 OR token2 OR ..."
        tokens = query.split()
        if not tokens:
            return []
        # Remove empty and single-char tokens, deduplicate
        tokens = list(dict.fromkeys(t for t in tokens if len(t) > 1))
        if not tokens:
            return []
        fts_query = " OR ".join(tokens)
        safe_query = fts_query.replace('"', '""')
        fts_sql = (
            f"SELECT rowid FROM procedural_fts "
            f"WHERE procedural_fts MATCH ? "
            f"ORDER BY rank LIMIT ?"
        )
        try:
            fts_rows = self._store.fetchall(fts_sql, (safe_query, limit))
        except Exception:
            logger.debug("FTS5 match failed for query: %s", query[:100])
            return []

        results = []
        for fts_row in (fts_rows or []):
            row = self._store.fetchone(
                f"SELECT * FROM {self._table} WHERE rowid = ?",
                (fts_row["rowid"],),
            )
            if row:
                results.append(self._row_to_obj(dict(row)))
        return results

    # ------------------------------------------------------------------
    # Reinforcement Learning Update
    # ------------------------------------------------------------------

    def record_outcome(self, proc_id: str, success: bool) -> Optional[MemoryObject]:
        """Record success/failure outcome and adjust activation_weight.

        Success: weight = min(weight * 1.05, 1.0)
        Failure: weight = max(weight * 0.85, 0.1)

        Also updates lifecycle stage based on new weight.
        """
        existing = self.get(proc_id)
        if existing is None:
            logger.warning("record_outcome: template not found: %s", proc_id[:8])
            return None

        now = datetime.now(timezone.utc).isoformat()

        # Read current weight from metadata since MemoryObject doesn't have it
        row = self._store.fetchone(
            f"SELECT activation_weight, success_count, failure_count FROM {self._table} WHERE id = ?",
            (proc_id,),
        )
        if row is None:
            return None

        old_weight = row["activation_weight"]
        s_count = row["success_count"]
        f_count = row["failure_count"]

        if success:
            new_weight = min(old_weight * RL_SUCCESS_FACTOR, RL_WEIGHT_MAX)
            s_count += 1
        else:
            new_weight = max(old_weight * RL_FAILURE_FACTOR, RL_WEIGHT_MIN)
            f_count += 1

        lifecycle = self._weight_to_lifecycle(new_weight)

        self._store.execute(
            f"""UPDATE {self._table}
                SET activation_weight = ?, success_count = ?, failure_count = ?,
                    last_triggered_at = ?, last_outcome_at = ?, lifecycle = ?,
                    updated_at = ?
                WHERE id = ?""",
            (new_weight, s_count, f_count, now, now, lifecycle, now, proc_id),
        )
        self._store.commit()

        logger.debug(
            "Outcome for %s: success=%s weight %.3f→%.3f lifecycle=%s",
            proc_id[:8], success, old_weight, new_weight, lifecycle,
        )
        return self.get(proc_id)

    # ------------------------------------------------------------------
    # Bulk access
    # ------------------------------------------------------------------

    def load_all(
        self, limit: int = 1000, offset: int = 0
    ) -> List[MemoryObject]:
        """Load procedural templates in batches (for inspection/consolidation)."""
        rows = self._store.fetchall(
            f"SELECT * FROM {self._table} ORDER BY activation_weight DESC LIMIT ? OFFSET ?",
            (limit, offset),
        )
        return [self._row_to_obj(dict(r)) for r in rows]

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def count(self) -> int:
        row = self._store.fetchone(
            f"SELECT COUNT(*) as cnt FROM {self._table}"
        )
        return row["cnt"] if row else 0

    def count_by_lifecycle(self) -> Dict[str, int]:
        rows = self._store.fetchall(
            f"SELECT lifecycle, COUNT(*) as cnt FROM {self._table} GROUP BY lifecycle"
        )
        return {r["lifecycle"]: r["cnt"] for r in rows}

    def avg_activation_weight(self) -> float:
        row = self._store.fetchone(
            f"SELECT AVG(activation_weight) as avg FROM {self._table}"
        )
        if row and row["avg"] is not None:
            return row["avg"]
        return 0.0

    def get_decaying(self, threshold: float = RL_DECAY_THRESHOLD) -> List[MemoryObject]:
        """Return templates with activation_weight below threshold.

        These are candidates for reflection review (repeated failures).
        """
        rows = self._store.fetchall(
            f"SELECT * FROM {self._table} WHERE activation_weight < ? ORDER BY activation_weight ASC",
            (threshold,),
        )
        return [self._row_to_obj(dict(r)) for r in rows]

    def count_with_conditions(self) -> int:
        """Count templates that have auto-trigger conditions."""
        row = self._store.fetchone(
            f"SELECT COUNT(*) as cnt FROM {self._table} WHERE condition_embedding IS NOT NULL"
        )
        return row["cnt"] if row else 0

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _row_to_obj(self, row: Dict[str, Any]) -> MemoryObject:
        """Convert a procedural_memory row to a MemoryObject.

        Maps the procedural-specific columns into the canonical MemoryObject:
        - template_content → content
        - activation_weight → metadata["activation_weight"]
        - success_count → metadata["success_count"]
        - etc.
        """
        # Start with standard from_row conversion
        # Map template_content → content for MemoryObject compatibility
        content = row.pop("template_content", row.get("content", ""))

        # Extract procedural-specific fields into metadata
        proc_meta = {
            "activation_weight": row.get("activation_weight", 0.5),
            "success_count": row.get("success_count", 0),
            "failure_count": row.get("failure_count", 0),
            "trigger_condition": row.get("trigger_condition"),
            "task_type": row.get("task_type"),
            "source_episode_ids": json.loads(row.get("source_episode_ids", "[]")),
            "last_triggered_at": row.get("last_triggered_at"),
            "last_outcome_at": row.get("last_outcome_at"),
        }

        # Merge with existing metadata_json
        meta_json = row.get("metadata_json", "{}")
        try:
            existing_meta = json.loads(meta_json) if isinstance(meta_json, str) else meta_json
        except (json.JSONDecodeError, TypeError):
            existing_meta = {}
        existing_meta.update(proc_meta)

        # Build a pseudo-row that MemoryObject.from_row can handle
        pseudo = {
            "id": row.get("id"),
            "content": content,
            "memory_type": "procedural",
            "lifecycle": row.get("lifecycle", "plastic"),
            "created_at": row.get("created_at", ""),
            "tags_json": row.get("tags_json", "[]"),
            "source_refs_json": json.dumps(proc_meta.get("source_episode_ids", [])),
            "metadata_json": json.dumps(existing_meta, ensure_ascii=False),
            "confidence": 0.0,
            "valence": 0.0,
            "consolidation_score": 0.0,
        }
        return MemoryObject.from_row(pseudo)

    @staticmethod
    def _weight_to_lifecycle(weight: float) -> str:
        """Map activation weight to lifecycle stage."""
        if weight >= 0.9:
            return "locked"
        elif weight >= 0.7:
            return "semi_stable"
        else:
            return "plastic"
