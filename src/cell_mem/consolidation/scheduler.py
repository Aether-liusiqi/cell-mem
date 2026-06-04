"""Consolidation scheduler — periodic consolidation cycles.

Orchestrates the full consolidation cycle:
1. Score all episodes via ConsolidationScorer
2. Identify forget candidates (below dynamic threshold)
3. Archive episodes with 3+ consecutive low-score cycles to cold storage
4. Run pattern detection via DBSCAN
5. Persist cycle state to meta table

Trigger: manual (memory_consolidate MCP tool) or lazy should_run() check.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Cycle trigger thresholds
MIN_CYCLE_INTERVAL = 300  # 5 minutes between cycles
COUNT_GROWTH_THRESHOLD = 0.20  # 20% new episodes since last cycle
MIN_EPISODES_FOR_CYCLE = 10  # Minimum episodes before consolidation triggers

# Forgetting
FORGET_CANDIDATE_CYCLES = 3  # Consecutive cycles below threshold → archive


class ConsolidationScheduler:
    """Orchestrates consolidation cycles: score → detect → forget.

    Constructor-injected dependencies:
    - episodic: EpisodicMemory
    - semantic: SemanticMemory
    - scorer: ConsolidationScorer
    - detector: PatternDetector
    - store: SqliteStore (for cold_storage table)
    """

    def __init__(
        self,
        episodic: Optional["EpisodicMemory"] = None,  # noqa: F821
        semantic: Optional["SemanticMemory"] = None,  # noqa: F821
        scorer: Optional["ConsolidationScorer"] = None,  # noqa: F821
        detector: Optional["PatternDetector"] = None,  # noqa: F821
        store: Optional["SqliteStore"] = None,  # noqa: F821
    ):
        from cell_mem.consolidation.detector import PatternDetector
        from cell_mem.consolidation.scorer import ConsolidationScorer
        from cell_mem.memory.episodic import EpisodicMemory
        from cell_mem.memory.semantic import SemanticMemory
        from cell_mem.storage.sqlite_store import SqliteStore

        self._episodic: Optional[EpisodicMemory] = episodic
        self._semantic: Optional[SemanticMemory] = semantic
        self._scorer: Optional[ConsolidationScorer] = scorer
        self._detector: Optional[PatternDetector] = detector
        self._store: Optional[SqliteStore] = store

        # Cycle tracking — restore from DB if available
        self._last_cycle_at: Optional[str] = None
        self._last_cycle_count: int = 0
        self._cycle_count: int = 0
        self._restore_state()

    def _restore_state(self) -> None:
        """Restore cycle state from meta table (survives server restart)."""
        if self._store is None:
            return
        try:
            raw_at = self._store.get_meta("consol_last_at")
            if raw_at:
                self._last_cycle_at = raw_at.decode("utf-8")
            raw_count = self._store.get_meta("consol_last_count")
            if raw_count:
                self._last_cycle_count = int(raw_count.decode("utf-8"))
            raw_cycle = self._store.get_meta("consol_cycle_count")
            if raw_cycle:
                self._cycle_count = int(raw_cycle.decode("utf-8"))
            if self._last_cycle_at:
                logger.debug(
                    "Restored consolidation state: cycle=%d, count=%d, last=%s",
                    self._cycle_count, self._last_cycle_count, self._last_cycle_at,
                )
        except Exception as exc:
            logger.warning("Could not restore consolidation state from meta table: %s", exc)

    # ------------------------------------------------------------------
    # Trigger logic
    # ------------------------------------------------------------------

    def should_run(self) -> bool:
        """Check if a consolidation cycle should run.

        Conditions (all must be met):
        1. At least MIN_EPISODES_FOR_CYCLE episodes exist.
        2. At least MIN_CYCLE_INTERVAL since last cycle.
        3. Episode count grown by COUNT_GROWTH_THRESHOLD since last cycle
           (or no cycle has ever run).
        """
        if self._episodic is None:
            return False

        current = self._episodic.count()
        if current < MIN_EPISODES_FOR_CYCLE:
            return False

        # First cycle
        if self._last_cycle_at is None:
            return True

        # Time check
        try:
            last = datetime.fromisoformat(self._last_cycle_at)
            elapsed = (datetime.now(timezone.utc) - last).total_seconds()
        except (ValueError, TypeError):
            elapsed = MIN_CYCLE_INTERVAL + 1

        if elapsed < MIN_CYCLE_INTERVAL:
            return False

        # Growth check
        if self._last_cycle_count > 0:
            growth = (current - self._last_cycle_count) / self._last_cycle_count
            return growth >= COUNT_GROWTH_THRESHOLD

        return True

    # ------------------------------------------------------------------
    # Consolidation cycle
    # ------------------------------------------------------------------

    def run_cycle(self, batch_size: int = 100) -> dict:
        """Run a full consolidation cycle.

        Steps:
        1. Score all episodes in batches, write consolidation_score back.
        2. Identify forget candidates (score < dynamic threshold).
        3. Increment forget_candidate_count in metadata.
        4. Archive episodes with >= FORGET_CANDIDATE_CYCLES consecutive cycles.
        5. Run pattern detection on current episodes.
        6. Persist cycle state.

        Returns:
            {"status": "ok", "data": {stats...}} or error dict.
        """
        if self._episodic is None or self._scorer is None or self._store is None:
            return {"status": "error", "error": "Scheduler not fully configured"}

        try:
            total = self._episodic.count()
            if total == 0:
                return {"status": "ok", "data": {"note": "No episodes to consolidate"}}

            threshold = self._scorer.get_forget_threshold(total)
            scored = 0
            candidates = 0
            archived = 0

            # Step 1-4: Score, mark candidates, archive
            offset = 0
            while True:
                batch = self._episodic.load_all(limit=batch_size, offset=offset)
                if not batch:
                    break

                for ep in batch:
                    score = self._scorer.score(ep)
                    scored += 1

                    # Write score back
                    self._episodic.update(ep.id, consolidation_score=score)

                    # Below threshold → forget candidate
                    if score < threshold:
                        candidates += 1
                        fc = ep.metadata.get("_forget_candidate_count", 0) + 1
                        new_meta = {**ep.metadata, "_forget_candidate_count": fc}
                        self._episodic.update(ep.id, metadata=new_meta)

                        # 3+ consecutive cycles → cold storage
                        if fc >= FORGET_CANDIDATE_CYCLES:
                            self._archive_to_cold(ep, score)
                            archived += 1

                offset += batch_size

            # Step 5: Pattern detection
            patterns = 0
            if self._detector is not None:
                try:
                    recent = self._episodic.load_all(limit=batch_size)
                    proposals = self._detector.detect(recent)
                    patterns = len(proposals)
                except Exception as exc:
                    logger.warning("Pattern detection failed (non-fatal): %s", exc)

            # Step 6: Persist state
            now = datetime.now(timezone.utc).isoformat()
            self._last_cycle_at = now
            self._last_cycle_count = total
            self._cycle_count += 1

            self._store.set_meta("consol_last_at", now.encode("utf-8"))
            self._store.set_meta("consol_last_count", str(total).encode("utf-8"))
            self._store.set_meta("consol_cycle_count", str(self._cycle_count).encode("utf-8"))

            logger.info(
                "Cycle #%d: scored=%d candidates=%d archived=%d patterns=%d threshold=%.2f",
                self._cycle_count, scored, candidates, archived, patterns, threshold,
            )

            return {
                "status": "ok",
                "data": {
                    "cycle_number": self._cycle_count,
                    "total_scored": scored,
                    "forget_candidates": candidates,
                    "archived": archived,
                    "patterns_detected": patterns,
                    "forget_threshold": round(threshold, 3),
                    "completed_at": now,
                },
            }

        except Exception as exc:
            logger.exception("Consolidation cycle failed: %s", exc)
            return {"status": "error", "error": str(exc)}

    # ------------------------------------------------------------------
    # Cold storage archival
    # ------------------------------------------------------------------

    def _archive_to_cold(self, episode, score: float) -> None:
        """Archive an episode to cold_storage, then delete from episodic."""
        if self._store is None or self._episodic is None:
            return

        summary = episode.content[:200]
        now = datetime.now(timezone.utc).isoformat()

        self._store.execute(
            """INSERT INTO cold_storage
               (id, original_id, content, summary, original_type,
                compressed_at, retrieval_count, score_at_archive)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (episode.id, episode.id, episode.content, summary,
             "episodic", now, 0, score),
        )
        self._store.commit()

        self._episodic.delete(episode.id)
        logger.debug("Archived %s → cold_storage (score=%.3f)", episode.id[:8], score)

    # ------------------------------------------------------------------
    # Rescue operations
    # ------------------------------------------------------------------

    def rescue_episode(self, episode_id: str, boost: float = 1.0) -> bool:
        """Bump score and reset forget_candidate_count.

        Called when user retrieves a candidate or marks it important.
        boost=1.0 for retrieval hit, boost=2.0 for user-marked important.
        """
        if self._episodic is None:
            return False

        ep = self._episodic.get(episode_id)
        if ep is None:
            return False

        new_score = ep.consolidation_score + boost
        new_meta = {**ep.metadata, "_forget_candidate_count": 0}
        self._episodic.update(ep.id, consolidation_score=new_score, metadata=new_meta)
        logger.info("Rescued %s (score +%.1f → %.3f)", episode_id[:8], boost, new_score)
        return True

    def get_cold_storage_count(self) -> int:
        """Return number of entries in cold storage."""
        if self._store is None:
            return 0
        return self._store.row_count("cold_storage")
