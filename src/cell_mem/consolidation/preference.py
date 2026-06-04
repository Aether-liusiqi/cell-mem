"""User Preference Pipeline — automatic preference extraction, processing, and injection.

Five-stage closed loop that reuses existing Cell-mem infrastructure:
  Hook A: Signal detection on every episodic save
  Hook B: Preference extraction during consolidation cycles
  Hook C: Conflict detection + confidence management during reflection
  Hook D: Context-triggered preference injection via procedural memory
  Hook E: Implicit feedback via record_outcome (no manual MCP call needed)

Fully automatic — no user intervention required once deployed.
"""

from __future__ import annotations

import json
import logging
import re
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Confidence thresholds
PREFERENCE_PROMOTE_THRESHOLD = 0.6   # Promote to confirmed
PREFERENCE_REJECT_THRESHOLD = 0.2   # Drop to rejected
PREFERENCE_DECAY_RATE = 0.03        # Per cycle without fresh evidence
PREFERENCE_CONFLICT_SIMILARITY = 0.85  # Cosine threshold for conflict check

# Injection limits (prevent context bloat)
MAX_INJECT_PER_RECALL = 3          # Max preferences injected per recall
MAX_INJECT_TOKENS = 300            # Max chars in injected preference block
MIN_INJECT_INTERVAL_SEC = 300      # No re-injection within 5 minutes
INJECT_CONFIDENCE_MIN = 0.5        # Only inject preferences with confidence >= 0.5

# Feedback from procedural outcome (passive, no MCP call needed)
PREFERENCE_SUCCESS_BOOST = 0.03    # +0.03 when procedural template used successfully
PREFERENCE_FAILURE_PENALTY = 0.05  # -0.05 when template used but failed

# Signal detection patterns (no LLM required)
_PREFERENCE_KEYWORDS: Dict[str, List[str]] = {
    "tool_choice": [
        "prefer", "like using", "always use", "never use", "hate",
        "my go-to", "favorite tool", "best editor", "best IDE",
    ],
    "workflow": [
        "my workflow", "my process", "I usually", "how I work",
        "my setup", "pipeline", "my routine", "each time I",
    ],
    "communication_style": [
        "explain", "show me", "just do it", "keep it short",
        "verbose", "step by step", "summarize", "give me details",
        "brief", "concise", "in depth",
    ],
    "skill_level": [
        "I know", "familiar with", "new to", "learning",
        "expert in", "experienced with", "beginner", "advanced",
    ],
}

# Patterns that indicate NEGATIVE preference (dislike/avoid)
_NEGATION_PATTERNS = [
    r"\bdon't\s+\w+\s+to\b", r"\bcan't stand\b", r"\bhate\b",
    r"\bnever\s+\w+\s+(again|more|like)\b", r"\bavoid\b",
    r"\bnot\s+a\s+fan\b", r"\bdon't\s+like\b",
]


# ---------------------------------------------------------------------------
# Data class for preference signals
# ---------------------------------------------------------------------------

class PreferenceSignal:
    """Lightweight signal detected from a single interaction."""

    __slots__ = (
        "source_episode_id", "preference_type", "matched_text",
        "is_negative", "strength", "context",
    )

    def __init__(
        self,
        source_episode_id: str,
        preference_type: str,
        matched_text: str,
        is_negative: bool = False,
        strength: float = 0.5,
        context: str = "",
    ):
        self.source_episode_id = source_episode_id
        self.preference_type = preference_type
        self.matched_text = matched_text
        self.is_negative = is_negative
        self.strength = strength
        self.context = context

    def to_dict(self) -> dict:
        return {
            "source_episode_id": self.source_episode_id,
            "preference_type": self.preference_type,
            "matched_text": self.matched_text,
            "is_negative": self.is_negative,
            "strength": self.strength,
            "context": self.context[:200],
        }


# ---------------------------------------------------------------------------
# Hook A: Signal Detection (runs on every episodic save)
# ---------------------------------------------------------------------------

class PreferenceSignalDetector:
    """Detect implicit preference signals from agent interaction text.

    Uses keyword patterns + negation detection + emotional valence weighting.
    Zero LLM dependency — fast enough to run on every save().
    """

    def __init__(self):
        self._negation_re = re.compile("|".join(_NEGATION_PATTERNS), re.IGNORECASE)

    def detect(self, content: str, episode_id: str = "", valence: float = 0.0) -> List[PreferenceSignal]:
        """Scan a single interaction for preference signals.

        Args:
            content: The interaction text to scan.
            episode_id: Source episode ID for traceability.
            valence: Emotional valence from episodic store (-1 to 1).
                     High magnitude content gets weighted higher.

        Returns:
            List of detected PreferenceSignals (may be empty).
        """
        if not content or len(content) < 10:
            return []

        content_lower = content.lower()
        signals: List[PreferenceSignal] = []

        for pref_type, keywords in _PREFERENCE_KEYWORDS.items():
            for kw in keywords:
                idx = content_lower.find(kw.lower())
                if idx == -1:
                    continue

                # Extract surrounding context (50 chars each side)
                start = max(0, idx - 50)
                end = min(len(content), idx + len(kw) + 50)
                snippet = content[start:end].strip()

                # Check for negation
                is_negative = bool(self._negation_re.search(snippet))

                # Strength: base + emotional magnitude boost
                strength = 0.5
                if abs(valence) > 0.3:
                    strength += abs(valence) * 0.3  # Emotional content = stronger signal
                strength = min(strength, 1.0)

                signals.append(PreferenceSignal(
                    source_episode_id=episode_id,
                    preference_type=pref_type,
                    matched_text=snippet[:200],
                    is_negative=is_negative,
                    strength=strength,
                    context=content[:300],
                ))
                break  # One signal per type per content

        return signals

    def batch_detect(
        self, episodes: List[Any],  # List[MemoryObject]
    ) -> List[PreferenceSignal]:
        """Scan a batch of episodes for preference signals.

        Args:
            episodes: List of MemoryObject from episodic memory.

        Returns:
            Aggregated list of all detected signals.
        """
        all_signals: List[PreferenceSignal] = []
        for ep in episodes:
            valence = getattr(ep, "valence", 0.0) or 0.0
            signals = self.detect(
                content=ep.content,
                episode_id=ep.id,
                valence=valence,
            )
            all_signals.extend(signals)
        return all_signals

    def has_signals(self, content: str) -> bool:
        """Quick check: does this content contain any preference keywords?"""
        if not content:
            return False
        content_lower = content.lower()
        return any(
            kw.lower() in content_lower
            for kws in _PREFERENCE_KEYWORDS.values()
            for kw in kws
        )


# ---------------------------------------------------------------------------
# Hook B: Preference Extraction (runs during consolidation cycle)
# ---------------------------------------------------------------------------

class PreferenceExtractor:
    """Aggregate signals across episodes into preference candidates.

    LLM-powered extraction with keyword-frequency fallback.
    Reuses: LLMClient, EmbeddingModel, SemanticMemory (for dedup).
    """

    def __init__(
        self,
        store: "SqliteStore",          # noqa: F821
        embed_model: "EmbeddingModel",  # noqa: F821
        semantic: "SemanticMemory",     # noqa: F821
        llm_client: Optional[Any] = None,
    ):
        from cell_mem.embedding.local import EmbeddingModel
        from cell_mem.memory.semantic import SemanticMemory
        from cell_mem.storage.sqlite_store import SqliteStore

        self._store: SqliteStore = store
        self._embed: EmbeddingModel = embed_model
        self._semantic: SemanticMemory = semantic
        self._llm = llm_client

    def extract(
        self,
        signals: List[PreferenceSignal],
        recent_episodes: Optional[List[Any]] = None,
    ) -> List[dict]:
        """Extract preference candidates from detected signals.

        If LLM is available, uses LLM for structured extraction.
        Falls back to keyword-frequency heuristics.
        """
        if not signals:
            return []

        if self._llm is not None:
            return self._llm_extract(signals, recent_episodes or [])
        else:
            return self._heuristic_extract(signals)

    def _llm_extract(
        self, signals: List[PreferenceSignal], episodes: List[Any],
    ) -> List[dict]:
        """LLM-powered structured preference extraction."""
        # Build signal summary
        signal_texts = []
        for s in signals[:20]:  # Limit to 20 signals
            neg = "(NEGATIVE) " if s.is_negative else ""
            signal_texts.append(
                f"[{s.preference_type}] {neg}{s.matched_text[:150]}"
            )

        episode_context = ""
        if episodes:
            episode_texts = [ep.content[:200] for ep in episodes[:5]]
            episode_context = "\n".join(f"- {t}" for t in episode_texts)

        prompt = (
            "Analyze these user interaction signals and extract distinct user preferences. "
            "Group related signals. Identify preferences about tools, workflows, "
            "communication style, and skill level.\n\n"
            f"## Detected Signals\n{chr(10).join(signal_texts)}\n\n"
            f"## Recent Interactions\n{episode_context}\n\n"
            "Return a JSON array of preference objects, each with:\n"
            '- preference_text: concise statement of the preference (e.g., "prefers short responses")\n'
            '- preference_type: one of tool_choice, workflow, communication_style, skill_level\n'
            '- confidence: 0.0 to 1.0 based on signal strength and frequency\n'
            '- trigger_context: when this preference should be activated (or null)\n'
        )

        schema_hint = {
            "preferences": "array — each with preference_text (string), "
                           "preference_type (string), confidence (float), "
                           "trigger_context (string|null)",
        }

        try:
            result = self._llm.generate(prompt, schema_hint)
            prefs = result.get("preferences", [])
            if isinstance(prefs, list) and prefs:
                logger.info("LLM extracted %d preference candidates", len(prefs))
                return prefs
        except Exception as exc:
            logger.warning("LLM preference extraction failed, falling back to heuristic: %s", exc)

        return self._heuristic_extract(signals)

    def _heuristic_extract(self, signals: List[PreferenceSignal]) -> List[dict]:
        """Keyword-frequency based extraction (no LLM)."""
        # Group signals by type
        by_type: Dict[str, List[PreferenceSignal]] = {}
        for s in signals:
            by_type.setdefault(s.preference_type, []).append(s)

        candidates = []
        for pref_type, type_signals in by_type.items():
            if len(type_signals) < 2:
                continue  # Need at least 2 signals to form a candidate

            # Average strength
            avg_strength = sum(s.strength for s in type_signals) / len(type_signals)
            neg_count = sum(1 for s in type_signals if s.is_negative)

            # Build preference text from most frequent matched text
            text_samples = list(set(s.matched_text[:100] for s in type_signals))
            summary = text_samples[0] if text_samples else type_signals[0].matched_text[:100]

            candidates.append({
                "preference_text": f"User shows {pref_type} preference: {summary}",
                "preference_type": pref_type,
                "confidence": min(avg_strength, 0.5),
                "trigger_context": None,
                "_signal_count": len(type_signals),
                "_negative_count": neg_count,
            })

        logger.info("Heuristic extraction produced %d candidates from %d signals",
                     len(candidates), len(signals))
        return candidates

    def deduplicate(self, candidates: List[dict]) -> List[dict]:
        """Merge semantically similar candidates via cosine similarity."""
        if len(candidates) <= 1:
            return candidates

        # Embed all candidate texts
        texts = [c.get("preference_text", "") for c in candidates]
        embeddings = [self._embed.embed(t) for t in texts]

        merged = []
        used = set()
        for i, ci in enumerate(candidates):
            if i in used:
                continue
            for j in range(i + 1, len(candidates)):
                if j in used:
                    continue
                sim = float(np.dot(embeddings[i], embeddings[j]) /
                           (np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[j]) + 1e-10))
                if sim > 0.85:
                    # Merge: average confidence, keep higher signal count
                    ci["confidence"] = (ci["confidence"] + candidates[j]["confidence"]) / 2
                    ci["_signal_count"] = ci.get("_signal_count", 1) + candidates[j].get("_signal_count", 1)
                    used.add(j)
            merged.append(ci)

        logger.debug("Dedup: %d → %d candidates", len(candidates), len(merged))
        return merged

    def _check_existing(self, preference_text: str) -> Optional[str]:
        """Check if a semantically similar preference already exists in semantic memory."""
        try:
            emb = self._embed.embed(preference_text)
            existing = self._semantic.search(emb, top_k=3)
            for obj, score in [(e, 0.0) for e in existing] if isinstance(existing, list) else []:
                if hasattr(obj, 'metadata') and obj.metadata.get("preference_type"):
                    return obj.id
        except Exception:
            pass
        return None


# ---------------------------------------------------------------------------
# Hook C: Preference Processing (runs during reflection/consolidation)
# ---------------------------------------------------------------------------

class PreferenceProcessor:
    """Process preference candidates: conflict detection, confidence lifecycle,
    environment verification, promotion to semantic memory.

    Reuses: ConditionEvaluator, SemanticMemory, SqliteStore.
    """

    def __init__(
        self,
        store: "SqliteStore",              # noqa: F821
        semantic: "SemanticMemory",         # noqa: F821
        condition_eval: Optional[Any] = None,  # ConditionEvaluator
        embed_model: Optional["EmbeddingModel"] = None,  # noqa: F821
    ):
        from cell_mem.embedding.local import EmbeddingModel
        from cell_mem.memory.semantic import SemanticMemory
        from cell_mem.storage.sqlite_store import SqliteStore

        self._store: SqliteStore = store
        self._semantic: SemanticMemory = semantic
        self._condition_eval = condition_eval
        self._embed: Optional[EmbeddingModel] = embed_model

    # ------------------------------------------------------------------
    # CRUD
    # ------------------------------------------------------------------

    def add_candidate(self, candidate: dict) -> str:
        """Insert a preference candidate into preference_candidates table."""
        import uuid

        now = datetime.now(timezone.utc).isoformat()
        pref_id = uuid.uuid4().hex[:16]

        self._store.execute(
            """INSERT INTO preference_candidates (
                id, preference_text, preference_type, confidence,
                source_episode_ids, signal_strength, status,
                trigger_context, lifecycle, created_at, updated_at, metadata_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                pref_id,
                candidate.get("preference_text", ""),
                candidate.get("preference_type", "general"),
                candidate.get("confidence", 0.3),
                json.dumps(candidate.get("source_episode_ids", [])),
                candidate.get("signal_strength", candidate.get("_signal_count", 1) * 0.15),
                "pending",
                candidate.get("trigger_context"),
                "plastic",
                now,
                now,
                json.dumps(candidate.get("metadata", {}), ensure_ascii=False),
            ),
        )
        self._store.commit()
        logger.debug("Preference candidate added: %s (type=%s)", pref_id[:8],
                     candidate.get("preference_type"))
        return pref_id

    def get_candidate(self, pref_id: str) -> Optional[dict]:
        """Get a preference candidate by ID."""
        row = self._store.fetchone(
            "SELECT * FROM preference_candidates WHERE id = ?", (pref_id,)
        )
        return dict(row) if row else None

    def get_pending(self, limit: int = 20) -> List[dict]:
        """Get pending candidates for processing."""
        rows = self._store.fetchall(
            "SELECT * FROM preference_candidates WHERE status = 'pending' "
            "ORDER BY signal_strength DESC LIMIT ?",
            (limit,),
        )
        return [dict(r) for r in rows]

    def get_confirmed(self, limit: int = 50) -> List[dict]:
        """Get confirmed preferences (confidence >= 0.6)."""
        rows = self._store.fetchall(
            "SELECT * FROM preference_candidates WHERE status = 'confirmed' "
            "AND confidence >= ? ORDER BY confidence DESC LIMIT ?",
            (PREFERENCE_PROMOTE_THRESHOLD, limit),
        )
        return [dict(r) for r in rows]

    def get_by_type(self, pref_type: str, min_confidence: float = 0.3) -> List[dict]:
        """Get preferences of a specific type."""
        rows = self._store.fetchall(
            "SELECT * FROM preference_candidates WHERE preference_type = ? "
            "AND confidence >= ? AND status != 'rejected' "
            "ORDER BY confidence DESC",
            (pref_type, min_confidence),
        )
        return [dict(r) for r in rows]

    # ------------------------------------------------------------------
    # Conflict Detection (Dim 5 of reflection)
    # ------------------------------------------------------------------

    def detect_conflicts(self) -> List[dict]:
        """Find contradictory preference pairs.

        Uses cosine similarity on embedded preference_text to find similar
        preferences, then checks for contradictory content via negation patterns.

        Returns:
            List of conflict dicts with {pref_a, pref_b, similarity, resolution_hint}.
        """
        confirmed = self.get_confirmed(limit=50)
        if len(confirmed) < 2:
            return []

        if self._embed is None:
            return []

        conflicts = []
        texts = [c["preference_text"] for c in confirmed]
        embs = [self._embed.embed(t) for t in texts]

        for i in range(len(confirmed)):
            for j in range(i + 1, len(confirmed)):
                # Only check same-type pairs
                if confirmed[i]["preference_type"] != confirmed[j]["preference_type"]:
                    continue

                sim = float(
                    np.dot(embs[i], embs[j]) /
                    (np.linalg.norm(embs[i]) * np.linalg.norm(embs[j]) + 1e-10)
                )

                if sim < PREFERENCE_CONFLICT_SIMILARITY:
                    continue

                # Check for contradiction: one positive, one negative framing
                ti = confirmed[i]["preference_text"].lower()
                tj = confirmed[j]["preference_text"].lower()
                neg_patterns = ["don't", "never", "not", "avoid", "hate", "dislike"]

                has_neg_i = any(p in ti for p in neg_patterns)
                has_neg_j = any(p in tj for p in neg_patterns)

                if has_neg_i != has_neg_j:
                    conflicts.append({
                        "pref_a_id": confirmed[i]["id"],
                        "pref_b_id": confirmed[j]["id"],
                        "pref_a_text": confirmed[i]["preference_text"],
                        "pref_b_text": confirmed[j]["preference_text"],
                        "similarity": round(sim, 3),
                        "resolution_hint": (
                            f"Context-dependent: '{confirmed[i]['preference_text']}' "
                            f"vs '{confirmed[j]['preference_text']}' — "
                            f"may depend on task type or situation"
                        ),
                    })

                    # Record conflict in both candidates
                    self._record_conflict(
                        confirmed[i]["id"], confirmed[j]["id"],
                        confirmed[i]["confidence"], confirmed[j]["confidence"],
                    )

        if conflicts:
            logger.info("Detected %d preference conflicts", len(conflicts))
        return conflicts

    def _record_conflict(
        self, id_a: str, id_b: str, conf_a: float, conf_b: float,
    ) -> None:
        """Record conflict and lower confidence of the weaker preference."""
        existing_a = self._store.fetchone(
            "SELECT conflict_with FROM preference_candidates WHERE id = ?", (id_a,)
        )
        existing_b = self._store.fetchone(
            "SELECT conflict_with FROM preference_candidates WHERE id = ?", (id_b,)
        )

        # Update conflict_with field
        for pref_id, other_id, existing_row in [
            (id_a, id_b, existing_a),
            (id_b, id_a, existing_b),
        ]:
            if existing_row is None:
                continue
            current = existing_row["conflict_with"]
            existing_list = json.loads(current) if current else []
            if other_id not in existing_list:
                existing_list.append(other_id)
                self._store.execute(
                    "UPDATE preference_candidates SET conflict_with = ? WHERE id = ?",
                    (json.dumps(existing_list), pref_id),
                )

        # Lower confidence of the weaker preference
        if conf_a < conf_b:
            new_conf = max(conf_a - 0.05, PREFERENCE_REJECT_THRESHOLD)
            self._store.execute(
                "UPDATE preference_candidates SET confidence = ? WHERE id = ?",
                (new_conf, id_a),
            )
        else:
            new_conf = max(conf_b - 0.05, PREFERENCE_REJECT_THRESHOLD)
            self._store.execute(
                "UPDATE preference_candidates SET confidence = ? WHERE id = ?",
                (new_conf, id_b),
            )
        self._store.commit()

    # ------------------------------------------------------------------
    # Confidence Lifecycle
    # ------------------------------------------------------------------

    def update_confidence(self, pref_id: str, delta: float) -> Optional[float]:
        """Adjust preference confidence and handle lifecycle transitions.

        confidence >= 0.6 → promote to semantic memory
        confidence < 0.2  → mark as rejected
        """
        row = self._store.fetchone(
            "SELECT confidence, status FROM preference_candidates WHERE id = ?",
            (pref_id,),
        )
        if row is None:
            return None

        old_conf = row["confidence"]
        new_conf = max(0.0, min(1.0, old_conf + delta))
        new_status = row["status"]

        if new_conf >= PREFERENCE_PROMOTE_THRESHOLD and row["status"] != "promoted":
            new_status = "confirmed"
            self._promote_to_semantic(pref_id)
        elif new_conf < PREFERENCE_REJECT_THRESHOLD:
            new_status = "rejected"

        now = datetime.now(timezone.utc).isoformat()
        self._store.execute(
            """UPDATE preference_candidates
               SET confidence = ?, status = ?, lifecycle = ?, updated_at = ?
               WHERE id = ?""",
            (new_conf, new_status,
             "locked" if new_conf >= 0.9 else ("semi_stable" if new_conf >= 0.7 else "plastic"),
             now, pref_id),
        )
        self._store.commit()

        logger.debug("Preference %s confidence: %.2f → %.2f (delta=%+.2f, status=%s)",
                     pref_id[:8], old_conf, new_conf, delta, new_status)
        return new_conf

    def apply_decay(self) -> int:
        """Apply decay to all pending/confirmed preferences without recent evidence.

        Called during consolidation cycle. Returns count of decayed preferences.
        """
        rows = self._store.fetchall(
            """SELECT id, confidence, updated_at, status FROM preference_candidates
               WHERE status IN ('pending', 'confirmed')"""
        )

        decayed = 0
        now = datetime.now(timezone.utc)
        for row in rows:
            try:
                updated = datetime.fromisoformat(row["updated_at"])
                days_since = (now - updated).days
            except (ValueError, TypeError):
                days_since = 7

            # Decay proportional to days without update (max 0.15)
            decay = min(days_since * PREFERENCE_DECAY_RATE, 0.15)
            if decay < 0.01:
                continue

            self.update_confidence(row["id"], -decay)
            decayed += 1

        if decayed:
            logger.info("Applied decay to %d preferences", decayed)
        return decayed

    def _promote_to_semantic(self, pref_id: str) -> None:
        """Promote a confirmed preference to semantic memory.

        Creates a semantic entry with the preference as content and
        a falsifiable condition if one exists on the candidate.
        """
        candidate = self.get_candidate(pref_id)
        if candidate is None:
            return

        # Check if already promoted
        existing = self._store.fetchone(
            "SELECT status FROM preference_candidates WHERE id = ?", (pref_id,)
        )
        if existing and existing["status"] == "promoted":
            return

        try:
            falsifiable = None
            if candidate.get("falsifiable_condition"):
                try:
                    falsifiable = json.loads(candidate["falsifiable_condition"]) \
                        if isinstance(candidate["falsifiable_condition"], str) \
                        else candidate["falsifiable_condition"]
                except (json.JSONDecodeError, TypeError):
                    pass

            self._semantic.add(
                content=f"[User Preference] {candidate['preference_text']}",
                confidence=candidate["confidence"],
                falsifiable_condition=falsifiable,
                metadata={
                    "preference_type": candidate["preference_type"],
                    "preference_id": pref_id,
                    "source": "preference_pipeline",
                    "trigger_context": candidate.get("trigger_context"),
                },
            )

            # Mark as promoted
            self._store.execute(
                "UPDATE preference_candidates SET status = 'promoted', updated_at = ? WHERE id = ?",
                (datetime.now(timezone.utc).isoformat(), pref_id),
            )
            self._store.commit()
            logger.info("Promoted preference %s → semantic memory", pref_id[:8])
        except Exception as exc:
            logger.error("Failed to promote preference %s: %s", pref_id[:8], exc)

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def count(self) -> int:
        row = self._store.fetchone("SELECT COUNT(*) as cnt FROM preference_candidates")
        return row["cnt"] if row else 0

    def count_by_status(self) -> Dict[str, int]:
        rows = self._store.fetchall(
            "SELECT status, COUNT(*) as cnt FROM preference_candidates GROUP BY status"
        )
        return {r["status"]: r["cnt"] for r in rows}

    def avg_confidence(self) -> float:
        row = self._store.fetchone(
            "SELECT AVG(confidence) as avg FROM preference_candidates WHERE status != 'rejected'"
        )
        return round(row["avg"], 3) if row and row["avg"] else 0.0

    def stats(self) -> dict:
        """Full stats for StatusReport."""
        return {
            "total_candidates": self.count(),
            "by_status": self.count_by_status(),
            "avg_confidence": self.avg_confidence(),
            "conflicts": self._count_conflicts(),
        }

    def _count_conflicts(self) -> int:
        row = self._store.fetchone(
            "SELECT COUNT(*) as cnt FROM preference_candidates "
            "WHERE conflict_with IS NOT NULL AND conflict_with != '[]'"
        )
        return row["cnt"] if row else 0


# ---------------------------------------------------------------------------
# Hook D: Preference Injection (runs on recall / match_procedural)
# ---------------------------------------------------------------------------

class PreferenceInjector:
    """Inject relevant preferences into agent context at decision points.

    Reuses: ProceduralMemory.match_by_context() for context-aware retrieval.
    Automatically creates procedural templates for confirmed preferences.

    Injection limits:
    - Max 3 preferences per recall
    - Max 300 chars in injection block
    - No re-injection within 5 minutes
    - Only preferences with confidence >= 0.5
    """

    def __init__(
        self,
        procedural: "ProceduralMemory",   # noqa: F821
        semantic: "SemanticMemory",        # noqa: F821
        embed_model: "EmbeddingModel",     # noqa: F821
        store: "SqliteStore",              # noqa: F821
    ):
        from cell_mem.embedding.local import EmbeddingModel
        from cell_mem.memory.procedural import ProceduralMemory
        from cell_mem.memory.semantic import SemanticMemory
        from cell_mem.storage.sqlite_store import SqliteStore

        self._procedural: ProceduralMemory = procedural
        self._semantic: SemanticMemory = semantic
        self._embed: EmbeddingModel = embed_model
        self._store: SqliteStore = store

    def inject_to_context(
        self,
        context_embedding: np.ndarray,
        min_confidence: float = INJECT_CONFIDENCE_MIN,
    ) -> List[dict]:
        """Find relevant preferences for current context.

        1. Search procedural memory for preference-triggered templates
        2. Search preference_candidates for confirmed, context-matching preferences
        3. Apply injection limits (count, tokens, interval)
        4. Return ranked list of injectable preferences
        """
        results: List[dict] = []

        # 1. Procedural memory: context-triggered preference templates
        proc_matches = self._procedural.match_by_context(
            context_embedding, threshold=0.6, top_k=3, explore_ratio=0.0,
        )
        for obj, sim in proc_matches:
            if obj.metadata.get("preference_id"):
                results.append({
                    "preference_id": obj.metadata["preference_id"],
                    "preference_text": obj.content,
                    "similarity": round(sim, 3),
                    "source": "procedural",
                })

        # 2. Direct search of confirmed preferences
        rows = self._store.fetchall(
            """SELECT id, preference_text, confidence, preference_type,
                      trigger_context, last_pushed_at
               FROM preference_candidates
               WHERE status IN ('confirmed', 'promoted')
                 AND confidence >= ?
               ORDER BY confidence DESC
               LIMIT 10""",
            (min_confidence,),
        )

        now = datetime.now(timezone.utc)
        for row in rows:
            r = dict(row)

            # Interval check: don't re-inject too frequently
            if r.get("last_pushed_at"):
                try:
                    last = datetime.fromisoformat(r["last_pushed_at"])
                    if (now - last).total_seconds() < MIN_INJECT_INTERVAL_SEC:
                        continue
                except (ValueError, TypeError):
                    pass

            # Context relevance via embedding similarity
            if self._embed:
                try:
                    pref_emb = self._embed.embed(r["preference_text"])
                    sim = float(
                        np.dot(context_embedding, pref_emb) /
                        (np.linalg.norm(context_embedding) * np.linalg.norm(pref_emb) + 1e-10)
                    )
                    if sim < 0.4:  # Minimum relevance threshold
                        continue
                    r["similarity"] = round(sim, 3)
                except Exception:
                    r["similarity"] = 0.5

            results.append({
                "preference_id": r["id"],
                "preference_text": r["preference_text"],
                "confidence": r["confidence"],
                "preference_type": r["preference_type"],
                "similarity": r.get("similarity", 0.5),
                "source": "candidate",
            })

        # 3. Deduplicate and rank
        seen = set()
        unique = []
        for r in sorted(results, key=lambda x: x.get("similarity", 0), reverse=True):
            key = r["preference_text"][:80]
            if key not in seen:
                seen.add(key)
                unique.append(r)

        # 4. Apply limits
        limited = unique[:MAX_INJECT_PER_RECALL]

        # 5. Update push timestamps
        for r in limited:
            if r["source"] == "candidate":
                self._store.execute(
                    """UPDATE preference_candidates
                       SET push_count = push_count + 1,
                           last_pushed_at = ?
                       WHERE id = ?""",
                    (now.isoformat(), r["preference_id"]),
                )
        if limited:
            self._store.commit()

        return limited

    def inject_to_recall(self, query: str) -> Optional[str]:
        """Build a preference context block for recall results.

        Returns formatted string suitable for appending to search results,
        or None if no relevant preferences found.
        """
        if not query:
            return None

        try:
            query_emb = self._embed.embed(query)
            prefs = self.inject_to_context(query_emb)
        except Exception:
            return None

        if not prefs:
            return None

        lines = []
        total_chars = 0
        for p in prefs:
            line = f"- {p['preference_text']} (confidence: {p.get('confidence', '?')})"
            total_chars += len(line)
            if total_chars > MAX_INJECT_TOKENS:
                break
            lines.append(line)

        if not lines:
            return None

        return "## User Preferences (auto-detected)\n" + "\n".join(lines)

    def build_context_block(self, context_embedding) -> Optional[str]:
        """Build injectable preference context for LLM system prompt.

        Returns None if no preferences are relevant to current context.
        """
        prefs = self.inject_to_context(context_embedding)
        if not prefs:
            return None

        lines = ["## User Preferences (auto-detected, confidence ≥ 0.5)"]
        for p in prefs:
            conf = p.get("confidence", "?")
            ptype = p.get("preference_type", "general")
            lines.append(f"- [{ptype}] {p['preference_text']} (confidence: {conf})")

        return "\n".join(lines)

    def promote_to_procedural(self, pref_id: str) -> bool:
        """Create a procedural template from a confirmed preference.

        The preference becomes automatically triggerable via match_by_context
        when the agent encounters a matching situation.
        """
        candidate = self._store.fetchone(
            "SELECT * FROM preference_candidates WHERE id = ?", (pref_id,)
        )
        if candidate is None:
            return False

        c = dict(candidate)
        trigger = c.get("trigger_context") or c.get("preference_text", "")

        try:
            self._procedural.store(
                template_content=f"[Preference] {c['preference_text']}",
                trigger_condition=trigger,
                task_type=f"preference_{c.get('preference_type', 'general')}",
                metadata={
                    "preference_id": pref_id,
                    "preference_type": c.get("preference_type"),
                    "auto_generated": True,
                    "source": "preference_pipeline",
                },
            )
            logger.info("Promoted preference %s → procedural template", pref_id[:8])
            return True
        except Exception as exc:
            logger.error("Failed to promote preference %s to procedural: %s", pref_id[:8], exc)
            return False

    # ------------------------------------------------------------------
    # Hook E: Implicit Feedback (no manual MCP call needed)
    # ------------------------------------------------------------------

    def process_procedural_feedback(self, proc_id: str, success: bool) -> None:
        """Auto-process preference feedback when a procedural template is used.

        Called by MemorySystem.record_procedural_outcome() — fully automatic.
        If the template was auto-generated from a preference, update its confidence.
        """
        obj = self._procedural.get(proc_id)
        if obj is None:
            return

        pref_id = obj.metadata.get("preference_id")
        if not pref_id:
            return  # Not a preference template, skip

        delta = PREFERENCE_SUCCESS_BOOST if success else -PREFERENCE_FAILURE_PENALTY

        # Update confidence in preference_candidates
        row = self._store.fetchone(
            "SELECT confidence FROM preference_candidates WHERE id = ?", (pref_id,)
        )
        if row is None:
            return

        new_conf = max(0.0, min(1.0, row["confidence"] + delta))
        now = datetime.now(timezone.utc).isoformat()
        self._store.execute(
            """UPDATE preference_candidates
               SET confidence = ?, updated_at = ?,
                   push_count = push_count + 1,
                   status = CASE WHEN ? >= 0.6 THEN 'confirmed' ELSE status END
               WHERE id = ?""",
            (new_conf, now, new_conf, pref_id),
        )
        self._store.commit()
        logger.debug("Preference %s auto-feedback: success=%s, confidence Δ=%+.3f",
                     pref_id[:8], success, delta)
