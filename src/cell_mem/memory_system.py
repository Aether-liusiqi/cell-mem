"""MemorySystem — top-level facade wiring all memory layers together.

This is the primary API that MCP tools use. It initializes all components
in the correct dependency order and provides save/recall/status entry points.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from cell_mem.consolidation import (
    ConsolidationScorer,
    ConsolidationScheduler,
    PatternDetector,
)
from cell_mem.embedding import EmbeddingModel, ProjectionMatrix
from cell_mem.graph import NetworkXGraphStore
from cell_mem.conditions import ConditionEvaluator
from cell_mem.memory import EpisodicMemory, ProceduralMemory, SemanticMemory, WorkingMemory
from cell_mem.models import MemoryObject, MemoryType, StatusReport
from cell_mem.reflection import ReflectionEngine
from cell_mem.storage import SqliteStore
from cell_mem.storage.search import SearchEngine
from cell_mem.storage.vector_store import ChromaStore, SqliteVecStore

logger = logging.getLogger(__name__)

# Default database location (in project directory)
DEFAULT_DB_PATH = "cell_mem.db"


class MemorySystem:
    """Top-level facade for the Cell-mem memory system.

    Initializes all four memory layers, embedding model, vector stores,
    and search engine in correct dependency order.

    Usage:
        ms = MemorySystem("cell_mem.db")
        ms.save("User prefers VSCode", memory_type="semantic")
        results = ms.recall("What editor does the user prefer?")
        status = ms.status()
        ms.shutdown()
    """

    def __init__(
        self,
        db_path: str = DEFAULT_DB_PATH,
        seed_config_path: str | None = None,
        vector_backend: str = "sqlite-vec",
        embedding_model_name: str = "all-MiniLM-L6-v2",
        embedding_device: str = "cpu",
        # --- Phase 3: LLM configuration (all optional for backward compat) ---
        llm_client: Optional[Any] = None,  # Pre-constructed LLMClient (for testing)
        llm_backend: str = "openai",
        llm_api_key: str | None = None,
        llm_daily_limit: int = 100,
        # --- HTTP authentication ---
        api_key: str | None = None,  # Shared secret for HTTP mode (Codex finding #1)
    ):
        logger.info("=== Cell-mem MemorySystem starting ===")

        # --- HTTP authentication ---
        self.api_key = api_key

        # --- Storage foundation ---
        self.store = SqliteStore(db_path)
        self.store.initialize_schema()

        # --- Embedding ---
        self.embed_model = EmbeddingModel(
            model_name=embedding_model_name,
            device=embedding_device,
        )

        # --- Projection matrix (pattern separation) ---
        self.projection = ProjectionMatrix(self.store)

        # --- Vector stores ---
        dim = self.embed_model.DIM
        if vector_backend == "sqlite-vec" and self.store.vec_available:
            self.episodic_vs = SqliteVecStore(self.store, "episodic_memory", dim)
            self.semantic_vs = SqliteVecStore(self.store, "semantic_memory", dim)
        else:
            logger.info("Using ChromaDB fallback for vector storage")
            db_dir = str(Path(db_path).parent)
            self.episodic_vs = ChromaStore(db_dir, "episodic_memory", dim)
            self.semantic_vs = ChromaStore(db_dir, "semantic_memory", dim)

        # --- Memory layers ---
        self.working = WorkingMemory(self.store)
        self.episodic = EpisodicMemory(
            self.store, self.episodic_vs, self.embed_model, self.projection
        )
        self.semantic = SemanticMemory(
            self.store, self.semantic_vs, self.embed_model
        )

        # --- Search ---
        self.search = SearchEngine(
            self.store, self.episodic_vs, self.semantic_vs, self.embed_model,
            graph_store=None,  # Wired below after graph init
        )

        # --- Phase 2a: real graph store ---
        self.graph = NetworkXGraphStore(self.store)
        self.search._graph = self.graph  # Enable two_pass strategy

        # --- Phase 2b: real consolidation ---
        from cell_mem.consolidation.emotional import RuleBasedScorer

        # --- Phase 3: LLM client (auto-construct if api_key given) ---
        self.llm_client = llm_client
        if self.llm_client is None and llm_api_key is not None:
            from cell_mem.llm import OpenAIBackend, ClaudeBackend, RateLimiter

            rl = RateLimiter(daily_limit=llm_daily_limit, store=self.store)
            if llm_backend == "openai":
                self.llm_client = OpenAIBackend(
                    api_key=llm_api_key, rate_limiter=rl,
                )
            elif llm_backend == "claude":
                self.llm_client = ClaudeBackend(
                    api_key=llm_api_key, rate_limiter=rl,
                )
            else:
                logger.warning("Unknown llm_backend '%s', LLM features disabled", llm_backend)
            if self.llm_client:
                logger.info("LLM client configured: backend=%s", llm_backend)

        # --- Phase 3: Emotional scorer with LLM fallback chain ---
        if self.llm_client is not None:
            from cell_mem.consolidation.emotional import FallbackScorer, LLMScorer

            self.emotional = FallbackScorer(
                primary=LLMScorer(self.llm_client),
                secondary=RuleBasedScorer(),
            )
        else:
            self.emotional = RuleBasedScorer()

        self.scorer = ConsolidationScorer(
            episodic=self.episodic,
            emotional_scorer=self.emotional,
            embed_model=self.embed_model,
        )

        # --- Phase 3: LLM-assisted pattern detection ---
        self.detector = PatternDetector(
            episodic=self.episodic,
            semantic=self.semantic,
            llm_client=self.llm_client,
        )

        self.scheduler = ConsolidationScheduler(
            episodic=self.episodic,
            semantic=self.semantic,
            scorer=self.scorer,
            detector=self.detector,
            store=self.store,
        )

        # --- Phase 3: Procedural memory ---
        self.procedural = ProceduralMemory(self.store, self.embed_model)

        # --- Phase 3: Reflection engine ---
        self.reflection = ReflectionEngine(
            episodic=self.episodic,
            llm_client=self.llm_client,
            embed_model=self.embed_model,
            procedural=self.procedural,  # Phase 4: full 4-dim reflection
            semantic=self.semantic,
        )

        # --- Phase 3: Condition evaluator ---
        self.condition_eval = ConditionEvaluator(sqlite_store=self.store)

        # --- Phase 4: Generative Replay Engine ---
        enable_replay = True  # Always attempt; disabled if no LLM
        if enable_replay and self.llm_client is not None:
            from cell_mem.replay import CreativePool, GenerativeReplayEngine

            self.creative_pool = CreativePool(
                sqlite_store=self.store,
                embed_model=self.embed_model,
                working=self.working,
                semantic=self.semantic,
            )
            self._replay_engine = GenerativeReplayEngine(
                semantic=self.semantic,
                episodic=self.episodic,
                graph=self.graph,
                llm_client=self.llm_client,
                embed_model=self.embed_model,
                working=self.working,
                creative_pool=self.creative_pool,
            )
        else:
            self.creative_pool = None
            self._replay_engine = None
            if enable_replay:
                logger.info("Replay engine disabled — no LLM configured")

        # --- Preference Pipeline: automatic user preference extraction/injection ---
        from cell_mem.consolidation.preference import (
            PreferenceSignalDetector,
            PreferenceExtractor,
            PreferenceProcessor,
            PreferenceInjector,
        )

        self.pref_detector = PreferenceSignalDetector()
        self.pref_extractor = PreferenceExtractor(
            store=self.store,
            embed_model=self.embed_model,
            semantic=self.semantic,
            llm_client=self.llm_client,
        )
        self.pref_processor = PreferenceProcessor(
            store=self.store,
            semantic=self.semantic,
            condition_eval=self.condition_eval,
            embed_model=self.embed_model,
        )
        self.pref_injector = PreferenceInjector(
            procedural=self.procedural,
            semantic=self.semantic,
            embed_model=self.embed_model,
            store=self.store,
        )
        logger.info("Preference pipeline ready (detector + extractor + processor + injector)")

        # --- Seed config ---
        if seed_config_path:
            count = self.semantic.import_seed_config(seed_config_path)
            logger.info("Seed config imported: %d entries", count)

        logger.info("=== MemorySystem ready ===")

    # ------------------------------------------------------------------
    # Public API (used by MCP tools)
    # ------------------------------------------------------------------

    def save(
        self,
        content: str,
        memory_type: str = "episodic",
        options: dict | None = None,
    ) -> dict:
        """Save a memory to the appropriate layer.

        Args:
            content: Text content to store.
            memory_type: "working" | "episodic" | "semantic" (default "episodic").
            options: Layer-specific options dict.

        Returns:
            {"status": "ok", "data": MemoryObject dict} or {"status": "error", "error": str}
        """
        # Content length guard (32KB — prevents OOM from oversized payloads)
        content = content[:32000]
        opts = options or {}
        try:
            mt = MemoryType(memory_type)
        except ValueError:
            return {
                "status": "error",
                "error": f"Invalid memory_type '{memory_type}'. "
                f"Must be one of: working, episodic, semantic, procedural",
            }

        try:
            if mt == MemoryType.WORKING:
                obj = self.working.add(
                    content=content,
                    session_id=opts.get("session_id", "default"),
                    base_priority=opts.get("base_priority", 1.0),
                    metadata=opts.get("metadata"),
                )
            elif mt == MemoryType.EPISODIC:
                obj = self.episodic.store(
                    content=content,
                    session_id=opts.get("session_id"),
                    task_id=opts.get("task_id"),
                    valence=opts.get("valence", 0.0),
                    was_in_wm=opts.get("was_in_wm", False),
                    metadata=opts.get("metadata"),
                )
                # Hook A: auto-detect preference signals on every episodic save
                self._detect_preference_signals(content, obj.id, opts.get("valence", 0.0))
            elif mt == MemoryType.SEMANTIC:
                obj = self.semantic.add(
                    content=content,
                    confidence=opts.get("confidence", 0.0),
                    falsifiable_condition=opts.get("falsifiable_condition"),
                    source_references=opts.get("source_references"),
                    metadata=opts.get("metadata"),
                )
            elif mt == MemoryType.PROCEDURAL:
                obj = self.procedural.store(
                    template_content=content,
                    trigger_condition=opts.get("trigger_condition"),
                    task_type=opts.get("task_type"),
                    source_episode_ids=opts.get("source_episode_ids"),
                    metadata=opts.get("metadata"),
                )
            else:
                return {"status": "error", "error": f"memory_type '{memory_type}' not supported"}

            return {"status": "ok", "data": obj.model_dump()}
        except Exception as exc:
            logger.exception("Save failed: %s", exc)
            return {"status": "error", "error": str(exc)}

    def recall(
        self,
        query: str,
        options: dict | None = None,
    ) -> dict:
        """Search across memory layers.

        Args:
            query: Natural language search query.
            options: Search options (see SearchEngine.search).

        Returns:
            {"status": "ok", "data": [MemoryObject dicts]} or error dict
        """
        opts = options or {}
        try:
            # Embed query once, reuse for refresh and search
            q_embedding = self.embed_model.embed_query(query)

            # Proactive refresh of working memory (lazy, not background thread)
            if self.working.should_refresh():
                try:
                    self.working.proactive_refresh(q_embedding, self.embed_model)
                except Exception:
                    pass  # Refresh failure is non-fatal

            results = self.search.search_by_vector(q_embedding, opts, query_text=query)
            return {
                "status": "ok",
                "data": [obj.model_dump() for obj in results],
                "count": len(results),
            }
        except Exception as exc:
            logger.exception("Recall failed: %s", exc)
            return {"status": "error", "error": str(exc)}

    def status(self) -> dict:
        """Aggregate health and statistics from all layers.

        Returns:
            StatusReport as dict.
        """
        try:
            report = StatusReport(
                layers={
                    "working": {
                        "entry_count": self.working.count(),
                        "avg_attention": round(self.working.avg_attention(), 3),
                    },
                    "episodic": {
                        "entry_count": self.episodic.count(),
                        "oldest_at": self.episodic.oldest_at(),
                        "by_confidence": self.episodic.count_by_confidence(),
                    },
                    "semantic": {
                        "entry_count": self.semantic.count(),
                        "avg_confidence": round(self.semantic.avg_confidence(), 3),
                        "with_falsifiable_conditions": self.semantic.count_with_falsifiable(),
                        "expired": self.semantic.count_expired(),
                    },
                    "procedural": {
                        "entry_count": self.procedural.count(),
                        "avg_activation_weight": round(self.procedural.avg_activation_weight(), 3),
                        "by_lifecycle": self.procedural.count_by_lifecycle(),
                        "with_conditions": self.procedural.count_with_conditions(),
                    },
                },
                vector_index={
                    "dimension": 384,
                    "backend": "sqlite-vec" if self.store.vec_available else "chromadb",
                },
                graph={
                    "node_count": self.graph.node_count(),
                    "edge_count": self.graph.edge_count(),
                },
                consolidation={
                    "cycles_run": self.scheduler._cycle_count,
                    "last_cycle_at": self.scheduler._last_cycle_at,
                    "cold_storage_count": self.scheduler.get_cold_storage_count(),
                    "phase": "4",
                },
                llm={
                    "configured": self.llm_client is not None,
                    "daily_remaining": (
                        self.llm_client._rl.remaining
                        if self.llm_client and hasattr(self.llm_client, "_rl") and self.llm_client._rl
                        else None
                    ),
                },
                creative_pool={
                    "total_hypotheses": self.creative_pool.count() if self.creative_pool else 0,
                    "by_status": self.creative_pool.count_by_status() if self.creative_pool else {},
                    "replay_available": self._replay_engine is not None,
                },
                reflection={
                    "dimensions": 4,
                    "recent_count": len(self.reflection.get_recent_reflections(limit=10)),
                },
                preferences=self.pref_processor.stats(),
                health="healthy",
            )
            return report.model_dump()
        except Exception as exc:
            logger.exception("Status failed: %s", exc)
            return StatusReport(health="error").model_dump()

    def associate(
        self,
        source_id: str,
        target_id: str,
        weight: float = 1.0,
        relation_type: str = "associated_with",
    ) -> dict:
        """Create an association edge between two memory items.

        Args:
            source_id: Source memory ID.
            target_id: Target memory ID.
            weight: Edge weight in [-1, 1].
            relation_type: Edge type (associated_with, causes, contradicts, etc.)

        Returns:
            {"status": "ok", "data": {...}} or error dict
        """
        try:
            # Validate weight range
            if not (-1.0 <= weight <= 1.0):
                return {
                    "status": "error",
                    "error": f"Weight {weight} out of range [-1, 1]",
                }

            # Validate relation_type
            valid_types = {"associated_with", "causes", "contradicts", "is_a", "part_of"}
            if relation_type not in valid_types:
                return {
                    "status": "error",
                    "error": f"Invalid relation_type '{relation_type}'. Must be one of: {sorted(valid_types)}",
                }

            self.graph.add_edge(source_id, target_id, weight, relation_type)
            return {
                "status": "ok",
                "data": {
                    "source": source_id,
                    "target": target_id,
                    "weight": weight,
                    "relation_type": relation_type,
                },
            }
        except Exception as exc:
            logger.exception("Associate failed: %s", exc)
            return {"status": "error", "error": str(exc)}

    def consolidate(self) -> dict:
        """Manually trigger a consolidation cycle.

        Runs scoring, forget candidate identification, cold storage archival,
        DBSCAN pattern detection, and automatic preference extraction + decay.

        Returns:
            {"status": "ok", "data": {cycle_stats...}} or error dict.
        """
        try:
            result = self.scheduler.run_cycle()

            # Hook B: Auto-extract preferences after consolidation cycle
            self._run_preference_cycle()

            return result
        except Exception as exc:
            logger.exception("Consolidate failed: %s", exc)
            return {"status": "error", "error": str(exc)}

    def forget(
        self,
        memory_id: str,
        reason: str = "",
        expire: bool = False,
    ) -> dict:
        """Forget or expire a specific memory.

        Searches episodic first, then semantic. If expire=True, marks
        invalidated_at instead of hard-deleting.

        Args:
            memory_id: ID of the memory to forget.
            reason: Reason for forgetting (logged in metadata).
            expire: If True, soft-delete (set invalidated_at) instead of hard delete.

        Returns:
            {"status": "ok", "data": {action, memory_id, reason}} or error dict.
        """
        import json
        from datetime import datetime, timezone

        try:
            # Try episodic first
            ep = self.episodic.get(memory_id)
            if ep is not None:
                if expire:
                    now = datetime.now(timezone.utc).isoformat()
                    new_meta = {**ep.metadata, "expire_reason": reason, "expired": True}
                    meta_json = json.dumps(new_meta, ensure_ascii=False)
                    self.store.execute(
                        "UPDATE episodic_memory SET invalidated_at = ?, metadata_json = ? WHERE id = ?",
                        (now, meta_json, memory_id),
                    )
                    self.store.commit()
                    return {
                        "status": "ok",
                        "data": {"action": "expired", "memory_id": memory_id, "reason": reason},
                    }
                else:
                    self.episodic.delete(memory_id)
                    return {
                        "status": "ok",
                        "data": {"action": "deleted", "memory_id": memory_id, "reason": reason},
                    }

            # Try semantic
            sem = self.semantic.get(memory_id)
            if sem is not None:
                if expire:
                    now = datetime.now(timezone.utc).isoformat()
                    self.store.execute(
                        "UPDATE semantic_memory SET invalidated_at = ? WHERE id = ?",
                        (now, memory_id),
                    )
                    self.store.commit()
                    return {
                        "status": "ok",
                        "data": {"action": "expired", "memory_id": memory_id, "reason": reason},
                    }
                else:
                    self.semantic.delete(memory_id)
                    return {
                        "status": "ok",
                        "data": {"action": "deleted", "memory_id": memory_id, "reason": reason},
                    }

            return {
                "status": "error",
                "error": f"Memory '{memory_id}' not found in episodic or semantic layers",
            }
        except Exception as exc:
            logger.exception("Forget failed: %s", exc)
            return {"status": "error", "error": str(exc)}

    # ------------------------------------------------------------------
    # Phase 3: Procedural memory API
    # ------------------------------------------------------------------

    def save_procedural(
        self,
        template_content: str,
        trigger_condition: str | None = None,
        task_type: str | None = None,
        source_episode_ids: list[str] | None = None,
        metadata: dict | None = None,
    ) -> dict:
        """Create a procedural memory template.

        Args:
            template_content: The procedure/skill description.
            trigger_condition: Natural language condition for auto-triggering.
            task_type: Optional tag (debug, refactor, deploy, etc.).
            source_episode_ids: Episode IDs that informed this template.
            metadata: Additional metadata dict.

        Returns:
            {"status": "ok", "data": MemoryObject dict} or error dict.
        """
        try:
            obj = self.procedural.store(
                template_content=template_content,
                trigger_condition=trigger_condition,
                task_type=task_type,
                source_episode_ids=source_episode_ids,
                metadata=metadata,
            )
            return {"status": "ok", "data": obj.model_dump()}
        except Exception as exc:
            logger.exception("save_procedural failed: %s", exc)
            return {"status": "error", "error": str(exc)}

    def match_procedural(
        self,
        context_text: str,
        threshold: float = 0.65,
        top_k: int = 5,
        explore_ratio: float = 0.2,
    ) -> dict:
        """Match a context description against procedural templates.

        Embeds the context text and finds templates with similar condition_embedding.
        Uses epsilon-greedy explore/exploit: explore_ratio=0.2 means 20% chance
        of picking a less-proven template for discovery.

        Args:
            context_text: Description of the current task/situation.
            threshold: Minimum cosine similarity (default 0.65).
            top_k: Maximum matches to return.
            explore_ratio: Probability of exploring non-best matches (0=always exploit, 1=always explore).

        Returns:
            {"status": "ok", "data": [...matches...]} or error dict.
        """
        try:
            ctx_embedding = self.embed_model.embed_query(context_text)
            matches = self.procedural.match_by_context(ctx_embedding, threshold, top_k, explore_ratio)
            return {
                "status": "ok",
                "data": [
                    {"template": obj.model_dump(), "similarity": round(sim, 4)}
                    for obj, sim in matches
                ],
                "count": len(matches),
            }
        except Exception as exc:
            logger.exception("match_procedural failed: %s", exc)
            return {"status": "error", "error": str(exc)}

    def record_procedural_outcome(self, proc_id: str, success: bool) -> dict:
        """Record success/failure outcome for a procedural template.

        Updates activation_weight via reinforcement learning factors.
        Automatically feeds back to preference confidence if the template
        was auto-generated from a user preference (Hook E — no manual call).

        Args:
            proc_id: The procedural template ID.
            success: True if the procedure worked, False if it failed.

        Returns:
            {"status": "ok", "data": updated MemoryObject dict} or error dict.
        """
        try:
            obj = self.procedural.record_outcome(proc_id, success)
            if obj is None:
                return {"status": "error", "error": f"Template '{proc_id}' not found"}

            # Hook E: Implicit preference feedback — auto-adjust preference
            # confidence when a preference-derived template is used
            try:
                self.pref_injector.process_procedural_feedback(proc_id, success)
            except Exception:
                pass  # Non-critical — don't fail the main call

            return {"status": "ok", "data": obj.model_dump()}
        except Exception as exc:
            logger.exception("record_procedural_outcome failed: %s", exc)
            return {"status": "error", "error": str(exc)}

    # ------------------------------------------------------------------
    # Phase 3: Reflection API
    # ------------------------------------------------------------------

    def reflect(
        self, task_description: str, outcome: str = "failure",
        dimensions: str = "failure",
    ) -> dict:
        """Trigger self-reflection.

        Args:
            task_description: Description of the task.
            outcome: "failure" (default) or "success".
            dimensions: "failure" (Phase 3 compat, dim 1 only),
                        "all" (4 dimensions), or "strategy,gaps,process".

        Returns:
            {"status": "ok", "data": {...}} or error dict.
        """
        try:
            if dimensions == "failure":
                return self.reflection.reflect_on_failure(task_description, outcome)
            return self.reflection.full_reflection(task_description, outcome)
        except Exception as exc:
            logger.exception("reflect failed: %s", exc)
            return {"status": "error", "error": str(exc)}

    # ------------------------------------------------------------------
    # Phase 3: Falsifiable condition API
    # ------------------------------------------------------------------

    def verify(self, entry_id: str, environment: dict | None = None) -> dict:
        """Verify falsifiable condition for a single semantic memory entry.

        If the condition is met, the entry is marked as expired.

        Args:
            entry_id: ID of the semantic memory entry to verify.
            environment: Dict of current environment values (e.g.,
                {"react_version": "18", "os": "linux"}). Required.

        Returns:
            {"status": "ok", "data": {verification_result}} or error dict.
        """
        try:
            if not environment:
                return {
                    "status": "error",
                    "error": "No environment provided. Pass current environment values as a dict.",
                }
            entry = self.semantic.get(entry_id)
            if entry is None:
                return {"status": "error", "error": f"Entry '{entry_id}' not found"}
            result = self.condition_eval.verify_entry(entry, environment)
            # Expire if condition met
            if result.get("action") == "expire":
                import json
                from datetime import datetime, timezone

                now = datetime.now(timezone.utc).isoformat()
                self.store.execute(
                    "UPDATE semantic_memory SET invalidated_at = ? WHERE id = ?",
                    (now, entry_id),
                )
                self.store.commit()
                result["expired"] = True
                logger.info("Expired semantic entry %s (condition met)", entry_id[:8])
            return {"status": "ok", "data": result}
        except Exception as exc:
            logger.exception("verify failed: %s", exc)
            return {"status": "error", "error": str(exc)}

    def verify_all(self, environment: dict) -> dict:
        """Verify all semantic entries with falsifiable conditions.

        Args:
            environment: Dict of current environment values.

        Returns:
            {"status": "ok", "data": {total_checked, triggered, still_valid}}.
        """
        try:
            return self.condition_eval.verify_all(self.semantic, environment)
        except Exception as exc:
            logger.exception("verify_all failed: %s", exc)
            return {"status": "error", "error": str(exc)}

    # ------------------------------------------------------------------
    # Phase 4: Generative Replay API
    # ------------------------------------------------------------------

    def replay(self, theme_text: str | None = None) -> dict:
        """Manually trigger a generative replay cycle.

        Args:
            theme_text: Optional topic to bias seed selection.

        Returns:
            {"status": "ok", "data": {"hypotheses_generated": N, ...}}
        """
        if self._replay_engine is None:
            return {"status": "error", "error": "Replay engine not available. Configure llm_api_key."}
        try:
            return self._replay_engine.run_replay_cycle(theme_text)
        except Exception as exc:
            logger.exception("replay failed: %s", exc)
            return {"status": "error", "error": str(exc)}

    def creative_pool_status(self, status: str | None = None) -> dict:
        """Query the creative pool.

        Args:
            status: Filter by status (pending/confirmed/rejected/promoted). None = pending.

        Returns:
            {"status": "ok", "data": {"entries": [...], "count_by_status": {...}}}
        """
        if self.creative_pool is None:
            return {"status": "error", "error": "Creative pool not available."}
        try:
            entries = self.creative_pool.get_by_status(status or "pending")
            return {
                "status": "ok",
                "data": {
                    "entries": entries,
                    "count_by_status": self.creative_pool.count_by_status(),
                },
            }
        except Exception as exc:
            logger.exception("creative_pool_status failed: %s", exc)
            return {"status": "error", "error": str(exc)}

    def record_hypothesis_feedback(self, hypothesis_id: str, confirmed: bool) -> dict:
        """Record user feedback on a creative pool hypothesis.

        Args:
            hypothesis_id: ID of the hypothesis from creative_pool.
            confirmed: True if user confirms hypothesis was useful.

        Returns:
            {"status": "ok", "data": {"action": "...", ...}}
        """
        if self.creative_pool is None:
            return {"status": "error", "error": "Creative pool not available."}
        try:
            return self.creative_pool.record_feedback(hypothesis_id, confirmed)
        except Exception as exc:
            logger.exception("record_hypothesis_feedback failed: %s", exc)
            return {"status": "error", "error": str(exc)}

    def check_environment(self, current_env: dict) -> dict:
        """Check for environment changes and auto-verify affected entries.

        Args:
            current_env: Current environment state dict.

        Returns:
            {"status": "ok", "data": {"changes_detected": ..., "triggered_entries": [...]}}
        """
        try:
            return self.condition_eval.auto_verify(self.semantic, current_env)
        except Exception as exc:
            logger.exception("check_environment failed: %s", exc)
            return {"status": "error", "error": str(exc)}

    # ------------------------------------------------------------------
    # Preference Pipeline API
    # ------------------------------------------------------------------

    def extract_preferences(self, session_id: str | None = None, limit: int = 50) -> dict:
        """Manually trigger preference extraction from recent episodes.

        Normally runs automatically during consolidation cycles.
        Use this to force extraction on demand.

        Args:
            session_id: Optional session filter.
            limit: Max recent episodes to scan.

        Returns:
            {"status": "ok", "data": {"candidates_created": N, "signals_found": N}}
        """
        try:
            episodes = self.episodic.load_all(limit=limit)
            signals = self.pref_detector.batch_detect(episodes)
            if not signals:
                return {"status": "ok", "data": {"candidates_created": 0, "signals_found": 0}}

            candidates = self.pref_extractor.extract(signals, episodes)
            candidates = self.pref_extractor.deduplicate(candidates)

            created = 0
            for c in candidates:
                self.pref_processor.add_candidate(c)
                created += 1

            logger.info("Preference extraction: %d signals → %d candidates", len(signals), created)
            return {"status": "ok", "data": {"candidates_created": created, "signals_found": len(signals)}}
        except Exception as exc:
            logger.exception("extract_preferences failed: %s", exc)
            return {"status": "error", "error": str(exc)}

    def get_preferences(
        self,
        context_text: str | None = None,
        min_confidence: float = 0.3,
        preference_type: str | None = None,
    ) -> dict:
        """Query active preferences, optionally filtered by context or type.

        Args:
            context_text: If provided, rank by relevance to this context.
            min_confidence: Minimum confidence threshold.
            preference_type: Filter by type (tool_choice, workflow, etc.).

        Returns:
            {"status": "ok", "data": {"preferences": [...], "count": N}}
        """
        try:
            if preference_type:
                prefs = self.pref_processor.get_by_type(preference_type, min_confidence)
            else:
                prefs = self.pref_processor.get_confirmed(limit=50)
                prefs = [p for p in prefs if p["confidence"] >= min_confidence]

            # If context provided, rank by relevance
            if context_text and prefs and self.embed_model:
                ctx_emb = self.embed_model.embed(context_text)
                for p in prefs:
                    try:
                        pref_emb = self.embed_model.embed(p["preference_text"])
                        p["relevance"] = round(float(
                            np.dot(ctx_emb, pref_emb) /
                            (np.linalg.norm(ctx_emb) * np.linalg.norm(pref_emb) + 1e-10)
                        ), 3)
                    except Exception:
                        p["relevance"] = 0.5
                prefs.sort(key=lambda x: x.get("relevance", 0), reverse=True)

            return {"status": "ok", "data": {"preferences": prefs, "count": len(prefs)}}
        except Exception as exc:
            logger.exception("get_preferences failed: %s", exc)
            return {"status": "error", "error": str(exc)}

    def check_preference_conflicts(self) -> dict:
        """Detect contradictory preferences.

        Returns:
            {"status": "ok", "data": {"conflicts": [...], "count": N}}
        """
        try:
            conflicts = self.pref_processor.detect_conflicts()
            return {"status": "ok", "data": {"conflicts": conflicts, "count": len(conflicts)}}
        except Exception as exc:
            logger.exception("check_preference_conflicts failed: %s", exc)
            return {"status": "error", "error": str(exc)}

    def inject_preference(self, preference_text: str, preference_type: str = "general",
                         confidence: float = 0.7, trigger_context: str | None = None,
                         falsifiable_condition: dict | None = None) -> dict:
        """Manually add a preference (e.g., from explicit user statement).

        Args:
            preference_text: The preference statement.
            preference_type: tool_choice, workflow, communication_style, skill_level.
            confidence: Initial confidence (0.0-1.0).
            trigger_context: When this preference should be activated (optional).
            falsifiable_condition: Optional condition for auto-expiry.

        Returns:
            {"status": "ok", "data": {"preference_id": "..."}}
        """
        try:
            candidate = {
                "preference_text": preference_text,
                "preference_type": preference_type,
                "confidence": confidence,
                "trigger_context": trigger_context,
                "signal_strength": 0.8,
                "source_episode_ids": [],
                "metadata": {"source": "manual_injection"},
            }
            if falsifiable_condition:
                candidate["falsifiable_condition"] = json.dumps(falsifiable_condition)

            pref_id = self.pref_processor.add_candidate(candidate)
            if confidence >= 0.6:
                self.pref_processor.update_confidence(pref_id, 0.0)  # triggers promote

            return {"status": "ok", "data": {"preference_id": pref_id}}
        except Exception as exc:
            logger.exception("inject_preference failed: %s", exc)
            return {"status": "error", "error": str(exc)}

    def record_preference_feedback(self, preference_id: str, confirmed: bool) -> dict:
        """Record feedback on whether a preference was accurate.

        Normally auto-triggered via record_procedural_outcome() for
        preference-derived procedural templates. Use this for manual feedback.

        Args:
            preference_id: ID from preference_candidates.
            confirmed: True if preference accurately described user behavior.

        Returns:
            {"status": "ok", "data": {"new_confidence": X, "action": "..."}}
        """
        try:
            delta = 0.1 if confirmed else -0.1
            new_conf = self.pref_processor.update_confidence(preference_id, delta)
            action = "strengthened" if confirmed else "weakened"
            return {"status": "ok", "data": {"new_confidence": new_conf, "action": action}}
        except Exception as exc:
            logger.exception("record_preference_feedback failed: %s", exc)
            return {"status": "error", "error": str(exc)}

    # ------------------------------------------------------------------
    # Internal: Preference Pipeline Hooks (fully automatic)
    # ------------------------------------------------------------------

    def _detect_preference_signals(self, content: str, episode_id: str, valence: float = 0.0) -> None:
        """Hook A: Auto-detect preference signals after episodic save.

        Runs inline — fast keyword check, no significant overhead.
        Detected signals are cached in memory for the next consolidation cycle.
        """
        try:
            if not self.pref_detector.has_signals(content):
                return
            signals = self.pref_detector.detect(content, episode_id, valence)
            if signals:
                # Store signals as metadata on the episode for later batch extraction
                existing_meta = {}
                try:
                    ep = self.episodic.get(episode_id)
                    if ep and ep.metadata:
                        existing_meta = dict(ep.metadata)
                except Exception:
                    pass
                existing_meta["_preference_signals"] = [s.to_dict() for s in signals]
                self.episodic.update(episode_id, metadata=existing_meta)
                logger.debug("Detected %d preference signal(s) in episode %s",
                            len(signals), episode_id[:8])
        except Exception as exc:
            logger.debug("Preference signal detection skipped: %s", exc)

    def _run_preference_cycle(self) -> None:
        """Hook B+C: Run full preference extraction + processing cycle.

        Called automatically after each consolidation cycle.
        1. Scan recent episodes for preference signals
        2. Extract candidates via LLM/heuristic
        3. Deduplicate against existing preferences
        4. Detect conflicts
        5. Apply decay to stale preferences
        """
        try:
            # 1. Scan recent episodic entries for signals
            episodes = self.episodic.load_all(limit=100)
            signals = self.pref_detector.batch_detect(episodes)
            if not signals:
                return

            # 2. Extract candidates
            candidates = self.pref_extractor.extract(signals, episodes)
            candidates = self.pref_extractor.deduplicate(candidates)

            # 3. Add new candidates (skip if similar preference already exists)
            created = 0
            for c in candidates:
                existing_id = self.pref_extractor._check_existing(c.get("preference_text", ""))
                if existing_id:
                    # Boost existing preference confidence slightly
                    self.pref_processor.update_confidence(existing_id, 0.02)
                else:
                    self.pref_processor.add_candidate(c)
                    created += 1

            # 4. Detect conflicts among active preferences
            self.pref_processor.detect_conflicts()

            # 5. Apply decay to stale preferences
            decayed = self.pref_processor.apply_decay()

            if created or decayed:
                logger.info("Preference cycle: %d new, %d decayed", created, decayed)
        except Exception as exc:
            logger.debug("Preference cycle skipped (non-critical): %s", exc)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def shutdown(self) -> None:
        """Close connections and release resources."""
        self.store.close()
        logger.info("MemorySystem shut down")
