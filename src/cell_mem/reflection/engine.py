"""Self-reflection engine — Phase 3: failure attribution (归因分析).

Phase 3 focuses on failure attribution only (why did a task fail?).
Phase 4 will add strategy evaluation, knowledge gap detection, and
meta-knowledge creation (the full 4-dimension reflection loop).

The engine searches relevant episodic memories, sends them to an LLM
with the task description, and stores the attribution analysis as a
new episodic memory entry tagged with metadata.reflection=True.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


class ReflectionEngine:
    """Primary self-reflection engine — failure attribution analysis.

    Constructor-injected dependencies:
    - episodic: EpisodicMemory (for finding relevant history and storing results)
    - llm_client: LLMClient (may be None — reflection is disabled without LLM)
    - embed_model: EmbeddingModel (for embedding task description)
    """

    def __init__(
        self,
        episodic: "EpisodicMemory",  # noqa: F821
        llm_client: Optional["LLMClient"],  # noqa: F821
        embed_model: "EmbeddingModel",  # noqa: F821
        procedural: Optional["ProceduralMemory"] = None,  # noqa: F821
        semantic: Optional["SemanticMemory"] = None,  # noqa: F821
    ):
        from cell_mem.embedding.local import EmbeddingModel
        from cell_mem.llm.client import LLMClient
        from cell_mem.memory.episodic import EpisodicMemory
        from cell_mem.memory.procedural import ProceduralMemory
        from cell_mem.memory.semantic import SemanticMemory

        self._episodic: EpisodicMemory = episodic
        self._llm: Optional[LLMClient] = llm_client
        self._embed: EmbeddingModel = embed_model
        self._procedural: Optional[ProceduralMemory] = procedural
        self._semantic: Optional[SemanticMemory] = semantic
        logger.info("ReflectionEngine ready (llm=%s, dims=%d)",
                    "configured" if llm_client else "none",
                    4 if procedural and semantic else 1)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reflect_on_failure(
        self,
        task_description: str,
        outcome: str = "failure",
    ) -> dict:
        """Analyze why a task failed.

        1. Embed the task description and search for relevant past episodes.
        2. Send task + history to LLM for root cause attribution.
        3. Store the reflection as a new episodic memory.

        Args:
            task_description: Description of the task that was attempted.
            outcome: "failure" (default) or "success". Phase 3 focuses on failure.

        Returns:
            {"status": "ok", "data": {"reflection_id": "...", "analysis": {...}}}
            or {"status": "error", "error": "..."}
        """
        if self._llm is None:
            return {
                "status": "error",
                "error": "No LLM configured for reflection. "
                         "Provide an llm_api_key to MemorySystem to enable self-reflection.",
            }

        try:
            # Step 1: Find relevant episodes
            q_embedding = self._embed.embed_query(task_description)
            relevant = self._episodic.recall_by_vector(q_embedding, top_k=10)

            # Step 2: Build attribution prompt
            history_lines = []
            for ep in relevant:
                history_lines.append(f"- [{ep.id[:8]}] {ep.content[:300]}")

            history_text = "\n".join(history_lines) if history_lines else "(no relevant history)"

            prompt = (
                "You are analyzing why a task failed. Review the task description "
                "and relevant memory history below. Identify the most likely root "
                "cause and contributing factors. Also note any signal the agent "
                "might have missed.\n\n"
                f"TASK: {task_description}\n\n"
                f"HISTORY:\n{history_text}"
            )

            schema_hint = {
                "root_cause": "string — the primary reason the task failed",
                "contributing_factors": "array of strings — other factors that contributed",
                "missed_signal": "string or null — any warning sign the agent overlooked",
                "confidence": "float between 0 and 1 — how confident you are in this analysis",
            }

            analysis = self._llm.generate(prompt, schema_hint)

            # Step 3: Store reflection as episodic memory
            root_cause = analysis.get("root_cause", "unknown")
            confidence = float(analysis.get("confidence", 0.0))
            factors = analysis.get("contributing_factors", [])
            factors_str = "; ".join(factors) if factors else "none identified"
            missed = analysis.get("missed_signal", "none")

            reflection_text = (
                f"[Self-Reflection] Task failed: {task_description[:200]}\n"
                f"Root cause: {root_cause}\n"
                f"Contributing factors: {factors_str}\n"
                f"Missed signal: {missed}\n"
                f"Confidence: {confidence:.2f}"
            )

            ep = self._episodic.store(
                content=reflection_text,
                session_id="reflection",
                valence=-0.3,  # Slightly negative — it's a failure analysis
                metadata={
                    "reflection": True,
                    "outcome": outcome,
                    "analysis": analysis,
                },
            )

            logger.info(
                "Reflection stored: %s (root_cause=%s, confidence=%.2f)",
                ep.id[:8], root_cause[:80], confidence,
            )

            return {
                "status": "ok",
                "data": {
                    "reflection_id": ep.id,
                    "analysis": analysis,
                },
            }

        except Exception as exc:
            logger.exception("Reflection failed: %s", exc)
            return {"status": "error", "error": str(exc)}

    def get_recent_reflections(self, limit: int = 10) -> List:
        """Return recent reflection entries from episodic memory.

        Filters by metadata.reflection == True.
        """
        all_eps = self._episodic.load_all(limit=500)
        reflections = [
            ep for ep in all_eps
            if ep.metadata.get("reflection") is True
        ]
        reflections.sort(key=lambda ep: ep.created_at, reverse=True)
        return reflections[:limit]

    # ------------------------------------------------------------------
    # Dimension 2: Strategy Effectiveness (Phase 4)
    # ------------------------------------------------------------------

    def evaluate_strategy(self) -> dict:
        """Evaluate procedural template effectiveness.

        Checks: success rate trends, better variant discovery, redundancy.
        Pure computation — no LLM required.
        """
        if self._procedural is None:
            return {"status": "skipped", "reason": "No procedural memory configured"}

        try:
            templates = self._procedural.load_all(limit=200)
            if not templates:
                return {"status": "ok", "data": {"success_rate_trend": "no_data", "better_variants": [], "redundant_pairs": []}}

            # Compute success rates
            rates = []
            for t in templates:
                s = t.metadata.get("success_count", 0)
                f = t.metadata.get("failure_count", 0)
                total = s + f
                rates.append({
                    "id": t.id,
                    "task_type": t.metadata.get("task_type", ""),
                    "content": t.content[:100],
                    "success_rate": s / total if total > 0 else 0.5,
                    "total_uses": total,
                })

            # Success rate trend: overall average
            avg_rate = sum(r["success_rate"] for r in rates) / len(rates) if rates else 0.0

            # Better variant detection: group by task_type, find best
            by_type: Dict[str, List] = {}
            for r in rates:
                tt = r["task_type"] or "general"
                by_type.setdefault(tt, []).append(r)
            better_variants = []
            for tt, items in by_type.items():
                if len(items) >= 2:
                    items.sort(key=lambda x: x["success_rate"], reverse=True)
                    best = items[0]
                    for other in items[1:]:
                        if best["success_rate"] - other["success_rate"] > 0.15:
                            better_variants.append({
                                "task_type": tt,
                                "best_id": best["id"],
                                "best_rate": round(best["success_rate"], 3),
                                "other_id": other["id"],
                                "other_rate": round(other["success_rate"], 3),
                            })

            # Redundancy check
            redundant_pairs = []
            for i in range(len(templates)):
                for j in range(i + 1, len(templates)):
                    if templates[i].metadata.get("task_type") == templates[j].metadata.get("task_type"):
                        ci = templates[i].metadata.get("trigger_condition", "")
                        cj = templates[j].metadata.get("trigger_condition", "")
                        if ci and cj:
                            ei = self._embed.embed(ci)
                            ej = self._embed.embed(cj)
                            sim = float(np.dot(ei, ej) / (np.linalg.norm(ei) * np.linalg.norm(ej)))
                            if sim > 0.9:
                                redundant_pairs.append({
                                    "id_a": templates[i].id,
                                    "id_b": templates[j].id,
                                    "similarity": round(sim, 3),
                                })

            return {
                "status": "ok",
                "data": {
                    "success_rate_avg": round(avg_rate, 3),
                    "better_variants": better_variants,
                    "redundant_pairs": redundant_pairs,
                },
            }
        except Exception as exc:
            return {"status": "error", "error": str(exc)}

    # ------------------------------------------------------------------
    # Dimension 3: Knowledge Gap Detection (Phase 4)
    # ------------------------------------------------------------------

    def detect_knowledge_gaps(self, failure_context: str) -> dict:
        """Detect what knowledge is missing or not retrieved.

        gap_type: "missing" (not in semantic), "retrieval" (exists but not accessed),
        or "none" (knowledge was present and retrieved).
        """
        if self._semantic is None:
            return {"status": "skipped", "reason": "No semantic memory configured"}

        try:
            ctx_emb = self._embed.embed_query(failure_context)
            similar = self._semantic.recall_by_vector(ctx_emb, top_k=10)

            if not similar or similar[0].metadata.get("_search_distance", 1.0) > 0.5:
                gap_type = "missing"
            elif similar[0].metadata.get("_search_distance", 1.0) < 0.3:
                gap_type = "retrieval"  # Knowledge exists but was not in context
            else:
                gap_type = "none"

            suggested_knowledge = "No LLM available for suggestion"
            if self._llm:
                try:
                    prompt = (
                        "Based on this failure context, what specific knowledge "
                        "would have helped prevent the failure?\n\n"
                        f"Failure: {failure_context[:500]}\n\n"
                        'Respond with JSON: {"knowledge_domain": "...", "specific_topic": "..."}'
                    )
                    result = self._llm.generate(prompt)
                    suggested_knowledge = result.get("specific_topic", result.get("knowledge_domain", ""))
                except Exception:
                    pass

            return {
                "status": "ok",
                "data": {
                    "gap_type": gap_type,
                    "suggested_knowledge": suggested_knowledge,
                    "existing_in_semantic": [
                        {"id": s.id, "content": s.content[:200]}
                        for s in similar[:3]
                    ],
                },
            }
        except Exception as exc:
            return {"status": "error", "error": str(exc)}

    # ------------------------------------------------------------------
    # Dimension 4: Result Processing (Phase 4)
    # ------------------------------------------------------------------

    def process_results(self, analysis: dict) -> dict:
        """Process reflection results: update templates, adjust weights, create meta-knowledge.

        Args:
            analysis: Combined analysis from dimensions 1-3.
        """
        procedural_updates = 0
        semantic_adjustments = 0
        meta_knowledge_id = None

        try:
            # Update procedural templates: slightly reduce weight on implicated templates
            # Navigate: effect_attribution → data → analysis (full_reflection structure)
            dim1_data = analysis.get("effect_attribution", {}).get("data", {})
            dim1 = dim1_data.get("analysis", {})
            root_cause = dim1.get("root_cause", "")

            if root_cause and self._procedural:
                matching = self._procedural.match_by_keyword(root_cause[:100], limit=3)
                for t in matching:
                    current_w = t.metadata.get("activation_weight", 0.5)
                    new_w = max(0.1, current_w - 0.05)
                    self._procedural.update(t.id, activation_weight=new_w)
                    procedural_updates += 1

            # Adjust semantic weights
            dim3 = analysis.get("knowledge_gaps", {}).get("data", {})
            existing = dim3.get("existing_in_semantic", [])
            if existing and self._semantic:
                for entry in existing[:3]:
                    sem = self._semantic.get(entry["id"])
                    if sem and sem.confidence > 0.3:
                        self._semantic.update(entry["id"], confidence=sem.confidence - 0.05)
                        semantic_adjustments += 1

            # Create meta-knowledge entry
            if root_cause:
                meta_text = (
                    f"[Meta-Knowledge] I tend to encounter issues with: {root_cause[:200]}\n"
                    f"Consider verifying related assumptions before similar tasks."
                )
                ep = self._episodic.store(
                    content=meta_text,
                    session_id="metacognition",
                    metadata={"meta_knowledge": True, "reflection": True},
                )
                meta_knowledge_id = ep.id

            return {
                "status": "ok",
                "data": {
                    "procedural_updates": procedural_updates,
                    "semantic_adjustments": semantic_adjustments,
                    "meta_knowledge_id": meta_knowledge_id,
                },
            }
        except Exception as exc:
            return {"status": "error", "error": str(exc)}

    # ------------------------------------------------------------------
    # Full 4-Dimension Reflection (Phase 4 orchestrator)
    # ------------------------------------------------------------------

    def full_reflection(
        self,
        task_description: str,
        outcome: str = "failure",
    ) -> dict:
        """Orchestrate all 4 dimensions of self-reflection.

        Each dimension runs sequentially. If one fails, the next proceeds.
        """
        dimensions = {}

        # Dimension 1: Effect Attribution (Phase 3)
        dim1 = self.reflect_on_failure(task_description, outcome)
        dimensions["effect_attribution"] = dim1

        # Dimension 2: Strategy Effectiveness
        dim2 = self.evaluate_strategy()
        dimensions["strategy_effectiveness"] = dim2

        # Dimension 3: Knowledge Gap Detection
        dim3 = self.detect_knowledge_gaps(task_description)
        dimensions["knowledge_gaps"] = dim3

        # Dimension 4: Result Processing
        dim4 = self.process_results(dimensions)
        dimensions["result_processing"] = dim4

        # Overall assessment
        overall = "completed"
        failed_dims = [k for k, v in dimensions.items() if v.get("status") == "error"]
        if failed_dims:
            overall = f"partial — {len(failed_dims)} dimensions failed: {failed_dims}"

        return {
            "status": "ok",
            "data": {
                "dimensions": dimensions,
                "overall_assessment": overall,
            },
        }
