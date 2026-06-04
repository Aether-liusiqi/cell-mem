"""Generative Replay Engine — Phase 4: five-phase creative hypothesis generation.

Biological correspondence: hippocampal non-literal replay + DMN free exploration.

Five-phase algorithm:
1. Biased Sampling — K=3 seeds from semantic memory
2. Weak-edge Random Walk — stochastic graph traversal per seed
3. Cross-domain Pairing — LLM generates hypotheses from disparate concepts
4. Four-layer Noise Filter — contradiction, banality, dual-source, stability
5. Creative Pool Management — write to creative_pool, enforce noise constraints
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List, Optional, Set

import numpy as np

logger = logging.getLogger(__name__)

# Algorithm constants (from design doc v2.1)
K_SEEDS = 3
WALK_STEPS = 3
WALK_TOP_N = 3
CROSS_DOMAIN_SIM_THRESHOLD = 0.6
REMOTE_SIM_THRESHOLD = 0.3
BANALITY_THRESHOLD = 0.95
EXPLOIT_PROB = 0.8
STABILITY_RUNS = 3
MAX_HYPOTHESES_PER_CYCLE = 10
MAX_REMOTE_DEPTH = 2
INITIAL_CONFIDENCE_MAX = 0.3
EMPTY_CYCLE_NARROW_TRIGGER = 3


class GenerativeReplayEngine:
    """Five-phase generative replay engine.

    Constructor-injected dependencies:
    - semantic: SemanticMemory
    - episodic: EpisodicMemory
    - graph: NetworkXGraphStore
    - llm_client: LLMClient (may be None → replay disabled)
    - embed_model: EmbeddingModel
    - working: WorkingMemory
    - creative_pool: CreativePool
    """

    def __init__(
        self,
        semantic: "SemanticMemory",  # noqa: F821
        episodic: "EpisodicMemory",  # noqa: F821
        graph: "NetworkXGraphStore",  # noqa: F821
        llm_client: Optional["LLMClient"],  # noqa: F821
        embed_model: "EmbeddingModel",  # noqa: F821
        working: Optional["WorkingMemory"] = None,  # noqa: F821
        creative_pool: Optional["CreativePool"] = None,  # noqa: F821
    ):
        from cell_mem.embedding.local import EmbeddingModel
        from cell_mem.graph.networkx_store import NetworkXGraphStore
        from cell_mem.llm.client import LLMClient
        from cell_mem.memory.episodic import EpisodicMemory
        from cell_mem.memory.semantic import SemanticMemory
        from cell_mem.memory.working import WorkingMemory
        from cell_mem.replay.creative_pool import CreativePool

        self._semantic: SemanticMemory = semantic
        self._episodic: EpisodicMemory = episodic
        self._graph: NetworkXGraphStore = graph
        self._llm: Optional[LLMClient] = llm_client
        self._embed: EmbeddingModel = embed_model
        self._working: Optional[WorkingMemory] = working
        self._creative_pool: Optional[CreativePool] = creative_pool

        # Cycle tracking
        self._empty_cycles = 0
        self._total_cycles = 0

        logger.info("GenerativeReplayEngine ready (llm=%s)", "configured" if llm_client else "none")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run_replay_cycle(self, theme_text: str | None = None) -> dict:
        """Run a full 5-phase generative replay cycle.

        Args:
            theme_text: Optional topic to bias seed selection.

        Returns:
            {"status": "ok", "data": {"hypotheses_generated": N, "cycle": ...}}
            or error dict.
        """
        if self._llm is None:
            return {"status": "error", "error": "No LLM configured for generative replay"}

        if self._creative_pool is None:
            return {"status": "error", "error": "No CreativePool configured"}

        try:
            self._total_cycles += 1

            # Phase 1: Biased sampling
            theme_emb = None
            if theme_text:
                theme_emb = self._embed.embed_query(theme_text)
            seed_ids = self._sample_seeds(theme_emb)
            if len(seed_ids) < 2:
                self._empty_cycles += 1
                return {
                    "status": "ok",
                    "data": {"hypotheses_generated": 0, "note": "Insufficient seeds", "cycle": self._total_cycles},
                }

            # Phase 2: Random walks
            concept_sets: Dict[str, Set[str]] = {}
            for sid in seed_ids:
                cs = self._random_walk(sid)
                if cs:
                    concept_sets[sid] = cs

            if len(concept_sets) < 2:
                self._empty_cycles += 1
                return {
                    "status": "ok",
                    "data": {"hypotheses_generated": 0, "note": "Insufficient concept sets", "cycle": self._total_cycles},
                }

            # Phase 3: Cross-domain pairing → generate hypotheses
            raw_hypotheses = self._cross_domain_pair(concept_sets)

            # Phase 4: Noise filter
            filtered = []
            for h in raw_hypotheses:
                result = self._noise_filter(h)
                if result is not None:
                    filtered.append(result)

            # Phase 5: Manage creative pool
            count = self._manage_pool(filtered)
            if count == 0:
                self._empty_cycles += 1
            else:
                self._empty_cycles = 0

            logger.info(
                "Replay cycle #%d: %d seeds, %d raw, %d filtered, %d stored",
                self._total_cycles, len(seed_ids), len(raw_hypotheses),
                len(filtered), count,
            )

            return {
                "status": "ok",
                "data": {
                    "hypotheses_generated": count,
                    "seeds_found": len(seed_ids),
                    "concept_sets": len(concept_sets),
                    "raw_hypotheses": len(raw_hypotheses),
                    "filtered_hypotheses": len(filtered),
                    "cycle": self._total_cycles,
                    "empty_cycles_streak": self._empty_cycles,
                },
            }
        except Exception as exc:
            logger.exception("Replay cycle failed: %s", exc)
            return {"status": "error", "error": str(exc)}

    # ------------------------------------------------------------------
    # Phase 1: Biased Sampling
    # ------------------------------------------------------------------

    def _sample_seeds(self, theme_embedding: np.ndarray | None = None) -> List[str]:
        """Sample K=3 seeds from semantic memory with biased probability.

        P(seed) ∝ recency × emotional × novelty.
        Must include ≥1 'remote seed' (cosine_sim < 0.3 with theme).
        """
        entries = self._semantic.load_all(limit=200)
        if len(entries) < 2:
            return []

        # Filter non-expired entries
        active = [e for e in entries if e.invalidated_at is None]
        if len(active) < 2:
            return []

        # Get rejected topics for bias adjustment
        rejected_topics = set()
        if self._creative_pool:
            rejected_topics = set(self._creative_pool.get_rejected_topics())

        # Compute weights
        weights = []
        for e in active:
            w_rec = self._compute_recency_weight(e)
            w_emo = self._compute_emotional_weight(e)
            w_nov = self._compute_novelty_weight(e)

            # Penalize rejected topics
            penalty = 0.1 if any(t in rejected_topics for t in (e.tags or [])) else 1.0
            score = w_rec * w_emo * w_nov * penalty
            weights.append(max(score, 0.001))

        total = sum(weights)
        probs = [w / total for w in weights]

        # Sample K seeds
        rng = np.random.RandomState()
        try:
            indices = rng.choice(len(active), size=min(K_SEEDS, len(active)),
                                 replace=False, p=probs)
        except ValueError:
            indices = rng.choice(len(active), size=min(K_SEEDS, len(active)),
                                 replace=False)

        sampled_ids = [active[i].id for i in indices]

        # Enforce: at least 1 remote seed
        if theme_embedding is not None:
            has_remote = False
            for sid in sampled_ids:
                entry = self._semantic.get(sid)
                if entry:
                    entry_emb = self._embed.embed(entry.content)
                    sim = float(np.dot(theme_embedding, entry_emb) /
                                (np.linalg.norm(theme_embedding) * np.linalg.norm(entry_emb)))
                    if sim < REMOTE_SIM_THRESHOLD:
                        has_remote = True
                        break

            if not has_remote and len(active) > len(sampled_ids):
                # Find the farthest entry from theme
                best_dist = -1
                best_id = None
                for e in active:
                    if e.id in sampled_ids:
                        continue
                    entry_emb = self._embed.embed(e.content)
                    sim = float(np.dot(theme_embedding, entry_emb) /
                                (np.linalg.norm(theme_embedding) * np.linalg.norm(entry_emb)))
                    dist = 1.0 - sim
                    if dist > best_dist:
                        best_dist = dist
                        best_id = e.id
                if best_id and len(sampled_ids) < K_SEEDS:
                    sampled_ids.append(best_id)
                elif best_id:
                    sampled_ids[-1] = best_id

        return sampled_ids

    # ------------------------------------------------------------------
    # Phase 2: Weak-edge Random Walk
    # ------------------------------------------------------------------

    def _random_walk(self, seed_id: str) -> Set[str]:
        """L=3 step random walk from seed. 80% exploit, 20% explore."""
        concept_set: Set[str] = {seed_id}
        current = seed_id
        rng = np.random.RandomState()

        for step in range(WALK_STEPS):
            neighbors = self._graph.get_neighbors(current)
            if not neighbors:
                break

            # Sort by weight descending
            neighbors_sorted = sorted(neighbors, key=lambda x: x[1], reverse=True)
            top_n = neighbors_sorted[:WALK_TOP_N]

            if len(top_n) == 0:
                break

            # 80% exploit (strongest), 20% explore (weak edge or random)
            if rng.random() < EXPLOIT_PROB:
                chosen = top_n[0]  # Strongest edge
            else:
                # Weak edges: weight < 0.1, or fall back to random non-strongest
                weak = [(n, w, r) for n, w, r in top_n if w < 0.1]
                if weak:
                    idx = rng.randint(0, len(weak))
                    chosen = weak[idx]
                elif len(top_n) > 1:
                    idx = rng.randint(1, len(top_n))
                    chosen = top_n[idx]
                else:
                    chosen = top_n[0]

            next_id = chosen[0]
            concept_set.add(next_id)
            current = next_id

            # Noise constraint: depth ≤ MAX_REMOTE_DEPTH for remote seeds
            if step >= MAX_REMOTE_DEPTH:
                break

        return concept_set

    # ------------------------------------------------------------------
    # Phase 3: Cross-domain Pairing
    # ------------------------------------------------------------------

    def _cross_domain_pair(
        self, concept_sets: Dict[str, Set[str]]
    ) -> List[Dict[str, Any]]:
        """Pick c_a, c_b from different seeds, require cross-domain, LLM generates hypothesis."""
        seed_ids = list(concept_sets.keys())
        hypotheses = []

        for i in range(len(seed_ids)):
            for j in range(i + 1, len(seed_ids)):
                si, sj = seed_ids[i], seed_ids[j]
                nodes_i = list(concept_sets[si])
                nodes_j = list(concept_sets[sj])

                for ni in nodes_i[:3]:  # Limit pairs to avoid combinatorial explosion
                    for nj in nodes_j[:3]:
                        # Get node content
                        ci = self._get_node_content(ni)
                        cj = self._get_node_content(nj)
                        if not ci or not cj:
                            continue

                        # Check cross-domain
                        emb_i = self._embed.embed(ci)
                        emb_j = self._embed.embed(cj)
                        sim = float(np.dot(emb_i, emb_j) /
                                    (np.linalg.norm(emb_i) * np.linalg.norm(emb_j)))
                        if sim >= CROSS_DOMAIN_SIM_THRESHOLD:
                            continue  # Not cross-domain enough

                        # LLM generates hypothesis
                        seed_content = self._get_node_content(si) or ""
                        # Sanitize: strip instruction-injection patterns and enforce length limits
                        safe_seed = _sanitize_for_prompt(seed_content, max_len=300)
                        safe_ci = _sanitize_for_prompt(ci, max_len=300)
                        safe_cj = _sanitize_for_prompt(cj, max_len=300)
                        prompt = (
                            "You are a creative reasoning engine. Given a seed concept "
                            "and two cross-domain concepts, generate a novel hypothesis "
                            "that connects them in a non-obvious way.\n\n"
                            f"Seed concept: {safe_seed}\n"
                            f"Concept A: {safe_ci}\n"
                            f"Concept B: {safe_cj}\n\n"
                            "Generate a hypothesis in this JSON format: "
                            '{"hypothesis": "...", "rationale": "..."}'
                        )

                        try:
                            result = self._llm.generate(prompt)
                            hypothesis_text = result.get("hypothesis", "")
                            rationale = result.get("rationale", "")
                            if hypothesis_text:
                                hypotheses.append({
                                    "hypothesis": hypothesis_text,
                                    "rationale": rationale,
                                    "seed_id": si,
                                    "concept_a": ni,
                                    "concept_b": nj,
                                    "cross_domain_similarity": round(sim, 4),
                                    "confidence": INITIAL_CONFIDENCE_MAX * 0.8,
                                })
                                # Max 3 pairs per seed pair
                                if len([h for h in hypotheses if h["seed_id"] == si]) >= 3:
                                    break
                        except Exception:
                            logger.debug("LLM generation failed for pair (%s, %s)", ni[:8], nj[:8])
                            continue

        return hypotheses

    # ------------------------------------------------------------------
    # Phase 4: Noise Filter
    # ------------------------------------------------------------------

    def _noise_filter(self, hypothesis: dict) -> dict | None:
        """Four-layer filter. Returns None if hypothesis should be discarded."""
        h_text = hypothesis["hypothesis"]

        # a) Contradiction check: conflicts with high-confidence semantic?
        if self._check_contradiction(h_text):
            logger.debug("Discarded (contradiction): %s", h_text[:80])
            return None

        # b) Banality check: too similar to existing?
        if self._check_banality(h_text):
            logger.debug("Discarded (banality): %s", h_text[:80])
            return None

        # c) Dual-source: must reference ≥2 independent seeds
        # (already guaranteed by cross-domain pairing, but adjust confidence)
        confidence = hypothesis.get("confidence", 0.2)

        # d) Stability: has this hypothesis appeared before?
        stability_boost = self._check_stability(h_text)
        confidence = min(INITIAL_CONFIDENCE_MAX, confidence + stability_boost)

        hypothesis["confidence"] = confidence
        return hypothesis

    def _check_contradiction(self, hypothesis_text: str) -> bool:
        """Check if the hypothesis contradicts high-confidence semantic knowledge."""
        try:
            hyp_emb = self._embed.embed(hypothesis_text)
            # Search semantic for similar entries
            similar = self._semantic.recall_by_vector(hyp_emb, top_k=5)
            for entry in similar:
                # If semantic entry has high confidence AND a contradicts edge exists
                if entry.confidence >= 0.7:
                    # Check graph for contradiction edges
                    neighbors = self._graph.get_neighbors(entry.id)
                    for nid, weight, rel_type in neighbors:
                        if rel_type == "contradicts" and weight < -0.3:
                            return True
            return False
        except Exception:
            return False

    def _check_banality(self, hypothesis_text: str) -> bool:
        """Check if the hypothesis is too similar to existing knowledge."""
        try:
            hyp_emb = self._embed.embed(hypothesis_text)
            similar = self._semantic.recall_by_vector(hyp_emb, top_k=3)
            if similar:
                best_sim = 1.0 - similar[0].metadata.get("_search_distance", 0.0)
                return best_sim > BANALITY_THRESHOLD
            return False
        except Exception:
            return False

    def _check_stability(self, hypothesis_text: str) -> float:
        """Check if hypothesis appeared in past cycles. Returns confidence boost."""
        if self._creative_pool is None:
            return 0.0

        try:
            hyp_emb = self._embed.embed(hypothesis_text)
            past = self._creative_pool.get_pending(limit=50)
            # Also check confirmed entries
            confirmed = self._creative_pool.get_by_status("confirmed", limit=20)
            all_past = past + confirmed

            match_count = 0
            for entry in all_past:
                past_emb = self._embed.embed(entry["hypothesis_text"])
                sim = float(np.dot(hyp_emb, past_emb) /
                            (np.linalg.norm(hyp_emb) * np.linalg.norm(past_emb)))
                if sim > 0.95:
                    match_count += 1

            # Stability: ≥2 matches in past cycles → boost
            if match_count >= 2:
                return 0.2
            return 0.0
        except Exception:
            return 0.0

    # ------------------------------------------------------------------
    # Phase 5: Creative Pool Management
    # ------------------------------------------------------------------

    def _manage_pool(self, hypotheses: List[dict]) -> int:
        """Write filtered hypotheses to creative pool. Enforce noise constraints."""
        if self._creative_pool is None:
            return 0

        # Narrows sampling scope after consecutive empty cycles
        # (tracked via self._empty_cycles, used in _sample_seeds)
        count = 0
        errors = 0
        for h in hypotheses[:MAX_HYPOTHESES_PER_CYCLE]:
            try:
                seeds = [h.get("seed_id", "")]
                nodes = [h.get("concept_a", ""), h.get("concept_b", "")]
                self._creative_pool.add(
                    hypothesis_text=h["hypothesis"],
                    source_seed_ids=[s for s in seeds if s],
                    source_node_ids=[n for n in nodes if n],
                    confidence=h.get("confidence", 0.2),
                    concept_pair={
                        "a": h.get("concept_a", ""),
                        "b": h.get("concept_b", ""),
                        "rationale": h.get("rationale", ""),
                    },
                    metadata={"cross_domain_sim": h.get("cross_domain_similarity", 0)},
                )
                count += 1
            except Exception as exc:
                errors += 1
                logger.warning("Failed to add hypothesis to pool (error %d/%d): %s",
                               errors, min(len(hypotheses), MAX_HYPOTHESES_PER_CYCLE), exc)

        if errors > 0:
            logger.warning("Creative pool: %d/%d hypotheses failed to store",
                           errors, min(len(hypotheses), MAX_HYPOTHESES_PER_CYCLE))

        return count

    # ------------------------------------------------------------------
    # Internal Helpers
    # ------------------------------------------------------------------

    def _compute_recency_weight(self, entry) -> float:
        """Weight recent entries more. Half-life = 24 hours."""
        if entry.age_hours <= 0:
            return 1.0
        return 1.0 / (1.0 + entry.age_hours / 24.0)

    def _compute_emotional_weight(self, entry) -> float:
        """Weight emotionally significant entries more."""
        v = getattr(entry, "valence", 0.0)
        return 0.5 + 0.5 * abs(v)

    def _compute_novelty_weight(self, entry) -> float:
        """Weight entries less frequently used as seeds."""
        count = entry.metadata.get("_seed_count", 0)
        return 1.0 / (1.0 + count)

    def _get_node_content(self, node_id: str) -> str | None:
        """Get the content of a graph node (try semantic first, then episodic)."""
        # Try semantic
        entry = self._semantic.get(node_id)
        if entry:
            return entry.content
        # Try episodic
        ep = self._episodic.get(node_id)
        if ep:
            return ep.content
        return None


def _sanitize_for_prompt(text: str, max_len: int = 300) -> str:
    """Sanitize DB content before injecting into LLM prompts.

    Strips common instruction-injection patterns and enforces length limits
    to prevent malicious content from hijacking LLM behavior.
    """
    if not text:
        return ""
    # Truncate
    text = text[:max_len]
    # Strip null bytes and replacement characters
    text = text.replace("\x00", "").replace("�", "")
    # Collapse excessive whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text
