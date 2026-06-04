"""Consolidation scoring — five-dimension model.

S_total = 1.5*S_emotional + 1.2*S_outcome + 1.0*S_repetition
        + 0.8*S_novelty + 1.0*S_recency

Deterministic scoring with default placeholders for
          S_emotional and S_outcome.
LLM-computed S_emotional and S_outcome replace defaults.
"""

from __future__ import annotations

import logging
import math
from typing import Optional

logger = logging.getLogger(__name__)

# Weight coefficients
W_EMOTIONAL = 1.5
W_OUTCOME = 1.2
W_REPETITION = 1.0
W_NOVELTY = 0.8
W_RECENCY = 1.0

# Recency half-life: 7 days = 168 hours
RECENCY_HALFLIFE_HOURS = 168.0

# Dynamic forget threshold: base value, scales with DB size
FORGET_BASE = 2.0
FORGET_SCALE = 0.3
FORGET_REF = 10000


class ConsolidationScorer:
    """Five-dimension consolidation scoring engine.

    Constructor-injected dependencies:
    - episodic: EpisodicMemory (for similarity queries)
    - emotional_scorer: EmotionalScorer (RuleBasedScorer by default)
    - embed_model: EmbeddingModel (for on-the-fly embedding)
    """

    def __init__(
        self,
        episodic: Optional["EpisodicMemory"] = None,  # noqa: F821
        emotional_scorer: Optional["EmotionalScorer"] = None,  # noqa: F821
        embed_model: Optional["EmbeddingModel"] = None,  # noqa: F821
    ):
        from cell_mem.consolidation.emotional import EmotionalScorer, RuleBasedScorer
        from cell_mem.embedding.local import EmbeddingModel
        from cell_mem.memory.episodic import EpisodicMemory

        self._episodic: Optional[EpisodicMemory] = episodic
        self._emotional: EmotionalScorer = emotional_scorer or RuleBasedScorer()
        self._embed: Optional[EmbeddingModel] = embed_model

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def score(self, episode) -> float:
        """Compute 5-dim consolidation score for an episode.

        Returns total score as float. Higher = retain, lower = forget candidate.
        """
        s_emo = self._compute_emotional(episode.valence)
        s_out = self._compute_outcome(episode)
        s_rep = self._compute_repetition(episode)
        s_nov = self._compute_novelty(episode)
        s_rec = self._compute_recency(episode)

        total = (
            W_EMOTIONAL * s_emo
            + W_OUTCOME * s_out
            + W_REPETITION * s_rep
            + W_NOVELTY * s_nov
            + W_RECENCY * s_rec
        )

        logger.debug(
            "Score %s: emo=%.3f out=%.3f rep=%.3f nov=%.3f rec=%.3f → %.3f",
            episode.id[:8], s_emo, s_out, s_rep, s_nov, s_rec, total,
        )
        return total

    def get_forget_threshold(self, total_episode_count: int) -> float:
        """Dynamic forget threshold: 2.0 * (1 + 0.3 * log(count/10000)).

        Scales with DB size so larger databases become more selective.
        """
        if total_episode_count == 0:
            return FORGET_BASE
        ratio = max(total_episode_count / FORGET_REF, 0.001)
        return FORGET_BASE * (1.0 + FORGET_SCALE * math.log(ratio))

    # ------------------------------------------------------------------
    # Dimension sub-scores
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_emotional(valence: float) -> float:
        """S_emotional = sigmoid(|valence|) * 1.5 (steep sigmoid).

        sigmoid(x) = 2/(1 + exp(-5*|v|)) - 1
        Neutral default (|v| ≈ 0): sigmoid(0) = 0, so we use 0.95 as baseline.
        """
        if abs(valence) < 0.001:
            return 0.95  # Neutral baseline
        steep = 2.0 / (1.0 + math.exp(-5.0 * abs(valence))) - 1.0
        return steep * 1.5

    @staticmethod
    def _compute_outcome(episode) -> float:
        """S_outcome = |outcome_valence| * 1.2.

        Reads from episode.metadata["outcome_valence"].
        Neutral default: 0.36 (= |0.3| * 1.2).
        """
        ov = episode.metadata.get("outcome_valence", 0.3)
        if ov is None:
            ov = 0.3
        return abs(float(ov)) * 1.2

    def _compute_repetition(self, episode) -> float:
        """S_repetition = min(1.0, similar_count/3) * 1.0.

        Uses FTS5 search to count similar episodes.
        At least 3 similar episodes to max out this dimension.
        """
        if self._episodic is None:
            return 0.0
        count = self._episodic.count_similar_fts(episode.content)
        return min(1.0, count / 3.0) * 1.0

    def _compute_novelty(self, episode) -> float:
        """S_novelty = (1 - max_similarity_to_existing) * 0.8.

        Uses vector search to find the most similar existing episode.
        """
        if self._episodic is None or self._embed is None:
            return 0.4
        try:
            query_vec = self._embed.embed(episode.content)
            max_sim = self._episodic.find_most_similar(query_vec, exclude_id=episode.id)
            return (1.0 - max_sim) * 0.8
        except Exception:
            return 0.4

    @staticmethod
    def _compute_recency(episode) -> float:
        """S_recency = exp(-age_hours/168) * 1.0.

        7-day half-life: after 168 hours score drops to ~37%.
        """
        return math.exp(-episode.age_hours / RECENCY_HALFLIFE_HOURS) * 1.0
