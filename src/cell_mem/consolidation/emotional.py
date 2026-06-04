"""Emotional valence scorers for consolidation.

Abstract base class with degrade() chain, plus a rule-based implementation
that scores text heuristically (exclamation marks, emoji, all-caps, negation).
LLMScorer provides LLM-based scoring with automatic fallback.
"""

from __future__ import annotations

import logging
import re
from abc import ABC, abstractmethod
from typing import Tuple

logger = logging.getLogger(__name__)

# Emoji unicode ranges (common emotional emoji)
_EMOJI_PATTERN = re.compile(
    "[\U0001F300-\U0001F5FF\U0001F600-\U0001F64F\U0001F680-\U0001F6FF"
    "\U0001F900-\U0001F9FF\U0001FA00-\U0001FA6F\U0001FA70-\U0001FAFF"
    "\U00002702-\U000027B0\U000024C2-\U0001F251]",
    re.UNICODE,
)
_POSITIVE_EMOJI = {
    "😊", "😄", "👍", "❤", "😍", "🎉", "✨", "😁", "🙂", "💪",
    "✅", "👏", "🥳", "💯", "🔥", "😎", "🤩", "🙌",
}
_NEGATIVE_EMOJI = {
    "😡", "😢", "👎", "😞", "😠", "😤", "💔", "😩", "😰", "😭",
    "❌", "😱", "🤬", "👿", "💀", "😓", "😖", "😣",
}
_NEGATION_WORDS = {
    "not", "never", "no", "cannot", "can't", "don't", "won't",
    "isn't", "aren't", "wasn't", "weren't", "nothing", "nobody",
    "neither", "nor", "hardly", "barely",
}


class EmotionalScorer(ABC):
    """Abstract base for emotional valence scoring.

    Each scorer returns (valence: float in [-1, 1], confidence: float in [0, 1]).
    The degrade() method models emotional decay across reconsolidation cycles.
    """

    @abstractmethod
    def score(self, text: str) -> Tuple[float, float]:
        """Score text for emotional valence.

        Returns:
            (valence, confidence) — valence ∈ [-1, 1], confidence ∈ [0, 1]
        """
        ...

    @staticmethod
    def degrade(
        valence: float, confidence: float, steps: int = 1
    ) -> Tuple[float, float]:
        """Degrade emotional signal across reconsolidation cycles.

        Each step: valence *= 0.85, confidence *= 0.9.
        Models the natural decay of emotional salience over time.
        """
        for _ in range(steps):
            valence = valence * 0.85
            confidence = confidence * 0.9
        return valence, confidence


class RuleBasedScorer(EmotionalScorer):
    """Rule-based heuristic emotional valence scorer.

    Scoring rules (applied additively):
    - Exclamation marks: +0.2 per '!' (max +0.6)
    - Positive emoji: +0.15 each (max +0.45)
    - Negative emoji: -0.15 each (max -0.45)
    - ALL-CAPS words (3+ chars): +0.1 per word (max +0.3)
    - Negation words (not, never, no, etc.): flip valence sign
    - Base valence: 0.0 (neutral)

    Confidence: starts at 0.8, reduced by 0.15 per conflicting signal.
    Returns (0.0, 0.5) when no signals detected.
    """

    def score(self, text: str) -> Tuple[float, float]:
        """Score text for emotional valence."""
        valence = 0.0
        signal_count = 0

        # Rule 1: Exclamation marks
        excl_count = text.count("!")
        if excl_count > 0:
            valence += min(excl_count * 0.2, 0.6)
            signal_count += 1

        # Rule 2: Emoji
        emojis = _EMOJI_PATTERN.findall(text)
        pos_count = sum(1 for e in emojis if e in _POSITIVE_EMOJI)
        neg_count = sum(1 for e in emojis if e in _NEGATIVE_EMOJI)
        if pos_count > 0:
            valence += min(pos_count * 0.15, 0.45)
            signal_count += 1
        if neg_count > 0:
            valence -= min(neg_count * 0.15, 0.45)
            signal_count += 1

        # Rule 3: ALL-CAPS words
        caps_words = re.findall(r'\b[A-Z]{3,}\b', text)
        if caps_words:
            abs_boost = min(len(caps_words) * 0.1, 0.3)
            if valence >= 0:
                valence += abs_boost
            else:
                valence -= abs_boost
            signal_count += 1

        # Rule 4: Negation flip
        words_lower = set(text.lower().split())
        has_negation = bool(words_lower & _NEGATION_WORDS)
        if has_negation:
            valence = -valence
            signal_count += 1

        # Clamp
        valence = max(-1.0, min(1.0, valence))

        # Confidence: base 0.8, reduced by conflicting signals
        if signal_count == 0:
            return 0.0, 0.5

        confidence = max(0.3, 0.8 - (signal_count - 1) * 0.15)
        return valence, confidence


# ---------------------------------------------------------------------------
# LLM-based emotional scoring with automatic fallback chain
# ---------------------------------------------------------------------------


class LLMScorer(EmotionalScorer):
    """LLM-based emotional valence scorer.

    Sends text to an LLM with a structured prompt asking for valence
    and confidence. Returns (0.0, 0.0) on any failure — the FallbackScorer
    chain handles degradation.

    Constructor-injected dependencies:
    - llm_client: LLMClient (any backend)
    """

    def __init__(self, llm_client: "LLMClient"):  # noqa: F821
        from cell_mem.llm.client import LLMClient

        self._llm: LLMClient = llm_client

    def score(self, text: str) -> Tuple[float, float]:
        """Score text using LLM.

        Returns (0.0, 0.0) on any failure — caller should have a fallback.
        """
        if not text.strip():
            return 0.0, 0.5

        prompt = (
            "Analyze the emotional valence of this text. "
            "Valence: -1 = very negative (angry, sad, frustrated), "
            "0 = neutral, 1 = very positive (happy, excited, grateful). "
            "Confidence: 0 = completely uncertain, 1 = very certain.\n\n"
            f'Text: """{text}"""'
        )

        schema_hint = {
            "valence": "float between -1 and 1",
            "confidence": "float between 0 and 1",
        }

        try:
            result = self._llm.generate(prompt, schema_hint)
            v = float(result.get("valence", 0.0))
            c = float(result.get("confidence", 0.0))
            v = max(-1.0, min(1.0, v))
            c = max(0.0, min(1.0, c))
            logger.debug("LLMScorer: valence=%.2f confidence=%.2f", v, c)
            return v, c
        except Exception:
            logger.debug("LLMScorer failed, returning (0.0, 0.0)", exc_info=True)
            return 0.0, 0.0


class FallbackScorer(EmotionalScorer):
    """Chain of emotional scorers with degrade strategy.

    Tries scorers in order: primary → secondary → neutral default.
    Each scorer signals "can't handle this" by returning confidence < 0.3
    or by raising an exception. The chain falls through to the next scorer.

    Default chain when LLM is configured:
        LLMScorer → RuleBasedScorer → neutral (0.0, 0.5)

    When LLM is not configured, MemorySystem directly uses RuleBasedScorer,
    skipping the FallbackScorer entirely.
    """

    def __init__(
        self,
        primary: EmotionalScorer,
        secondary: EmotionalScorer | None = None,
    ):
        self._primary = primary
        self._secondary = secondary or RuleBasedScorer()

    def score(self, text: str) -> Tuple[float, float]:
        """Score text through the fallback chain.

        LLM → RuleBased → neutral default
        """
        # Tier 1: Primary (LLM)
        try:
            v, c = self._primary.score(text)
            if c >= 0.3:
                return v, c
            logger.debug("Primary scorer confidence too low (%.2f), degrading", c)
        except Exception:
            logger.debug("Primary scorer failed, degrading", exc_info=True)

        # Tier 2: Secondary (RuleBased)
        try:
            v, c = self._secondary.score(text)
            if c >= 0.3:
                return v, c
            logger.debug("Secondary scorer confidence too low (%.2f), degrading", c)
        except Exception:
            logger.debug("Secondary scorer failed, using neutral default", exc_info=True)

        # Tier 3: Neutral default
        return 0.0, 0.5
