"""Consolidation processor — scoring, pattern detection, and scheduling."""

from cell_mem.consolidation.emotional import EmotionalScorer, RuleBasedScorer
from cell_mem.consolidation.scorer import ConsolidationScorer
from cell_mem.consolidation.detector import PatternDetector
from cell_mem.consolidation.scheduler import ConsolidationScheduler

__all__ = [
    "ConsolidationScorer",
    "PatternDetector",
    "ConsolidationScheduler",
    "EmotionalScorer",
    "RuleBasedScorer",
]
