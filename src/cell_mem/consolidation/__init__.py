"""Consolidation processor — scoring, pattern detection, scheduling, and user preference pipeline."""

from cell_mem.consolidation.emotional import EmotionalScorer, RuleBasedScorer
from cell_mem.consolidation.scorer import ConsolidationScorer
from cell_mem.consolidation.detector import PatternDetector
from cell_mem.consolidation.scheduler import ConsolidationScheduler
from cell_mem.consolidation.preference import (
    PreferenceSignalDetector,
    PreferenceExtractor,
    PreferenceProcessor,
    PreferenceInjector,
)

__all__ = [
    "ConsolidationScorer",
    "PatternDetector",
    "ConsolidationScheduler",
    "EmotionalScorer",
    "RuleBasedScorer",
    "PreferenceSignalDetector",
    "PreferenceExtractor",
    "PreferenceProcessor",
    "PreferenceInjector",
]
