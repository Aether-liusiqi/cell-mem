"""Pattern detection — Phase 2b: DBSCAN clustering on 2048d projection vectors.

Uses sklearn.cluster.DBSCAN with cosine metric to group similar episodic
memories. Clusters of >= 3 episodes trigger automatic semantic entry creation.

Phase 3: LLM-assisted pattern detection on top of DBSCAN pre-filtering.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

# DBSCAN parameters (from design doc v2.1)
DBSCAN_EPS = 0.3
DBSCAN_MIN_SAMPLES = 3
DBSCAN_METRIC = "cosine"


class PatternProposal:
    """A detected pattern from DBSCAN clustering — ready for semantic creation."""

    def __init__(
        self,
        cluster_id: int,
        episode_ids: List[str],
        centroid_text: str,
        initial_confidence: float = 0.35,
    ):
        self.cluster_id = cluster_id
        self.episode_ids = episode_ids
        self.centroid_text = centroid_text
        self.initial_confidence = initial_confidence

    def to_dict(self) -> Dict[str, Any]:
        return {
            "cluster_id": self.cluster_id,
            "episode_ids": self.episode_ids,
            "centroid_text": self.centroid_text[:200],
            "initial_confidence": self.initial_confidence,
            "episode_count": len(self.episode_ids),
        }


class PatternDetector:
    """Detect patterns from episodic memories via DBSCAN clustering.

    Lazy-imports sklearn inside detect() to avoid hard dependency at import time.

    Constructor-injected dependencies:
    - episodic: EpisodicMemory (to load projection vectors)
    - semantic: SemanticMemory (to auto-create semantic entries from proposals)
    """

    def __init__(
        self,
        episodic: Optional["EpisodicMemory"] = None,  # noqa: F821
        semantic: Optional["SemanticMemory"] = None,  # noqa: F821
        llm_client: Optional["LLMClient"] = None,  # noqa: F821
    ):
        from cell_mem.llm.client import LLMClient
        from cell_mem.memory.episodic import EpisodicMemory
        from cell_mem.memory.semantic import SemanticMemory

        self._episodic: Optional[EpisodicMemory] = episodic
        self._semantic: Optional[SemanticMemory] = semantic
        self._llm: Optional[LLMClient] = llm_client

    def detect(self, episodes: List) -> List[PatternProposal]:
        """Run DBSCAN clustering, then optionally review with LLM.

        Args:
            episodes: List of MemoryObject to cluster.

        Returns:
            List of PatternProposal for validated clusters.
        """
        # Step 1: DBSCAN clustering (Phase 2b)
        proposals = self._dbscan_cluster(episodes)

        # Step 2: LLM post-hoc review (Phase 3)
        if self._llm is not None and proposals:
            proposals = self._llm_review_clusters(proposals)

        return proposals

    def _dbscan_cluster(self, episodes: List) -> List[PatternProposal]:
        """Run DBSCAN clustering on 2048d projection vectors.

        Args:
            episodes: List of MemoryObject to cluster. Each must have a
                      projection_vector stored in episodic_memory.

        Returns:
            List of PatternProposal for clusters with >= 3 members.
            Noise points (label=-1) and clusters with < 3 members are excluded.
        """
        if len(episodes) < DBSCAN_MIN_SAMPLES:
            logger.debug(
                "Too few episodes for clustering: %d < %d",
                len(episodes), DBSCAN_MIN_SAMPLES,
            )
            return []

        # Lazy-import sklearn (avoid import-time cost for module loading)
        try:
            from sklearn.cluster import DBSCAN
        except ImportError:
            logger.error("scikit-learn not installed; pattern detection unavailable")
            return []

        # Build feature matrix from projection vectors
        # Load from DB rather than assuming episodes carry them in-memory
        proj_map = {}
        if self._episodic is not None:
            proj_map = self._episodic.load_all_projection_vectors()

        vectors = []
        valid_episodes = []
        for ep in episodes:
            pv = proj_map.get(ep.id)
            if pv is not None and len(pv) > 0:
                vectors.append(pv)
                valid_episodes.append(ep)

        if len(valid_episodes) < DBSCAN_MIN_SAMPLES:
            logger.debug(
                "Too few episodes with projection vectors: %d", len(valid_episodes)
            )
            return []

        X = np.vstack(vectors)
        logger.info(
            "Clustering %d episodes in %d-dim space (eps=%.2f, min_samples=%d, metric=%s)",
            len(vectors), X.shape[1], DBSCAN_EPS, DBSCAN_MIN_SAMPLES, DBSCAN_METRIC,
        )

        clustering = DBSCAN(
            eps=DBSCAN_EPS,
            min_samples=DBSCAN_MIN_SAMPLES,
            metric=DBSCAN_METRIC,
        ).fit(X)

        labels = clustering.labels_

        # Group by cluster label
        clusters: Dict[int, List] = {}
        for idx, label in enumerate(labels):
            if label == -1:
                continue  # Noise
            clusters.setdefault(int(label), []).append(valid_episodes[idx])

        # Build proposals
        proposals = []
        for cluster_id, members in clusters.items():
            if len(members) < DBSCAN_MIN_SAMPLES:
                continue

            # Centroid text: longest content as exemplar
            exemplar = max(members, key=lambda m: len(m.content))
            episode_ids = [m.id for m in members]

            proposal = PatternProposal(
                cluster_id=cluster_id,
                episode_ids=episode_ids,
                centroid_text=exemplar.content[:500],
                initial_confidence=0.35,
            )
            proposals.append(proposal)

            # Auto-create semantic entry
            if self._semantic is not None:
                self._create_semantic_entry(proposal)

        logger.info(
            "Detected %d pattern(s) from %d episodes (%d noise points)",
            len(proposals), len(valid_episodes), list(labels).count(-1),
        )
        return proposals

    def _create_semantic_entry(self, proposal: PatternProposal) -> None:
        """Create a semantic memory entry from a pattern proposal."""
        if self._semantic is None:
            return
        try:
            label = proposal.centroid_text[:300]
            self._semantic.add(
                content=f"[Pattern] {label}",
                confidence=proposal.initial_confidence,
                source_references=proposal.episode_ids,
                metadata={
                    "pattern_cluster_id": proposal.cluster_id,
                    "episode_count": len(proposal.episode_ids),
                    "detection_method": "dbscan",
                },
            )
            logger.info(
                "Created semantic entry for cluster %d (%d episodes)",
                proposal.cluster_id, len(proposal.episode_ids),
            )
        except Exception as exc:
            logger.warning(
                "Failed to create semantic entry for cluster %d: %s",
                proposal.cluster_id, exc,
            )

    def _llm_review_clusters(
        self, proposals: List[PatternProposal]
    ) -> List[PatternProposal]:
        """Review DBSCAN clusters with LLM post-hoc.

        For each cluster, asks LLM whether it represents a coherent pattern.
        Filters out false positives. Valid clusters get LLM-generated labels
        and adjusted confidence scores.

        If LLM call fails for a cluster, the original proposal is kept
        (fail-open — don't discard DBSCAN results because LLM is down).
        """
        if self._llm is None:
            return proposals

        validated = []
        for p in proposals:
            try:
                # Build a prompt with cluster members (up to 5 exemplars)
                exemplars = p.episode_ids[:5]
                episodes_text = ""
                if self._episodic is not None:
                    for eid in exemplars:
                        ep = self._episodic.get(eid)
                        if ep:
                            episodes_text += f"- {ep.content[:200]}\n"

                if not episodes_text.strip():
                    validated.append(p)
                    continue

                prompt = (
                    "The following memory episodes were clustered together by "
                    "a pattern detection algorithm. Determine if they represent "
                    "a coherent, meaningful pattern (e.g., same type of bug, "
                    "same workflow, same user preference).\n\n"
                    f"Episodes:\n{episodes_text}\n"
                    "Is this a coherent pattern? If yes, provide a short label "
                    "(max 10 words) and a confidence score (0.0-1.0)."
                )

                schema_hint = {
                    "is_pattern": "boolean",
                    "label": "string (max 10 words)",
                    "confidence": "float between 0 and 1",
                }

                result = self._llm.generate(prompt, schema_hint)

                if result.get("is_pattern", False):
                    new_label = result.get("label", "")
                    new_conf = float(result.get("confidence", p.initial_confidence))
                    if new_label:
                        p.centroid_text = new_label
                    p.initial_confidence = max(0.2, min(0.7, new_conf))
                    # Update semantic entry that was already created by _dbscan_cluster
                    # We don't re-create — just note the LLM review in metadata
                    p._llm_reviewed = True
                    validated.append(p)
                    logger.debug(
                        "LLM validated cluster %d: '%s' (conf=%.2f)",
                        p.cluster_id, p.centroid_text, p.initial_confidence,
                    )
                else:
                    logger.debug(
                        "LLM rejected cluster %d as not a coherent pattern",
                        p.cluster_id,
                    )
                    # Don't add to validated — this cluster is filtered out
            except Exception:
                logger.debug(
                    "LLM review failed for cluster %d, keeping original",
                    p.cluster_id, exc_info=True,
                )
                validated.append(p)

        return validated
