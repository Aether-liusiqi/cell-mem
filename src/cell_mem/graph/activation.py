"""BFS activation spreading on the memory association graph (Phase 2a).

Implements the spreading activation algorithm from the design doc:
1. Seed nodes start with activation = 1.0
2. BFS propagation: activation[neighbor] += activation[node] * weight * decay
3. Multi-path: take max activation per node (sublinear dendritic integration)
4. ReLU: negative activation clamped to 0
5. Prune below threshold
"""

from __future__ import annotations

import logging
from collections import deque
from typing import Dict, List

logger = logging.getLogger(__name__)


def spread_activation(
    seed_ids: List[str],
    graph_store: "GraphStore",  # noqa: F821
    max_hops: int = 2,
    decay: float = 0.3,
    threshold: float = 0.01,
) -> Dict[str, float]:
    """Spread activation from seed nodes through the association graph.

    BFS-based algorithm that propagates activation along outgoing edges,
    decaying with each hop. Multiple incoming activations take the max
    value (sublinear dendritic integration). Negative edges are skipped
    (they'd be ReLU-clamped to 0 anyway).

    Args:
        seed_ids: Starting node IDs (e.g., top results from vector search).
        graph_store: GraphStore with get_out_edges support.
        max_hops: Maximum BFS depth (default 2, trisynaptic circuit analog).
        decay: Activation decay per hop (default 0.3, CA3 sparsity analog).
        threshold: Only nodes with activation > threshold are returned.

    Returns:
        Dict mapping node_id to final activation score, sorted by activation
        descending. Seed nodes ARE included (caller can filter them out
        by comparing with the original search result set).
    """
    if not seed_ids:
        return {}

    # Initialize BFS queue: (node_id, depth, incoming_activation)
    queue = deque()
    for seed in seed_ids:
        queue.append((seed, 0, 1.0))

    # Track max activation per node (sublinear dendritic integration)
    activations: Dict[str, float] = {}
    # Track best depth visited for cycle avoidance
    visited_depth: Dict[str, int] = {}

    while queue:
        node_id, depth, value = queue.popleft()

        # Prune below threshold
        if value < threshold:
            continue

        # Take max activation for multi-path inputs
        if node_id in activations and activations[node_id] >= value:
            continue

        # Skip if already visited at same or shallower depth
        if node_id in visited_depth and visited_depth[node_id] <= depth:
            if activations.get(node_id, 0.0) >= value:
                continue

        activations[node_id] = max(activations.get(node_id, 0.0), value)
        visited_depth[node_id] = depth

        # Stop at max_hops
        if depth >= max_hops:
            continue

        # Spread to neighbors
        out_edges = graph_store.get_out_edges(node_id)
        for _source, target, weight, _rel_type in out_edges:
            # Negative edges inhibit — skip them
            if weight <= 0:
                continue

            next_value = value * weight * decay
            if next_value >= threshold:
                queue.append((target, depth + 1, next_value))

    # Sort by activation descending
    return dict(
        sorted(activations.items(), key=lambda x: x[1], reverse=True)
    )
