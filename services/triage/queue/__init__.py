"""Public API for P2-02 sampling, clustering and daily review-queue construction.

This package is offline and stdlib-only apart from the shared settings base. It
consumes turn events and failure scores through injectable protocols, so it does
not import or depend on the in-flight P2-01 scoring package. See ``README.md``
for the seam and the queue-item contract.
"""

from __future__ import annotations

from .clustering import Cluster, ClusterItem, cluster_turns, normalise_text, text_similarity
from .queue import (
    DEFAULT_FIXTURES_PATH,
    SCHEMA_VERSION,
    ExplanationScoreSource,
    FixtureTurnSource,
    InMemoryScoreSource,
    InMemoryTurnSource,
    LowConfidenceSpan,
    QueueError,
    QueueItem,
    QueueReport,
    QueueStats,
    ScoreFileSource,
    ScoreSnapshot,
    ScoreSource,
    TurnRecord,
    TurnSource,
    build_queue,
    main,
)
from .sampling import (
    FAILURE_SLICE,
    RANDOM_SLICE,
    rng_for_seed,
    select_blind_items,
    select_random_slice,
    slice_target,
)
from .settings import TriageQueueSettings

__all__ = [
    "DEFAULT_FIXTURES_PATH",
    "FAILURE_SLICE",
    "RANDOM_SLICE",
    "SCHEMA_VERSION",
    "Cluster",
    "ClusterItem",
    "ExplanationScoreSource",
    "FixtureTurnSource",
    "InMemoryScoreSource",
    "InMemoryTurnSource",
    "LowConfidenceSpan",
    "QueueError",
    "QueueItem",
    "QueueReport",
    "QueueStats",
    "ScoreFileSource",
    "ScoreSnapshot",
    "ScoreSource",
    "TriageQueueSettings",
    "TurnRecord",
    "TurnSource",
    "build_queue",
    "cluster_turns",
    "main",
    "normalise_text",
    "rng_for_seed",
    "select_blind_items",
    "select_random_slice",
    "slice_target",
    "text_similarity",
]
