"""P3-06 alias sync agent: read-only champion polling and transactional serving swaps."""

from __future__ import annotations

from .artifacts import (
    ArtifactRef,
    ArtifactStore,
    DownloadedArtifact,
    FakeArtifactStore,
    hash_payload,
    hash_tree,
    verify_artifact,
)
from .health import (
    FakeHealthProbe,
    HealthProbe,
    HealthResult,
    HealthStatus,
    ServingStateProbe,
)
from .registry import (
    CHAMPION_ALIAS,
    AliasState,
    ChampionRegistry,
    ChampionSnapshot,
    FakeChampionRegistry,
    GateAliasChampionRegistry,
    ReadOnlyAliasStore,
)
from .serving import (
    TARGET_ORDER,
    FakeServingAdapter,
    FakeServingStack,
    ServingAdapter,
    ServingStack,
    ServingState,
    ServingTarget,
    SwapPlan,
)
from .settings import ReleaseSyncSettings
from .sync import (
    AliasSyncAgent,
    CollectingLogger,
    LogEvent,
    StructlogEventLogger,
    StructuredLogger,
    SyncOutcome,
    SyncResult,
)

__all__ = [
    "CHAMPION_ALIAS",
    "TARGET_ORDER",
    "AliasState",
    "AliasSyncAgent",
    "ArtifactRef",
    "ArtifactStore",
    "ChampionRegistry",
    "ChampionSnapshot",
    "CollectingLogger",
    "DownloadedArtifact",
    "FakeArtifactStore",
    "FakeChampionRegistry",
    "FakeHealthProbe",
    "FakeServingAdapter",
    "FakeServingStack",
    "GateAliasChampionRegistry",
    "HealthProbe",
    "HealthResult",
    "HealthStatus",
    "LogEvent",
    "ReadOnlyAliasStore",
    "ReleaseSyncSettings",
    "ServingAdapter",
    "ServingStack",
    "ServingState",
    "ServingStateProbe",
    "ServingTarget",
    "StructlogEventLogger",
    "StructuredLogger",
    "SwapPlan",
    "SyncOutcome",
    "SyncResult",
    "hash_payload",
    "hash_tree",
    "verify_artifact",
]
