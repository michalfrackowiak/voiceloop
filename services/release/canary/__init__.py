"""P4-01 canary with automatic, bounded rollback (synthetic rehearsal only).

This package is a background-only reference implementation. Every input is an
injected deterministic fake, every output is labelled
``SYNTHETIC/FIXTURE STAND-IN - NOT LIVE CANARY EVIDENCE``, and the only alias
write it can perform is a single one-step rollback of the current alias.

It consumes the P2-09 gate's promotion record and alias-store seam, the P3-06
read-only alias-registry protocol and the P3-07 shadow outcome/digest helpers
without modifying any of those packages.
"""

from __future__ import annotations

from .metrics import (
    CanaryMetrics,
    CanaryPair,
    PredictionRecord,
    RegressionVerdict,
    detect_regression,
    score_canary,
)
from .rollback import (
    AliasStore,
    CanaryDeployment,
    CollectingRollbackAudit,
    InMemoryRollbackAudit,
    RollbackAuditLog,
    RollbackController,
    RollbackRecord,
    RollbackResult,
    canonical_json,
    sha256_hex,
)
from .runner import (
    CanaryAuditEvent,
    CanaryCallObservation,
    CanaryRunner,
    CanaryTickResult,
    CollectingCanaryAudit,
    FakeEventSource,
    run_fixture_rehearsal,
)
from .settings import (
    DEFAULT_TASK_ID,
    MAX_CANARY_FRACTION,
    MAX_WINDOW_SECONDS,
    SYNTHETIC_LABEL,
    CanarySettings,
)
from .split import Arm, SplitAssignment, TrafficSplit

__all__ = [
    "DEFAULT_TASK_ID",
    "MAX_CANARY_FRACTION",
    "MAX_WINDOW_SECONDS",
    "SYNTHETIC_LABEL",
    "AliasStore",
    "Arm",
    "CanaryAuditEvent",
    "CanaryCallObservation",
    "CanaryDeployment",
    "CanaryMetrics",
    "CanaryPair",
    "CanaryRunner",
    "CanarySettings",
    "CanaryTickResult",
    "CollectingCanaryAudit",
    "CollectingRollbackAudit",
    "FakeEventSource",
    "InMemoryRollbackAudit",
    "PredictionRecord",
    "RegressionVerdict",
    "RollbackAuditLog",
    "RollbackController",
    "RollbackRecord",
    "RollbackResult",
    "SplitAssignment",
    "TrafficSplit",
    "canonical_json",
    "detect_regression",
    "run_fixture_rehearsal",
    "score_canary",
    "sha256_hex",
]
