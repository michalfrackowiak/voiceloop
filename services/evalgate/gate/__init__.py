"""P2-09 gate and promotion logic: rules, conversion hook, alias flip and audit."""

from __future__ import annotations

from .aliases import (
    CHALLENGER_ALIAS,
    CHAMPION_ALIAS,
    AliasStore,
    InMemoryAliasStore,
    promote,
    rollback,
    rollback_promotion,
)
from .audit import (
    AUDIT_TASK_ID,
    AuditLog,
    PromotionAction,
    PromotionAuditLog,
    PromotionRecord,
)
from .gate import (
    ArtifactConverter,
    ArtifactEvaluator,
    ConvertedArtifact,
    Gate,
    GateResult,
    PrefixArtifactConverter,
    StaticReportEvaluator,
)
from .rules import (
    ACCEPTED_EVAL_SETS,
    CRITICAL_SLICES,
    GateDecision,
    GateError,
    decide_gate,
)
from .settings import GateSettings

__all__ = [
    "ACCEPTED_EVAL_SETS",
    "AUDIT_TASK_ID",
    "CHALLENGER_ALIAS",
    "CHAMPION_ALIAS",
    "CRITICAL_SLICES",
    "AliasStore",
    "ArtifactConverter",
    "ArtifactEvaluator",
    "AuditLog",
    "ConvertedArtifact",
    "Gate",
    "GateDecision",
    "GateError",
    "GateResult",
    "GateSettings",
    "InMemoryAliasStore",
    "PrefixArtifactConverter",
    "PromotionAction",
    "PromotionAuditLog",
    "PromotionRecord",
    "StaticReportEvaluator",
    "decide_gate",
    "promote",
    "rollback",
    "rollback_promotion",
]
