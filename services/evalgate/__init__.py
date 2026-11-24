"""Evaluation harness v0: metrics, paired bootstrap and contract reports."""

from __future__ import annotations

from .bootstrap import (
    ConfidenceInterval,
    bootstrap_deltas,
    paired_bootstrap_ci,
    percentile_interval,
)
from .metrics import (
    CORE_METRIC_NAMES,
    AlignmentCounts,
    E2EResult,
    EntityPatterns,
    EntityResult,
    EvaluationError,
    Prediction,
    SlotResult,
    TurnMetrics,
    aggregate_metrics,
    e2e_booking_accuracy,
    e2e_turn_result,
    entity_error,
    evaluate_turn,
    is_schema_valid,
    levenshtein_counts,
    schema_valid_rate,
    slot_f1,
    tokenize,
    word_error_rate,
)
from .report import REPORT_SCHEMA_VERSION, build_report, write_report
from .settings import ArtifactKind, EvalgateSettings, EvalSetName

__all__ = [
    "CORE_METRIC_NAMES",
    "REPORT_SCHEMA_VERSION",
    "AlignmentCounts",
    "ArtifactKind",
    "ConfidenceInterval",
    "E2EResult",
    "EntityPatterns",
    "EntityResult",
    "EvalSetName",
    "EvalgateSettings",
    "EvaluationError",
    "Prediction",
    "SlotResult",
    "TurnMetrics",
    "aggregate_metrics",
    "bootstrap_deltas",
    "build_report",
    "entity_error",
    "e2e_booking_accuracy",
    "e2e_turn_result",
    "evaluate_turn",
    "is_schema_valid",
    "levenshtein_counts",
    "paired_bootstrap_ci",
    "percentile_interval",
    "schema_valid_rate",
    "slot_f1",
    "tokenize",
    "word_error_rate",
    "write_report",
]
