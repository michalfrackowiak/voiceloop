"""Offline teacher-model triage signal for P4-04 (fixture stand-in only).

All integrations are injectable protocols with deterministic in-memory fakes.
No real model, GPU, network, audio bytes, P2-01 threshold change or alias state
is involved. Every report is labelled SYNTHETIC/FIXTURE STAND-IN and precision
numbers are measured only against simulated reviewed fixture labels.
"""

from __future__ import annotations

from .batch import (
    BatchResult,
    DeterministicFixtureTeacher,
    FixtureTurnSource,
    InMemoryTurnSource,
    TeacherBatchRecord,
    TeacherTranscriber,
    TeacherTranscript,
    TurnRecord,
    TurnSource,
    canonical_hash,
    fixture_inputs_digest,
    run_batch,
)
from .disagreement import (
    Disagreement,
    EventProductionTranscriptSource,
    InMemoryProductionTranscriptSource,
    ProductionTranscript,
    ProductionTranscriptSource,
    score_disagreement,
    tokenize,
)
from .report import (
    DepthComparison,
    FixtureReviewedLabelSource,
    InMemoryReportSink,
    InMemoryReviewedLabelSource,
    JsonFileReportSink,
    PrecisionComparison,
    PrecisionResult,
    ReportSink,
    ReviewedLabel,
    ReviewedLabelSource,
    build_report,
    evaluate_precision,
    validate_report,
    verify_read_only_inputs,
)
from .scorer import (
    BaselineQueueSelector,
    BaselineScore,
    BaselineScorer,
    InMemoryBaselineScorer,
    P2_01BaselineScorer,
    QueueCandidate,
    QueueSelector,
    TeacherOnlySelector,
    TeacherQueueSelector,
    TeacherSignal,
    build_candidates,
)
from .settings import (
    EXPECTED_GOLDEN_CONTENT_HASH,
    RUN_LABEL,
    SIGNAL_NAME,
    TeacherSettings,
)

__all__ = [
    "EXPECTED_GOLDEN_CONTENT_HASH",
    "RUN_LABEL",
    "SIGNAL_NAME",
    "BaselineQueueSelector",
    "BaselineScore",
    "BaselineScorer",
    "BatchResult",
    "DepthComparison",
    "DeterministicFixtureTeacher",
    "Disagreement",
    "EventProductionTranscriptSource",
    "FixtureReviewedLabelSource",
    "FixtureTurnSource",
    "InMemoryBaselineScorer",
    "InMemoryProductionTranscriptSource",
    "InMemoryReportSink",
    "InMemoryReviewedLabelSource",
    "InMemoryTurnSource",
    "JsonFileReportSink",
    "P2_01BaselineScorer",
    "PrecisionComparison",
    "PrecisionResult",
    "ProductionTranscript",
    "ProductionTranscriptSource",
    "QueueCandidate",
    "QueueSelector",
    "ReportSink",
    "ReviewedLabel",
    "ReviewedLabelSource",
    "TeacherBatchRecord",
    "TeacherOnlySelector",
    "TeacherQueueSelector",
    "TeacherSettings",
    "TeacherSignal",
    "TeacherTranscript",
    "TeacherTranscriber",
    "TurnRecord",
    "TurnSource",
    "build_candidates",
    "build_report",
    "canonical_hash",
    "evaluate_precision",
    "fixture_inputs_digest",
    "run_batch",
    "score_disagreement",
    "tokenize",
    "validate_report",
    "verify_read_only_inputs",
]
