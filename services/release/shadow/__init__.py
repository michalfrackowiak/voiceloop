"""P3-07 shadow mode: bounded call-path mirroring and outcome-based comparison.

This package is a deterministic fake-only rehearsal. Real ingest queues,
converted-artifact runners, booking outcome stores and structured logging plug
in through the public protocols; none of those heavy dependencies is imported.
No artifact produced here promotes a model or writes an MLflow alias.
"""

from .mirror import (
    AuditEvent,
    AuditSink,
    BoundedMirrorSink,
    Clock,
    CollectingAuditSink,
    FakeClock,
    FakeTurnSource,
    MirroredTurn,
    MirrorReason,
    MirrorReceipt,
    MirrorSink,
    MirrorStats,
    ShadowMirror,
    ShadowTurn,
    StructlogAuditSink,
    TurnSource,
)
from .report import (
    CollectingReportSink,
    FileReportSink,
    ReportSink,
    build_shadow_report,
    validate_shadow_report,
    write_shadow_report,
)
from .runner import (
    CandidateRunner,
    ChampionRunner,
    ModelResult,
    ModelSource,
    ResultStatus,
    ScriptedModelRunner,
    ShadowRunner,
    field_digests,
)
from .scoring import (
    ComparisonData,
    FakeOutcomeSource,
    OutcomeRecord,
    OutcomeSource,
    ScoredCall,
    score_comparison,
)
from .settings import ShadowSettings

__all__ = [
    "AuditEvent",
    "AuditSink",
    "BoundedMirrorSink",
    "CandidateRunner",
    "ChampionRunner",
    "Clock",
    "CollectingAuditSink",
    "CollectingReportSink",
    "ComparisonData",
    "FakeClock",
    "FakeOutcomeSource",
    "FakeTurnSource",
    "FileReportSink",
    "MirrorReceipt",
    "MirrorReason",
    "MirrorSink",
    "MirrorStats",
    "MirroredTurn",
    "ModelResult",
    "ModelSource",
    "OutcomeRecord",
    "OutcomeSource",
    "ReportSink",
    "ResultStatus",
    "ScoredCall",
    "ScriptedModelRunner",
    "ShadowMirror",
    "ShadowRunner",
    "ShadowSettings",
    "ShadowTurn",
    "StructlogAuditSink",
    "TurnSource",
    "build_shadow_report",
    "field_digests",
    "score_comparison",
    "validate_shadow_report",
    "write_shadow_report",
]
