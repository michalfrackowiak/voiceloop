"""Public API for P2-01 failure scoring.

The package consumes schema-valid turn-event dictionaries (``turn_event`` 1.0.0)
and returns a score in [0, 1] with an explanation of every fired signal. No
service, network, GPU or audio decoding is required: all integrations are
protocol seams with deterministic fakes.
"""

from __future__ import annotations

from .explanation import (
    ExplanationStore,
    FiredSignal,
    InMemoryExplanationStore,
    ScoreExplanation,
)
from .mismatch import detect_mismatches
from .repair import RepairContext, build_repair_context, detect_repair, load_prior_turns
from .scoring import (
    ScoringEngine,
    ScoringError,
    load_fixture_scores,
    main,
    score_turn,
)
from .settings import TriageScoringSettings
from .signals import (
    FakeTeacherModel,
    Observation,
    TeacherModel,
    TeacherVerdict,
    extract_signals,
)

__all__ = [
    "ExplanationStore",
    "FakeTeacherModel",
    "FiredSignal",
    "InMemoryExplanationStore",
    "Observation",
    "RepairContext",
    "ScoringEngine",
    "ScoringError",
    "ScoreExplanation",
    "TeacherModel",
    "TeacherVerdict",
    "TriageScoringSettings",
    "build_repair_context",
    "detect_mismatches",
    "detect_repair",
    "extract_signals",
    "load_fixture_scores",
    "load_prior_turns",
    "main",
    "score_turn",
]
