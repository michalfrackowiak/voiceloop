"""Parakeet/NeMo trainer (P3-02) — synthetic fixture stand-in only.

Nothing in this package imports NeMo, torch or any GPU/network dependency.
The fake fine-tune consumes P2-05's read-only NeMo manifest export, exports a
labelled hash-addressed stand-in artifact, and emits schema-shaped per-word
confidences that are scored by P2-01 through an injectable Protocol.
"""

from __future__ import annotations

from .confidences import (
    ConfidenceEmission,
    ConfidenceEmitter,
    InMemorySignalSink,
    P2TriageSignalSink,
    ScoreRecord,
    TriageSignalSink,
    WordConfidence,
    WordConfidenceOutput,
    build_turn_event,
    fake_word_confidences,
    score_event_with_triage,
)
from .manifest import (
    InMemoryNemoManifestSource,
    JsonlNemoManifestSource,
    NemoManifestRow,
    NemoManifestSource,
    canonical_json,
    sha256_text,
)
from .settings import (
    DEFAULT_CHECKPOINT_URI,
    DEFAULT_CONFIDENCE_MODEL_ID,
    PROVENANCE_LABEL,
    NemoBackend,
    NemoSettings,
)
from .trainer import (
    ArtifactStore,
    CheckpointSource,
    FakeArtifactStore,
    FakeCheckpoint,
    FakeCheckpointSource,
    FakeDelta,
    FakeNeMoArtifact,
    FakeNeMoFinetuneBackend,
    NeMoFinetuneBackend,
    ParakeetNeMoTrainer,
)

__all__ = [
    "DEFAULT_CHECKPOINT_URI",
    "DEFAULT_CONFIDENCE_MODEL_ID",
    "PROVENANCE_LABEL",
    "ArtifactStore",
    "CheckpointSource",
    "ConfidenceEmission",
    "ConfidenceEmitter",
    "FakeArtifactStore",
    "FakeCheckpoint",
    "FakeCheckpointSource",
    "FakeDelta",
    "FakeNeMoArtifact",
    "FakeNeMoFinetuneBackend",
    "InMemoryNemoManifestSource",
    "InMemorySignalSink",
    "JsonlNemoManifestSource",
    "NeMoFinetuneBackend",
    "NemoBackend",
    "NemoManifestRow",
    "NemoManifestSource",
    "NemoSettings",
    "P2TriageSignalSink",
    "ParakeetNeMoTrainer",
    "ScoreRecord",
    "TriageSignalSink",
    "WordConfidence",
    "WordConfidenceOutput",
    "build_turn_event",
    "canonical_json",
    "fake_word_confidences",
    "score_event_with_triage",
    "sha256_text",
]
