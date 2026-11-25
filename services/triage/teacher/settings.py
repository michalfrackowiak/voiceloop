"""Typed, environment-backed settings for the P4-04 teacher triage signal."""

from __future__ import annotations

from pydantic import Field

from voiceloop_common import BaseSettings, settings_config

# Committed synthetic v1 golden stand-in hash (P1-06). Verified read-only only.
EXPECTED_GOLDEN_CONTENT_HASH = "662cdaf43ee2d5fde03faf5e69fde196114d4f312406d94c00bf53714d03da8e"
RUN_LABEL = "SYNTHETIC/FIXTURE STAND-IN - NOT A PRODUCTION PRECISION CLAIM"
SIGNAL_NAME = "teacher_transcript_disagreement"
ARTIFACT_TYPE = "voiceloop.teacher.triage.report"
ARTIFACT_SCHEMA_VERSION = "internal-1.0.0"
HASH_ALGORITHM = "sha256"


class TeacherSettings(BaseSettings):
    """Configuration for the offline teacher rehearsal.

    Every field reads an environment variable prefixed with ``VOICELOOP_TEACHER_``.
    Inputs (fixtures and the sealed golden stand-in) are read-only; ``output_path``
    is the only path this package writes. Defaults contain no real model names,
    thresholds from a tuned model, or secrets.
    """

    model_config = settings_config("VOICELOOP_TEACHER_")

    fixtures_path: str = "tests/fixtures/turns.jsonl"
    golden_artifact_dir: str = "services/datasets/golden/artifacts"
    golden_version: str = "golden.v1"
    expected_golden_content_hash: str = EXPECTED_GOLDEN_CONTENT_HASH
    output_path: str = "reports/triage/teacher/report.json"

    teacher_model_id: str = "fake://teacher/fixture-standin-v1"
    queue_size: int = Field(default=10, ge=1)
    signal_weight: float = Field(default=1.0, ge=0.0, le=10.0)
    low_confidence_ceiling: float = Field(default=0.60, ge=0.0, le=1.0)
    no_speech_prob_ceiling: float = Field(default=0.50, ge=0.0, le=1.0)
    hash_algorithm: str = HASH_ALGORITHM
