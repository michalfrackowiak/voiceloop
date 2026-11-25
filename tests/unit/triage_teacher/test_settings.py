"""Typed settings and env-prefix tests for P4-04."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from services.triage.teacher.settings import (
    EXPECTED_GOLDEN_CONTENT_HASH,
    RUN_LABEL,
    SIGNAL_NAME,
    TeacherSettings,
)


def test_settings_use_teacher_env_prefix(monkeypatch: pytest.MonkeyPatch) -> None:
    """Only VOICELOOP_TEACHER_-prefixed variables configure this package."""
    monkeypatch.setenv("VOICELOOP_TEACHER_QUEUE_SIZE", "4")
    monkeypatch.setenv("VOICELOOP_TEACHER_TEACHER_MODEL_ID", "fake://teacher/test")
    config = TeacherSettings()
    assert config.queue_size == 4
    assert config.teacher_model_id == "fake://teacher/test"
    assert config.fixtures_path == "tests/fixtures/turns.jsonl"


def test_settings_validate_bounds(monkeypatch: pytest.MonkeyPatch) -> None:
    """Invalid numeric settings fail closed with a pydantic validation error."""
    monkeypatch.setenv("VOICELOOP_TEACHER_QUEUE_SIZE", "0")
    with pytest.raises(ValidationError):
        TeacherSettings()


def test_constants_are_explicit_standin_markers() -> None:
    """Constants make the fake-only nature and sealed hash auditable."""
    assert SIGNAL_NAME == "teacher_transcript_disagreement"
    assert "SYNTHETIC/FIXTURE STAND-IN" in RUN_LABEL
    assert len(EXPECTED_GOLDEN_CONTENT_HASH) == 64
