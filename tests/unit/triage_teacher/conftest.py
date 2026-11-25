"""Shared fixtures for P4-04 teacher tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from services.triage.teacher.settings import TeacherSettings


@pytest.fixture
def repo_root() -> Path:
    """Return the repository root."""
    return Path(__file__).resolve().parents[3]


@pytest.fixture
def settings(tmp_path: Path) -> TeacherSettings:
    """Return deterministic settings with a temporary output path."""
    return TeacherSettings(
        fixtures_path="tests/fixtures/turns.jsonl",
        output_path=str(tmp_path / "teacher_report.json"),
        queue_size=10,
    )
