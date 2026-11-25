"""Tests for typed dashboard settings."""

from __future__ import annotations

import pytest
from pydantic import ValidationError as PydanticValidationError

from infra.dashboards.settings import DashboardSettings


def test_defaults_are_local_and_synthetic() -> None:
    settings = DashboardSettings()
    assert settings.fixtures_path == "tests/fixtures/turns.jsonl"
    assert settings.snapshot_path == "infra/dashboards/examples/synthetic-snapshot.json"
    assert settings.title == "VoiceLoop local dashboard"
    assert settings.format == "html"
    assert settings.stale_after_minutes == 1440


def test_environment_prefix_is_scoped(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VOICELOOP_DASHBOARDS_FORMAT", "markdown")
    monkeypatch.setenv("VOICELOOP_DASHBOARDS_STALE_AFTER_MINUTES", "60")
    settings = DashboardSettings()
    assert settings.format == "markdown"
    assert settings.stale_after_minutes == 60


def test_empty_labels_and_non_positive_freshness_are_rejected() -> None:
    with pytest.raises(PydanticValidationError):
        DashboardSettings(title="   ")
    with pytest.raises(PydanticValidationError):
        DashboardSettings(stale_after_minutes=0)
