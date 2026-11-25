"""P3-06: settings read only the VOICELOOP_RELEASE_SYNC_ prefix."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from services.release.sync import ReleaseSyncSettings


def test_defaults() -> None:
    settings = ReleaseSyncSettings()

    assert settings.alias == "@champion"
    assert settings.task_id == "P3-06"
    assert settings.poll_interval_seconds == pytest.approx(30.0)
    assert settings.temp_root is None
    assert settings.log_level == "INFO"


def test_env_prefix(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("VOICELOOP_RELEASE_SYNC_ALIAS", "@champion-canary")
    monkeypatch.setenv("VOICELOOP_RELEASE_SYNC_POLL_INTERVAL_SECONDS", "5")
    monkeypatch.setenv("VOICELOOP_RELEASE_SYNC_TEMP_ROOT", str(tmp_path))
    monkeypatch.setenv("VOICELOOP_RELEASE_SYNC_TASK_ID", "P3-06-custom")
    monkeypatch.setenv("VOICELOOP_RELEASE_SYNC_LOG_LEVEL", "DEBUG")

    settings = ReleaseSyncSettings()

    assert settings.alias == "@champion-canary"
    assert settings.poll_interval_seconds == pytest.approx(5.0)
    assert settings.temp_root == tmp_path
    assert settings.task_id == "P3-06-custom"
    assert settings.log_level == "DEBUG"


def test_unrelated_prefix_is_ignored(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VOICELOOP_RELEASE_ALIAS", "@challenger")

    assert ReleaseSyncSettings().alias == "@champion"


def test_invalid_values_are_rejected() -> None:
    with pytest.raises(ValidationError):
        ReleaseSyncSettings(poll_interval_seconds=0)
    with pytest.raises(ValidationError):
        ReleaseSyncSettings(alias="   ")
    with pytest.raises(ValidationError):
        ReleaseSyncSettings(task_id="")
