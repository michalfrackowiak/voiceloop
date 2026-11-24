"""Tests for the shared typed settings base."""

import pytest

from voiceloop_common import BaseSettings, settings_config


def test_settings_config_carries_prefix_and_common_defaults() -> None:
    config = settings_config("VOICELOOP_UNIT_")

    assert config["env_prefix"] == "VOICELOOP_UNIT_"
    assert config["extra"] == "ignore"
    assert config["case_sensitive"] is False


def test_subclass_reads_only_its_prefixed_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    class UnitSettings(BaseSettings):
        model_config = settings_config("VOICELOOP_UNIT_")
        log_level: str = "INFO"

    monkeypatch.setenv("VOICELOOP_UNIT_LOG_LEVEL", "DEBUG")
    monkeypatch.setenv("VOICELOOP_OTHER_LOG_LEVEL", "TRACE")

    settings = UnitSettings()

    assert settings.log_level == "DEBUG"


def test_subclass_falls_back_to_default_without_prefix(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class UnitSettings(BaseSettings):
        model_config = settings_config("VOICELOOP_UNIT_")
        log_level: str = "INFO"

    monkeypatch.delenv("VOICELOOP_UNIT_LOG_LEVEL", raising=False)

    assert UnitSettings().log_level == "INFO"
