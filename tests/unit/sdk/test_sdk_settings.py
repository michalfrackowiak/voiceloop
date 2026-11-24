"""Tests for the SDK's typed, prefixed settings."""

import pytest

from sdk import SdkSettings


def test_defaults_are_safe_and_fail_open(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in (
        "VOICELOOP_SDK_ENDPOINT",
        "VOICELOOP_SDK_QUEUE_CAPACITY",
        "VOICELOOP_SDK_MAX_RETRIES",
    ):
        monkeypatch.delenv(name, raising=False)

    settings = SdkSettings()

    assert settings.endpoint == "http://localhost:8000/v1/turns"
    assert settings.timeout_seconds > 0
    assert settings.queue_capacity >= 1
    assert settings.max_retries >= 0
    assert settings.retry_backoff_seconds >= 0
    assert settings.circuit_breaker_failure_threshold >= 1
    assert settings.circuit_breaker_recovery_seconds > 0


def test_environment_uses_voiceLoop_sdk_prefix(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VOICELOOP_SDK_QUEUE_CAPACITY", "7")
    monkeypatch.setenv("VOICELOOP_SDK_MAX_RETRIES", "1")
    monkeypatch.setenv("VOICELOOP_SDK_ENDPOINT", "http://ingest.test/v1/turns")
    monkeypatch.setenv("VOICELOOP_INGEST_QUEUE_CAPACITY", "999")

    settings = SdkSettings()

    assert settings.queue_capacity == 7
    assert settings.max_retries == 1
    assert settings.endpoint == "http://ingest.test/v1/turns"
