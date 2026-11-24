"""Tests for typed ingest settings."""

from __future__ import annotations

import pytest

from services.ingest.settings import IngestSettings


def test_ingest_settings_use_prefixed_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VOICELOOP_INGEST_LOG_LEVEL", "DEBUG")
    monkeypatch.setenv("VOICELOOP_INGEST_QUEUE_CAPACITY", "7")
    monkeypatch.setenv("VOICELOOP_INGEST_AUDIO_BUCKET", "fixture-bucket")
    monkeypatch.setenv("VOICELOOP_INGEST_REPOSITORY_BACKEND", "postgres")
    monkeypatch.setenv("VOICELOOP_OTHER_AUDIO_BUCKET", "ignored")

    settings = IngestSettings()

    assert settings.log_level == "DEBUG"
    assert settings.queue_capacity == 7
    assert settings.audio_bucket == "fixture-bucket"
    assert settings.repository_backend == "postgres"
    assert settings.object_store_backend == "memory"


def test_defaults_are_fake_backed_and_non_blocking() -> None:
    settings = IngestSettings()

    assert settings.repository_backend == "memory"
    assert settings.object_store_backend == "memory"
    assert settings.queue_capacity >= 1
    assert settings.worker_poll_interval_s > 0
    assert settings.max_audio_bytes > 0
    assert settings.presign_ttl_s > 0


def test_presign_secret_is_ephemeral_and_masked() -> None:
    first = IngestSettings()
    second = IngestSettings()

    assert first.presign_secret.get_secret_value() != second.presign_secret.get_secret_value()
    assert "token" not in repr(first)
