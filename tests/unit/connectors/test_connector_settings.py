"""Typed settings tests for the connector package."""

from __future__ import annotations

import pytest

from sdk.connectors import ConnectorMode, ConnectorSettings


def test_settings_use_connector_env_prefix(monkeypatch: pytest.MonkeyPatch) -> None:
    """Connector settings read only VOICELOOP_CONNECTOR_* variables."""
    monkeypatch.setenv("VOICELOOP_CONNECTOR_SOURCE_MODE", "poller")
    monkeypatch.setenv("VOICELOOP_CONNECTOR_INGEST_ENDPOINT", "http://ingest.test")
    monkeypatch.setenv("VOICELOOP_CONNECTOR_MAX_EVENTS_PER_RUN", "7")

    settings = ConnectorSettings()

    assert settings.source_mode is ConnectorMode.POLLER
    assert settings.ingest_endpoint == "http://ingest.test"
    assert settings.max_events_per_run == 7
    assert settings.timeout_seconds == 1.0


def test_defaults_select_the_mock_webhook_connector(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Roadmap defaults are mock booking plus webhook mode."""
    monkeypatch.delenv("VOICELOOP_CONNECTOR_SOURCE_MODE", raising=False)
    settings = ConnectorSettings()
    assert settings.source_mode is ConnectorMode.WEBHOOK
    assert settings.fixture_dir.parts[-2:] == ("tests", "fixtures")
