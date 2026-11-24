"""Typed settings for outcome connectors (``VOICELOOP_CONNECTOR_*``)."""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path

from pydantic import Field

from voiceloop_common.settings import BaseSettings, settings_config


class ConnectorMode(StrEnum):
    """Source implementation selected by settings."""

    WEBHOOK = "webhook"
    POLLER = "poller"


class ConnectorSettings(BaseSettings):
    """Connector settings; every environment variable uses the configured prefix.

    The booking adapter itself defaults to the in-memory mock; a real vendor API
    can be injected through :func:`sdk.connectors.create_connector` without
    changing these settings.
    """

    model_config = settings_config("VOICELOOP_CONNECTOR_")

    source_mode: ConnectorMode = ConnectorMode.WEBHOOK
    ingest_endpoint: str = "http://localhost:8000"
    timeout_seconds: float = Field(default=1.0, gt=0.0)
    max_events_per_run: int = Field(default=100, ge=1)
    webhook_queue_size: int = Field(default=256, ge=1)
    fixture_dir: Path = Path("tests/fixtures")
