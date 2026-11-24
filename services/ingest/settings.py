"""Typed ingest settings, all read from ``VOICELOOP_INGEST_*`` environment variables."""

from __future__ import annotations

import secrets
from typing import Literal

from pydantic import Field, SecretStr

from voiceloop_common import BaseSettings, settings_config


class IngestSettings(BaseSettings):
    """Environment-backed configuration for the ingest service.

    Real Postgres/S3 drivers are optional: the default backends are in-memory
    fakes with the same interface. Selecting a real backend without a driver
    wired in raises ``ConfigError`` at startup rather than failing mid-request.
    """

    model_config = settings_config("VOICELOOP_INGEST_")

    log_level: str = "INFO"
    repository_backend: Literal["memory", "postgres"] = "memory"
    object_store_backend: Literal["memory", "s3"] = "memory"
    database_url: str | None = None

    queue_capacity: int = Field(default=1024, ge=1)
    worker_poll_interval_s: float = Field(default=0.01, gt=0)
    worker_shutdown_timeout_s: float = Field(default=2.0, gt=0)
    autostart_worker: bool = True

    audio_bucket: str = "voiceloop-local"
    audio_uri_scheme: str = "s3"
    presign_ttl_s: int = Field(default=900, ge=1)
    presign_secret: SecretStr = Field(default_factory=lambda: SecretStr(secrets.token_urlsafe(32)))
    max_audio_bytes: int = Field(default=25 * 1024 * 1024, ge=1)
