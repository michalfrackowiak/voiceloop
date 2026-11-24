"""Typed settings for the VoiceLoop Python SDK."""

from __future__ import annotations

from pydantic import Field

from voiceloop_common.settings import BaseSettings, settings_config


class SdkSettings(BaseSettings):
    """SDK settings; every environment variable uses the ``VOICELOOP_SDK_`` prefix."""

    model_config = settings_config("VOICELOOP_SDK_")

    endpoint: str = "http://localhost:8000/v1/turns"
    timeout_seconds: float = Field(default=1.0, gt=0.0)
    queue_capacity: int = Field(default=1024, ge=1)
    max_retries: int = Field(default=3, ge=0)
    retry_backoff_seconds: float = Field(default=0.25, ge=0.0)
    circuit_breaker_failure_threshold: int = Field(default=5, ge=1)
    circuit_breaker_recovery_seconds: float = Field(default=30.0, gt=0.0)
