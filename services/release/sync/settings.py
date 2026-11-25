"""Typed settings for the P3-06 release sync agent.

Every field is read from an environment variable prefixed with
``VOICELOOP_RELEASE_SYNC_``. There are no credentials, model names or serving
addresses here: real adapters receive them through their own configuration.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import Field, field_validator

from voiceloop_common import BaseSettings, settings_config

DEFAULT_ALIAS = "@champion"
DEFAULT_TASK_ID = "P3-06"


class ReleaseSyncSettings(BaseSettings):
    """Environment-backed configuration for one alias sync agent."""

    model_config = settings_config("VOICELOOP_RELEASE_SYNC_")

    alias: str = DEFAULT_ALIAS
    poll_interval_seconds: float = Field(default=30.0, gt=0)
    temp_root: Path | None = None
    task_id: str = DEFAULT_TASK_ID
    log_level: str = "INFO"

    @field_validator("alias", "task_id", "log_level")
    @classmethod
    def _require_text(cls, value: str) -> str:
        """Reject blank text for identifiers and log levels."""
        if not value.strip():
            raise ValueError("must be a non-blank string")
        return value
