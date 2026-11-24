"""Typed settings base shared by every VoiceLoop component."""

from __future__ import annotations

from pydantic_settings import BaseSettings as PydanticBaseSettings
from pydantic_settings import SettingsConfigDict


def settings_config(env_prefix: str) -> SettingsConfigDict:
    """Build the standard settings config with a component-specific env prefix.

    Args:
        env_prefix: Prefix applied to every environment variable read by a
            settings subclass, e.g. ``VOICELOOP_INGEST_``.

    Returns:
        A pydantic-settings configuration dict.

    """
    return SettingsConfigDict(
        env_prefix=env_prefix,
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )


class BaseSettings(PydanticBaseSettings):
    """Environment-backed settings base class for all components.

    Subclasses should set their own ``env_prefix`` in ``model_config`` (see
    :func:`settings_config`) so that each component reads only its own
    environment variables. No secrets belong in source code or defaults.
    """

    model_config = settings_config("VOICELOOP_")
