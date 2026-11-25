"""Typed, environment-backed settings for the P2-03 review adapter."""

from __future__ import annotations

from pydantic import Field

from voiceloop_common import BaseSettings, settings_config


class ReviewAdapterSettings(BaseSettings):
    """Configuration for P2-03; every field reads ``VOICELOOP_REVIEW_*``.

    ``tool`` selects among the fake implementations defined in ``tool.py``.
    A real Label Studio or Argilla backend is selected in that same module, so
    swapping tools never changes callers, task construction or persistence.
    No real service, audio bytes or credentials are configured here.
    """

    model_config = settings_config("VOICELOOP_REVIEW_")

    tool: str = Field(default="fake_in_memory", min_length=1)
    booking_schema_path: str | None = Field(default=None)
    default_reviewer_id: str = Field(default="fixture-reviewer", min_length=1)
