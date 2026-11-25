"""Raw-audio retention policy configuration (P1-09.3).

This module provides an engineering control only. It does **not** decide a
legally acceptable retention period: `docs/compliance.md` §5 records the raw
audio retention value as ``OPEN — counsel review required``. The numeric default
below is a placeholder for local development and must be set by the human data
protection policy owner before any production use.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Protocol

from pydantic import Field

from voiceloop_common import BaseSettings, settings_config, utcnow

# ENGINEERING DEFAULT ONLY — NOT A LEGAL OR COMPLIANCE DECISION.
# The policy owner (<TBD — human data protection lead>, docs/compliance.md §5)
# must set this value. Status remains OPEN — counsel review required.
ENGINEERING_DEFAULT_MAX_RAW_AUDIO_AGE_DAYS = 30


class RetentionSettings(BaseSettings):
    """Typed settings for raw-audio retention; prefix ``VOICELOOP_RETENTION_``."""

    model_config = settings_config("VOICELOOP_RETENTION_")

    max_raw_audio_age_days: int = Field(
        default=ENGINEERING_DEFAULT_MAX_RAW_AUDIO_AGE_DAYS,
        gt=0,
        description="Maximum age of raw audio before deletion eligibility.",
    )


@dataclass(frozen=True, slots=True)
class RawAudioRecord:
    """Metadata-only reference to one raw audio object; never holds audio bytes.

    Attributes:
        uri: Object-store URI (or other resolvable reference) for the audio.
        captured_at: Capture timestamp; naive values are interpreted as UTC.

    """

    uri: str
    captured_at: datetime


class RetentionPolicy(Protocol):
    """Policy interface for deciding whether raw audio may be deleted."""

    def is_deletion_eligible(self, record: RawAudioRecord, now: datetime | None = None) -> bool:
        """Return whether ``record`` is eligible for deletion now."""
        ...


class FixedAgeRetentionPolicy:
    """Deletion policy driven by a configurable maximum raw-audio age."""

    def __init__(self, settings: RetentionSettings | None = None) -> None:
        """Store typed settings, using the marked engineering default if absent."""
        self._settings = settings if settings is not None else RetentionSettings()

    @property
    def max_raw_audio_age_days(self) -> int:
        """Return the configured maximum raw-audio age in days."""
        return self._settings.max_raw_audio_age_days

    def is_deletion_eligible(self, record: RawAudioRecord, now: datetime | None = None) -> bool:
        """Return whether captured audio has reached the configured maximum age.

        Args:
            record: Metadata-only raw audio reference.
            now: Optional explicit UTC time; when omitted, ``utcnow()`` is used
                so deterministic tests can inject a clock.

        Returns:
            ``True`` when the age is at least the configured maximum. Future
            capture timestamps are never deletion-eligible.

        """
        current = _as_utc(now if now is not None else utcnow())
        captured = _as_utc(record.captured_at)
        return current - captured >= timedelta(days=self.max_raw_audio_age_days)


def is_deletion_eligible(
    record: RawAudioRecord,
    settings: RetentionSettings | None = None,
    now: datetime | None = None,
) -> bool:
    """Check deletion eligibility under the configured fixed-age policy.

    Args:
        record: Metadata-only raw audio reference.
        settings: Optional typed settings; defaults to ``VOICELOOP_RETENTION_*``.
        now: Optional explicit time; otherwise the injectable ``utcnow()``.

    Returns:
        ``True`` when the audio may be deleted under the configured control.

    """
    return FixedAgeRetentionPolicy(settings).is_deletion_eligible(record, now)


def _as_utc(value: datetime) -> datetime:
    """Normalise a datetime to UTC, treating naive values as UTC."""
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)
