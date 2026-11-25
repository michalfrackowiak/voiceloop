"""P1-09.3: typed retention settings and deletion eligibility."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from services.ingest.consent import (
    ENGINEERING_DEFAULT_MAX_RAW_AUDIO_AGE_DAYS,
    FixedAgeRetentionPolicy,
    RawAudioRecord,
    RetentionPolicy,
    RetentionSettings,
    is_deletion_eligible,
)
from voiceloop_common import frozen_clock, use_clock

NOW = datetime(2026, 11, 25, 12, 0, tzinfo=UTC)
RECORD = RawAudioRecord(
    uri="s3://voiceloop-artifacts/calls/call-0001/turn-000.wav",
    captured_at=NOW - timedelta(days=7),
)


def test_settings_use_the_retention_prefix(monkeypatch: pytest.MonkeyPatch) -> None:
    """The typed setting is configurable via VOICELOOP_RETENTION_*."""
    monkeypatch.setenv("VOICELOOP_RETENTION_MAX_RAW_AUDIO_AGE_DAYS", "3")
    assert RetentionSettings().max_raw_audio_age_days == 3

    monkeypatch.delenv("VOICELOOP_RETENTION_MAX_RAW_AUDIO_AGE_DAYS")
    default = RetentionSettings().max_raw_audio_age_days
    assert default == ENGINEERING_DEFAULT_MAX_RAW_AUDIO_AGE_DAYS
    assert default > 0


def test_deletion_eligibility_boundary_and_future_timestamps() -> None:
    """Deletion is allowed at the configured age, never for future audio."""
    settings = RetentionSettings(max_raw_audio_age_days=7)
    assert is_deletion_eligible(RECORD, settings, now=NOW) is True

    younger = RawAudioRecord(RECORD.uri, NOW - timedelta(days=6, hours=23))
    assert is_deletion_eligible(younger, settings, now=NOW) is False

    future = RawAudioRecord(RECORD.uri, NOW + timedelta(days=1))
    assert is_deletion_eligible(future, settings, now=NOW) is False


def test_utcnow_is_used_when_now_is_omitted() -> None:
    """The injectable shared clock controls the eligibility decision."""
    settings = RetentionSettings(max_raw_audio_age_days=7)
    with use_clock(frozen_clock(NOW)):
        assert is_deletion_eligible(RECORD, settings) is True
        recent = RawAudioRecord(RECORD.uri, NOW - timedelta(days=1))
        assert is_deletion_eligible(recent, settings) is False

    naive = RawAudioRecord(RECORD.uri, (NOW - timedelta(days=7)).replace(tzinfo=None))
    assert is_deletion_eligible(naive, settings, now=NOW) is True


def test_policy_interface_accepts_a_fake() -> None:
    """Consumers can depend on the small RetentionPolicy interface."""

    class AlwaysEligible:
        """Test double for a human-approved deletion policy."""

        def is_deletion_eligible(self, record: RawAudioRecord, now: datetime | None = None) -> bool:
            """Return True for any metadata-only record."""
            assert record.uri
            return True

    policy: RetentionPolicy = AlwaysEligible()
    assert policy.is_deletion_eligible(RECORD, now=NOW) is True

    configured = FixedAgeRetentionPolicy(RetentionSettings(max_raw_audio_age_days=7))
    assert configured.max_raw_audio_age_days == 7
    assert configured.is_deletion_eligible(RECORD, now=NOW) is True


def test_raw_audio_record_carries_no_bytes() -> None:
    """Retention records are metadata only; audio never enters Postgres."""
    assert {field.name for field in RECORD.__dataclass_fields__.values()} == {
        "uri",
        "captured_at",
    }
