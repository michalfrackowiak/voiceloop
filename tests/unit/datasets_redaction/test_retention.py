"""Fake retention job: raw-only deletion, auditable counts and idempotency."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from services.datasets.redaction.retention import (
    REASON_EXPIRED_RAW,
    REASON_FUTURE_CAPTURE,
    REASON_NOT_RAW,
    REASON_WITHIN_WINDOW,
    AssetKind,
    InMemoryRetentionStore,
    RetainedAsset,
    asset_from_dict,
    run_retention,
)
from services.datasets.redaction.settings import (
    POLICY_STATUS,
    SYNTHETIC_LABEL,
    FixedClock,
    InMemoryAuditSink,
    RedactionSettings,
)
from services.datasets.redaction.spans import RedactionError
from voiceloop_common import StorageError

NOW = datetime(2026, 3, 1, 12, 0, tzinfo=UTC)


def _asset(uri: str, kind: AssetKind, age_days: float) -> RetainedAsset:
    """Build a metadata-only asset record at a relative age."""
    return RetainedAsset(uri, kind, NOW - timedelta(days=age_days), "call-synthetic")


def test_retention_deletes_only_expired_raw_audio(settings: RedactionSettings) -> None:
    store = InMemoryRetentionStore()
    for asset in (
        _asset("raw://expired", AssetKind.RAW_AUDIO, 10),
        _asset("raw://fresh", AssetKind.RAW_AUDIO, 2),
        _asset("redacted://old", AssetKind.REDACTED_AUDIO, 400),
        _asset("train://old", AssetKind.TRAINING_DERIVED, 400),
    ):
        store.add(asset)

    report = run_retention(store, settings=settings, clock=FixedClock(NOW))

    assert report.deleted_count == 1
    assert report.kept_count == 3
    assert report.window_days == 7
    assert report.reason_counts() == {
        REASON_EXPIRED_RAW: 1,
        REASON_NOT_RAW: 2,
        REASON_WITHIN_WINDOW: 1,
    }
    assert "raw://expired" not in store.assets
    assert store.deleted_uris == ["raw://expired"]
    assert "redacted://old" in store.assets
    assert "train://old" in store.assets
    assert "raw://fresh" in store.assets
    assert report.to_dict()["synthetic"] is True
    assert report.to_dict()["label"] == SYNTHETIC_LABEL
    assert report.policy_status == POLICY_STATUS
    assert "no real object store" in report.deletion_claim


def test_retention_is_idempotent(settings: RedactionSettings) -> None:
    store = InMemoryRetentionStore()
    store.add(_asset("raw://expired", AssetKind.RAW_AUDIO, 30))
    clock = FixedClock(NOW)

    first = run_retention(store, settings=settings, clock=clock)
    second = run_retention(store, settings=settings, clock=clock)

    assert first.deleted_count == 1
    assert second.deleted_count == 0
    assert second.kept_count == 0
    assert second.decisions == ()
    assert store.deleted_uris == ["raw://expired"]


def test_retention_boundary_is_inclusive(settings: RedactionSettings) -> None:
    store = InMemoryRetentionStore()
    store.add(_asset("raw://boundary", AssetKind.RAW_AUDIO, 7))
    report = run_retention(store, settings=settings, clock=FixedClock(NOW))
    assert report.deleted_count == 1
    assert report.decisions[0].reason == REASON_EXPIRED_RAW


def test_future_capture_is_kept(settings: RedactionSettings) -> None:
    store = InMemoryRetentionStore()
    store.add(_asset("raw://future", AssetKind.RAW_AUDIO, -1))
    report = run_retention(store, settings=settings, clock=FixedClock(NOW))
    assert report.deleted_count == 0
    assert report.decisions[0].reason == REASON_FUTURE_CAPTURE
    assert "raw://future" in store.assets


def test_retention_audit_events_are_pii_free(settings: RedactionSettings) -> None:
    audit = InMemoryAuditSink()
    store = InMemoryRetentionStore()
    store.add(_asset("raw://expired", AssetKind.RAW_AUDIO, 10))
    run_retention(store, settings=settings, clock=FixedClock(NOW), audit=audit)
    assert [event.event_type for event in audit.events] == [
        "raw_audio_fake_deleted",
        "retention_run",
    ]
    payload = str(audit.to_dicts())
    assert "call-synthetic" not in payload
    assert "raw://expired" in payload
    assert "deletion" in payload.lower()


def test_store_rejects_unknown_and_non_raw_deletion() -> None:
    store = InMemoryRetentionStore()
    with pytest.raises(StorageError):
        store.delete_raw("raw://missing")
    store.add(_asset("redacted://guarded", AssetKind.REDACTED_AUDIO, 400))
    with pytest.raises(StorageError):
        store.delete_raw("redacted://guarded")


def test_asset_from_dict_validates_metadata() -> None:
    asset = asset_from_dict(
        {
            "uri": "raw://valid",
            "kind": "raw_audio",
            "captured_at": "2026-01-01T00:00:00+00:00",
        }
    )
    assert asset.uri == "raw://valid"
    assert asset.kind is AssetKind.RAW_AUDIO
    assert asset.source_call_id == ""

    for bad in (
        {"uri": "", "kind": "raw_audio", "captured_at": "2026-01-01T00:00:00+00:00"},
        {"uri": "raw://x", "kind": "unknown", "captured_at": "2026-01-01T00:00:00+00:00"},
        {"uri": "raw://x", "kind": "raw_audio", "captured_at": "not-a-date"},
    ):
        with pytest.raises(RedactionError) as exc:
            asset_from_dict(bad)
        assert exc.value.code == "invalid_asset_record"


def test_naive_capture_timestamps_are_treated_as_utc(settings: RedactionSettings) -> None:
    store = InMemoryRetentionStore()
    store.add(RetainedAsset("raw://naive", AssetKind.RAW_AUDIO, datetime(2026, 2, 1, 12, 0)))
    report = run_retention(store, settings=settings, clock=FixedClock(NOW))
    assert report.deleted_count == 1
    assert report.decisions[0].age_days == 28.0
