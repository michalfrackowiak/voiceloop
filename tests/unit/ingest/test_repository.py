"""Tests for the in-memory repository fake and its merge rules."""

from __future__ import annotations

from typing import Any

from services.ingest.repository import InMemoryRepository, model_version_id


def _event(event_id: str = "evt-1", call_id: str = "call-1", turn_index: int = 0) -> dict[str, Any]:
    return {
        "event_id": event_id,
        "call_id": call_id,
        "turn_index": turn_index,
        "audio": {"uri": "s3://bucket/turn.wav", "sample_rate_hz": 8000},
        "stt": {"text": "hello", "model_id": "stt", "model_version": "1"},
        "consent": {"recording": True, "training": False},
    }


def test_claim_event_is_atomic_and_returns_original_receipt() -> None:
    repo = InMemoryRepository()
    receipt = {"event_id": "evt-1", "status": "queued"}

    assert repo.claim_event("evt-1", receipt) is None
    assert repo.claim_event("evt-1", {"event_id": "evt-1", "status": "other"}) == receipt

    repo.save_receipt("evt-1", {"event_id": "evt-1", "status": "persisted"})
    assert repo.get_receipt("evt-1") == {"event_id": "evt-1", "status": "persisted"}
    assert repo.get_receipt("missing") is None


def test_save_turn_stores_metadata_only_and_preserves_consent() -> None:
    repo = InMemoryRepository()
    repo.save_call("call-1", {"recording": True, "training": True}, seen_at="2026-01-01T00:00:00Z")
    repo.save_call(
        "call-1", {"recording": False, "training": False}, seen_at="2026-01-02T00:00:00Z"
    )

    turn = repo.save_turn(_event(), audio_state="stored")

    assert repo.get_turn("evt-1") == turn
    assert turn["audio"]["uri"] == "s3://bucket/turn.wav"
    assert "bytes" not in turn and "data" not in turn
    assert turn["consent"] == {"recording": True, "training": False}
    call = repo.get_call("call-1")
    assert call is not None
    assert call["consent_recording"] is True
    assert call["consent_training"] is True
    assert call["turn_count"] == 1
    assert len(repo.list_turns("call-1")) == 1
    assert repo.list_turns("other") == []


def test_merge_outcome_uses_updated_at_and_merges_final_record() -> None:
    repo = InMemoryRepository()
    first = {
        "status": "confirmed",
        "final_record": {"guests": 2},
        "updated_at": "2026-01-02T00:00:00Z",
    }
    second = {"status": "edited_by_staff", "final_record": {"room_type": "suite"}}

    repo.merge_outcome("call-1", first, source_event_id="e1")
    merged = repo.merge_outcome("call-1", second, source_event_id="e2")

    assert merged["status"] == "edited_by_staff"
    assert merged["final_record"] == {"guests": 2, "room_type": "suite"}
    assert merged["merge_count"] == 2

    stale = repo.merge_outcome(
        "call-1",
        {"status": "abandoned", "updated_at": "2026-01-01T00:00:00Z"},
        source_event_id="e3",
    )
    assert stale["status"] == "edited_by_staff"

    unknown = repo.merge_outcome(
        "call-1",
        {"status": "unknown", "updated_at": "2026-01-03T00:00:00Z"},
        source_event_id="e4",
    )
    assert unknown["status"] == "edited_by_staff"
    assert repo.get_outcome("call-1") == unknown


def test_model_versions_are_deduplicated() -> None:
    repo = InMemoryRepository()
    repo.save_model_version("stt", "whisper", "1", seen_at="2026-01-01T00:00:00Z")
    repo.save_model_version("stt", "whisper", "1", seen_at="2026-01-02T00:00:00Z")

    versions = repo.list_model_versions()

    assert len(versions) == 1
    assert versions[0]["model_version_id"] == model_version_id("stt", "whisper", "1")
    assert versions[0]["first_seen_at"] == "2026-01-01T00:00:00Z"
