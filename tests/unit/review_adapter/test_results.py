"""Stored result shape, API projection, round trip and Postgres-style idempotency."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
import yaml

from services.review_adapter import (
    ROUTING_LABELS,
    InMemoryReviewStore,
    ReviewAdapterError,
    StoredReviewResult,
)
from tests.unit.review_adapter.conftest import CONTRACT


def result(event_id: str = "event-1", item_id: str = "item-event-1") -> StoredReviewResult:
    """Build a fully populated stored fake result."""
    return StoredReviewResult(
        schema_version="1.0.0",
        item_id=item_id,
        event_id=event_id,
        call_id="call-1",
        turn_index=2,
        audio_uri="s3://bucket/turn-002.wav",
        hypothesis="Mr Smythe",
        corrected_transcript="Mr. Smith",
        slots={"guest_name": "Mr. Smith"},
        routing="stt_wrong",
        reviewer_id="reviewer-1",
        started_at=datetime(2026, 11, 25, 10, tzinfo=UTC),
        completed_at=datetime(2026, 11, 25, 10, 1, 12, tzinfo=UTC),
        warnings=("fake_reviewer_stand_in",),
        blind=False,
        tool_name="fake_in_memory",
        training_consent=True,
    )


def test_api_result_matches_openapi_required_fields() -> None:
    """Warnings and original IDs stay out of the contract payload."""
    document = yaml.safe_load(Path(CONTRACT).read_text(encoding="utf-8"))
    assert isinstance(document, dict)
    required = set(document["components"]["schemas"]["ReviewResult"]["required"])
    api = result().to_api_result()
    assert set(api) == required
    assert "warnings" not in api
    assert "event_id" not in api
    assert api["started_at"] == "2026-11-25T10:00:00Z"
    assert api["completed_at"] == "2026-11-25T10:01:12Z"
    assert api["routing"] in ROUTING_LABELS


def test_full_record_keeps_lineage_warnings_and_tool_name() -> None:
    """The internal/Postgres record carries more than the public API."""
    record = result().to_dict()
    assert record["event_id"] == "event-1"
    assert record["call_id"] == "call-1"
    assert record["turn_index"] == 2
    assert record["warnings"] == ["fake_reviewer_stand_in"]
    assert record["reviewer_id"] == "reviewer-1"
    assert record["tool_name"] == "fake_in_memory"
    assert result().to_dataset_view()["source_event_id"] == "event-1"


def test_dict_round_trip_preserves_all_fields() -> None:
    """Stored records survive a JSON-shaped serialize/deserialize cycle."""
    original = result()
    assert StoredReviewResult.from_dict(original.to_dict()) == original


def test_store_upserts_per_event_without_duplicates() -> None:
    """Duplicate submissions return the first row; IDs cannot be remapped."""
    store = InMemoryReviewStore()
    first = result()
    assert store.save(first) is first
    assert store.save(first) is first
    assert store.save(result(item_id="item-duplicate")) is first
    assert store.get(first.item_id) is first
    assert store.get_by_event("event-1") is first
    assert store.get("item-duplicate") is None
    assert len(store.all()) == 1
    with pytest.raises(ReviewAdapterError, match="already mapped"):
        store.save(result(event_id="event-2"))


def test_invalid_routing_is_rejected_by_the_record() -> None:
    """The four-label restriction is enforced at the storage boundary too."""
    payload: dict[str, Any] = result().to_dict()
    payload["routing"] = "not_a_label"
    with pytest.raises(ReviewAdapterError, match="unknown routing label"):
        StoredReviewResult.from_dict(payload)
