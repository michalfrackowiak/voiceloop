"""Tests for late-arriving outcomes merged by call id."""

from __future__ import annotations

import json

from fastapi.testclient import TestClient

from services.ingest.repository import InMemoryRepository
from services.ingest.service import IngestService


def test_outcome_is_queued_and_merged_by_call_id(
    client: TestClient, service: IngestService, repository: InMemoryRepository
) -> None:
    response = client.patch(
        "/v1/calls/call-late/outcome",
        json={
            "status": "confirmed",
            "final_record": {"guests": 2, "room_type": "standard"},
            "updated_at": "2026-11-25T10:00:00Z",
        },
    )

    assert response.status_code == 202, response.text
    receipt = response.json()
    assert receipt["kind"] == "outcome"
    assert receipt["call_id"] == "call-late"
    assert service.process_pending() == 1

    outcome = repository.get_outcome("call-late")
    assert outcome is not None
    assert outcome["status"] == "confirmed"
    assert outcome["final_record"] == {"guests": 2, "room_type": "standard"}
    call = repository.get_call("call-late")
    assert call is not None


def test_later_outcome_merges_fields_and_duplicate_is_idempotent(
    client: TestClient, service: IngestService, repository: InMemoryRepository
) -> None:
    payload = {
        "event_id": "11111111-1111-4111-8111-111111111111",
        "status": "confirmed",
        "final_record": {"guests": 2},
        "updated_at": "2026-11-25T10:00:00Z",
    }
    first = client.patch("/v1/calls/call-merge/outcome", json=payload)
    service.process_pending()
    duplicate = client.patch("/v1/calls/call-merge/outcome", json=payload)
    service.process_pending()

    assert json.dumps(first.json(), sort_keys=True) == json.dumps(duplicate.json(), sort_keys=True)
    assert duplicate.headers["X-VoiceLoop-Idempotent"] == "hit"
    first_outcome = repository.get_outcome("call-merge")
    assert first_outcome is not None
    assert first_outcome["merge_count"] == 1

    later = client.patch(
        "/v1/calls/call-merge/outcome",
        json={
            "status": "edited_by_staff",
            "final_record": {"room_type": "suite"},
            "updated_at": "2026-11-25T11:00:00Z",
        },
    )
    assert later.status_code == 202
    service.process_pending()

    outcome = repository.get_outcome("call-merge")
    assert outcome is not None
    assert outcome["status"] == "edited_by_staff"
    assert outcome["final_record"] == {"guests": 2, "room_type": "suite"}
    assert outcome["merge_count"] == 2


def test_invalid_outcome_status_returns_structured_error(client: TestClient) -> None:
    response = client.patch(
        "/v1/calls/call-bad/outcome",
        json={"status": "booked", "updated_at": "2026-11-25T10:00:00Z"},
    )

    assert response.status_code == 422
    body = response.json()
    assert body["code"] == "validation_error"
    assert body["details"]["schema"] == "turn_event"
    assert body["details"]["field"] in {"outcome", "outcome.status"}


def test_unknown_outcome_field_is_rejected(client: TestClient) -> None:
    response = client.patch(
        "/v1/calls/call-bad/outcome",
        json={"status": "confirmed", "surprise": 1},
    )

    assert response.status_code == 422
    assert response.json()["code"] == "validation_error"
