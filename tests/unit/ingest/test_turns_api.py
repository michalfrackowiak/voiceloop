"""End-to-end tests for POST /v1/turns with synthetic fixture events."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

from fastapi.testclient import TestClient

from services.ingest.repository import InMemoryRepository
from services.ingest.service import IngestService

REPO_ROOT = Path(__file__).resolve().parents[3]


def _relative(path: Path) -> str:
    """Return a repo-relative POSIX path for a fixture file."""
    return path.relative_to(REPO_ROOT).as_posix()


def _assert_structured_error(response: Any, code: str = "validation_error") -> None:
    """Assert the shared error body shape and machine-readable code."""
    body = response.json()
    assert set(body) == {"code", "message", "details"}
    assert body["code"] == code
    assert isinstance(body["message"], str) and body["message"]
    assert isinstance(body["details"], dict)


def test_all_fixture_events_ingest_end_to_end(
    client: TestClient,
    service: IngestService,
    repository: InMemoryRepository,
    event_files: list[Path],
    load_json: Callable[[str], Any],
) -> None:
    assert len(event_files) >= 30

    for path in event_files:
        event = load_json(_relative(path))
        response = client.post("/v1/turns", json=event)
        assert response.status_code == 202, response.text
        receipt = response.json()
        assert receipt["event_id"] == event["event_id"]
        assert receipt["call_id"] == event["call_id"]
        assert receipt["status"] == "queued"

    assert service.stats()["queued"] == len(event_files)
    assert service.process_pending() == len(event_files)

    turns = repository.list_turns()
    assert len(turns) == len(event_files)
    assert {turn["call_id"] for turn in turns} == {
        "call-0001",
        "call-0002",
        "call-0003",
        "call-0004",
        "call-0005",
        "call-0006",
    }
    assert any(turn["consent"] == {"recording": True, "training": False} for turn in turns)
    assert all("bytes" not in turn["audio"] for turn in turns)
    assert len({turn["event_id"] for turn in turns}) == len(event_files)
    call_ids = [f"call-{index:04d}" for index in range(1, 7)]
    assert sum(1 for call_id in call_ids if repository.get_outcome(call_id)) == 6
    assert len(repository.list_model_versions()) >= 2

    first = turns[0]
    stored = client.get(f"/v1/turns/{first['event_id']}")
    assert stored.status_code == 200
    assert stored.json()["turn"] == first
    assert stored.json()["receipt"]["status"] == "persisted"

    call = client.get(f"/v1/calls/{first['call_id']}")
    assert call.status_code == 200
    assert call.json()["call"]["turn_count"] == 6


def test_duplicate_event_id_returns_original_receipt_without_second_turn(
    client: TestClient,
    service: IngestService,
    repository: InMemoryRepository,
    load_json: Callable[[str], Any],
) -> None:
    event = load_json("tests/fixtures/events/call-0002/turn-003.json")

    first = client.post("/v1/turns", json=event)
    service.process_pending()
    second = client.post("/v1/turns", json=event)
    service.process_pending()

    assert first.status_code == second.status_code == 202
    assert json.dumps(first.json(), sort_keys=True) == json.dumps(second.json(), sort_keys=True)
    assert second.headers["X-VoiceLoop-Idempotent"] == "hit"
    assert len(repository.list_turns()) == 1
    assert service.stats()["duplicates"] == 1


def test_invalid_events_return_structured_errors(
    client: TestClient,
    repository: InMemoryRepository,
    load_json: Callable[[str], Any],
) -> None:
    valid = load_json("schemas/examples/turn_event.valid.full.json")

    missing_stt = dict(valid)
    missing_stt.pop("stt")
    bad_uuid = {**valid, "event_id": "not-a-uuid"}
    bad_status = {**valid, "outcome": {**valid["outcome"], "status": "booked"}}
    extra_field = {**valid, "unexpected": True}

    for payload in (missing_stt, bad_uuid, bad_status, extra_field):
        response = client.post("/v1/turns", json=payload)
        assert response.status_code == 422, response.text
        _assert_structured_error(response)
        assert response.json()["details"]["schema"] == "turn_event"

    response = client.post("/v1/turns", json=[valid])
    assert response.status_code == 422
    _assert_structured_error(response)
    assert repository.list_turns() == []


def test_not_found_errors_use_structured_shape(client: TestClient) -> None:
    missing_turn = client.get("/v1/turns/does-not-exist")
    assert missing_turn.status_code == 404
    _assert_structured_error(missing_turn, "not_found")

    missing_call = client.get("/v1/calls/does-not-exist")
    assert missing_call.status_code == 404
    _assert_structured_error(missing_call, "not_found")

    unknown_route = client.get("/v9/nope")
    assert unknown_route.status_code == 404
    _assert_structured_error(unknown_route, "not_found")
