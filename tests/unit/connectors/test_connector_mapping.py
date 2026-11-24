"""Booking-state mapping and shared-schema validation tests."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

import pytest

from sdk.connectors import (
    BookingEvent,
    OutcomeMappingError,
    OutcomeStatus,
    build_outcome_payload,
    map_booking_state,
    outcome_to_booking_state,
    parse_booking_event,
    validate_outcome_payload,
)

RECORD: dict[str, Any] = {
    "check_in": "2026-12-12",
    "check_out": "2026-12-15",
    "guests": 2,
    "room_type": "deluxe",
    "guest_name": "Ada Fixture",
    "contact": {"email": "ada.fixture@example.com", "phone": "+1 555 010 0001"},
}

STAMP = "2026-11-25T12:00:00Z"


@pytest.mark.parametrize(
    ("state", "expected"),
    [
        ("confirmed", OutcomeStatus.CONFIRMED),
        ("BOOKED", OutcomeStatus.CONFIRMED),
        ("staff-edited", OutcomeStatus.EDITED_BY_STAFF),
        ("modified_by_staff", OutcomeStatus.EDITED_BY_STAFF),
        ("abandoned", OutcomeStatus.ABANDONED),
        ("cancelled", OutcomeStatus.ABANDONED),
        ("transferred", OutcomeStatus.TRANSFERRED),
        ("escalated", OutcomeStatus.TRANSFERRED),
        ("something_new", OutcomeStatus.UNKNOWN),
        ("", OutcomeStatus.UNKNOWN),
    ],
)
def test_booking_states_map_to_outcome_enum(state: str, expected: OutcomeStatus) -> None:
    """Vendor aliases map to the schema enum, with unknown as the fallback."""
    assert map_booking_state(state) is expected


def test_outcome_status_round_trips_to_booking_state() -> None:
    """Fixture outcome statuses seed canonical booking states."""
    for status in OutcomeStatus:
        assert map_booking_state(outcome_to_booking_state(status.value)) is status


def test_payload_matches_outcome_object_and_validates() -> None:
    """A mapped payload has only schema keys and passes the shared validator."""
    event = BookingEvent(
        event_id="evt-1",
        call_id="call-0001",
        state="booked",
        final_record=deepcopy(RECORD),
        updated_at=STAMP,
    )
    payload = build_outcome_payload(event)
    assert payload["status"] == "confirmed"
    assert payload["final_record"] == RECORD
    assert payload["updated_at"] == STAMP
    assert set(payload) <= {"status", "final_record", "updated_at"}
    validate_outcome_payload(event.call_id, payload)


def test_unknown_state_does_not_fabricate_a_record() -> None:
    """An unknown booking result carries no invented final record."""
    event = BookingEvent(event_id="evt-2", call_id="call-0002", state="mystery")
    payload = build_outcome_payload(event)
    assert payload == {"status": "unknown"}


@pytest.mark.parametrize(
    "changes",
    [
        {"check_out": "2026-12-10"},
        {"room_type": "penthouse"},
        {"guest_name": ""},
        {"guests": 0},
        {"contact": {"email": "not-an-email"}},
    ],
)
def test_invalid_final_record_is_rejected(changes: dict[str, Any]) -> None:
    """Invalid booking slots are skipped instead of emitted as outcomes."""
    record = deepcopy(RECORD)
    record.update(changes)
    event = BookingEvent(
        event_id="evt-3", call_id="call-0001", state="confirmed", final_record=record
    )
    with pytest.raises(OutcomeMappingError):
        build_outcome_payload(event)


def test_payload_with_bad_status_fails_contract_validation() -> None:
    """The shared schema, not a local shape check, rejects bad statuses."""
    with pytest.raises(OutcomeMappingError):
        validate_outcome_payload("call-0001", {"status": "perhaps"})


def test_parse_booking_event_accepts_flat_and_nested_webhooks() -> None:
    """Webhook bodies may carry status at the top level or under ``outcome``."""
    flat = parse_booking_event({"event_id": "evt-4", "call_id": "call-0001", "state": "confirmed"})
    nested = parse_booking_event(
        {
            "event_id": "evt-5",
            "call_id": "call-0001",
            "outcome": {"status": "escalated", "updated_at": STAMP},
        }
    )
    assert flat.state == "confirmed"
    assert nested.state == "escalated"
    assert nested.updated_at == STAMP


@pytest.mark.parametrize(
    "body",
    [
        {"state": "confirmed"},
        {"call_id": "", "state": "confirmed"},
        {"call_id": "call-0001", "state": ""},
        {"call_id": "call-0001", "state": "confirmed", "final_record": []},
        {"call_id": "call-0001", "outcome": "confirmed"},
    ],
)
def test_parse_booking_event_rejects_malformed_bodies(body: dict[str, Any]) -> None:
    """Malformed webhook bodies produce a mapping error, never silent data."""
    with pytest.raises(OutcomeMappingError):
        parse_booking_event(body)
