"""Map booking states to the turn-event ``outcome`` object and validate it."""

from __future__ import annotations

import copy
import re
from collections.abc import Mapping
from enum import StrEnum
from typing import Any
from urllib.parse import quote
from uuid import NAMESPACE_URL, uuid5

from schemas.booking import validate_booking
from voiceloop_common import validate
from voiceloop_common.ids import new_id

from .interface import BookingEvent


class OutcomeStatus(StrEnum):
    """The ``outcome.status`` enum from ``turn_event.schema.json`` 1.0.0."""

    CONFIRMED = "confirmed"
    EDITED_BY_STAFF = "edited_by_staff"
    ABANDONED = "abandoned"
    TRANSFERRED = "transferred"
    UNKNOWN = "unknown"


class OutcomeMappingError(ValueError):
    """Raised when a booking event cannot be mapped to a valid outcome payload."""


_STATE_ALIASES: dict[str, OutcomeStatus] = {
    "confirmed": OutcomeStatus.CONFIRMED,
    "booked": OutcomeStatus.CONFIRMED,
    "completed": OutcomeStatus.CONFIRMED,
    "success": OutcomeStatus.CONFIRMED,
    "edited_by_staff": OutcomeStatus.EDITED_BY_STAFF,
    "staff_edited": OutcomeStatus.EDITED_BY_STAFF,
    "modified_by_staff": OutcomeStatus.EDITED_BY_STAFF,
    "corrected_by_staff": OutcomeStatus.EDITED_BY_STAFF,
    "abandoned": OutcomeStatus.ABANDONED,
    "cancelled": OutcomeStatus.ABANDONED,
    "canceled": OutcomeStatus.ABANDONED,
    "dropped": OutcomeStatus.ABANDONED,
    "transferred": OutcomeStatus.TRANSFERRED,
    "escalated": OutcomeStatus.TRANSFERRED,
    "transferred_to_human": OutcomeStatus.TRANSFERRED,
    "unknown": OutcomeStatus.UNKNOWN,
}

_OUTCOME_TO_STATE: dict[str, str] = {
    OutcomeStatus.CONFIRMED.value: "confirmed",
    OutcomeStatus.EDITED_BY_STAFF.value: "edited_by_staff",
    OutcomeStatus.ABANDONED.value: "abandoned",
    OutcomeStatus.TRANSFERRED.value: "transferred",
    OutcomeStatus.UNKNOWN.value: "unknown",
}


def normalize_state(state: str) -> str:
    """Normalize a vendor booking state for lookup.

    Args:
        state: Raw state name from the booking system.

    Returns:
        A lowercase snake-case key; an empty string for unusable input.

    """
    if not isinstance(state, str):
        return ""
    return re.sub(r"[^a-z0-9]+", "_", state.strip().lower()).strip("_")


def map_booking_state(state: str) -> OutcomeStatus:
    """Map a booking state to ``outcome.status``, defaulting to ``unknown``.

    Args:
        state: Raw booking state, e.g. ``confirmed`` or ``staff_edited``.

    Returns:
        The matching :class:`OutcomeStatus`; unrecognised states are ``UNKNOWN``.

    """
    return _STATE_ALIASES.get(normalize_state(state), OutcomeStatus.UNKNOWN)


def outcome_to_booking_state(status: str) -> str:
    """Return the canonical booking state for a fixture outcome status.

    Used only to seed the mock booking system from synthetic fixtures.
    """
    return _OUTCOME_TO_STATE.get(normalize_state(status), "unknown")


def parse_booking_event(payload: Mapping[str, Any]) -> BookingEvent:
    """Normalize a webhook-style booking body into a :class:`BookingEvent`.

    Args:
        payload: Mapping with ``call_id``, a ``state`` (or ``status``), optional
            ``event_id``, ``final_record`` and ``updated_at``. A nested
            ``outcome`` object is also accepted for vendors that wrap the body.

    Returns:
        A frozen event with a copied final record.

    Raises:
        OutcomeMappingError: If required fields are missing or have wrong types.

    """
    if not isinstance(payload, Mapping):
        raise OutcomeMappingError("Booking webhook body must be an object.")

    call_id = payload.get("call_id")
    if not isinstance(call_id, str) or not call_id.strip():
        raise OutcomeMappingError("Booking webhook body requires a non-empty call_id.")

    nested = payload.get("outcome")
    if nested is not None and not isinstance(nested, Mapping):
        raise OutcomeMappingError("Booking webhook 'outcome' must be an object when present.")
    source: Mapping[str, Any] = nested if isinstance(nested, Mapping) else payload

    state = source.get("status", payload.get("state", payload.get("status")))
    if not isinstance(state, str) or not state.strip():
        raise OutcomeMappingError("Booking webhook body requires a booking state.")

    raw_event_id = payload.get("event_id") or new_id()
    if not isinstance(raw_event_id, str) or not raw_event_id.strip():
        raise OutcomeMappingError("Booking webhook event_id must be a non-empty string.")

    record = payload.get("final_record", source.get("final_record"))
    if record is not None and not isinstance(record, Mapping):
        raise OutcomeMappingError("Booking webhook final_record must be an object when present.")

    updated_at = payload.get("updated_at", source.get("updated_at"))
    if updated_at is not None and not isinstance(updated_at, str):
        raise OutcomeMappingError("Booking webhook updated_at must be a string when present.")

    return BookingEvent(
        event_id=raw_event_id,
        call_id=call_id,
        state=state,
        final_record=copy.deepcopy(dict(record)) if isinstance(record, Mapping) else None,
        updated_at=updated_at,
    )


def build_outcome_payload(event: BookingEvent) -> dict[str, Any]:
    """Build a PATCH body matching the ``outcome`` object in the turn-event schema.

    Args:
        event: Normalized booking event. Its final record, when present, must
            satisfy the active booking schema and its cross-field validators.

    Returns:
        A dict with ``status`` and optional ``final_record``/``updated_at``.

    Raises:
        OutcomeMappingError: If the call id or final record is invalid.

    """
    if not isinstance(event.call_id, str) or not event.call_id.strip():
        raise OutcomeMappingError("Booking event requires a non-empty call_id.")

    payload: dict[str, Any] = {"status": map_booking_state(event.state).value}
    if event.final_record is not None:
        if not isinstance(event.final_record, Mapping):
            raise OutcomeMappingError("Booking event final_record must be an object.")
        record = copy.deepcopy(dict(event.final_record))
        try:
            validate_booking(record)
            validate(record, "booking.schema.json")
        except Exception as exc:
            raise OutcomeMappingError(
                f"Invalid final_record for call {event.call_id}: {exc}"
            ) from exc
        payload["final_record"] = record
    if event.updated_at is not None:
        if not isinstance(event.updated_at, str):
            raise OutcomeMappingError("Booking event updated_at must be a string.")
        payload["updated_at"] = event.updated_at
    return payload


def validate_outcome_payload(call_id: str, payload: Mapping[str, Any]) -> None:
    """Validate a PATCH body against the authoritative turn-event schema.

    The body is wrapped in a minimal carrier event so validation uses the
    committed ``outcome`` definition rather than a parallel local schema.

    Args:
        call_id: Destination call; the only join key used by the connector.
        payload: The ``outcome`` object destined for the PATCH endpoint.

    Raises:
        OutcomeMappingError: If the merged carrier event fails validation.

    """
    outcome = copy.deepcopy(dict(payload))
    stamp = outcome.get("updated_at", "")
    identity = f"urn:voiceloop:outcome:{call_id}:{stamp}:{outcome.get('status', '')}"
    carrier: dict[str, Any] = {
        "event_id": str(uuid5(NAMESPACE_URL, identity)),
        "call_id": call_id,
        "turn_index": 0,
        "audio": {"uri": f"voiceloop://outcome/{quote(call_id, safe='')}"},
        "stt": {"text": ""},
        "outcome": outcome,
    }
    try:
        validate(carrier, "turn_event")
    except Exception as exc:
        raise OutcomeMappingError(
            f"Outcome payload for call {call_id} failed turn_event validation: {exc}"
        ) from exc
