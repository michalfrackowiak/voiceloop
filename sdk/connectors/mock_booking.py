"""In-memory mock booking system and fixture-event sink for connector demos."""

from __future__ import annotations

import copy
import json
import threading
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any
from uuid import NAMESPACE_URL, uuid5

import structlog

from schemas.booking import validate_booking
from voiceloop_common import utcnow, validate
from voiceloop_common.ids import new_id

from .interface import BookingEvent, DeliveryReceipt
from .mapping import OutcomeMappingError, outcome_to_booking_state, validate_outcome_payload

logger = structlog.get_logger("voiceloop.connectors.mock")

_REPO_ROOT = Path(__file__).resolve().parents[2]


def _fixture_root(fixture_dir: str | Path | None = None) -> Path:
    """Resolve a fixture directory relative to the repository root."""
    if fixture_dir is None:
        return _REPO_ROOT / "tests" / "fixtures"
    candidate = Path(fixture_dir).expanduser()
    if not candidate.is_absolute():
        candidate = _REPO_ROOT / candidate
    return candidate.resolve()


def _load_labels(root: Path) -> list[dict[str, Any]]:
    """Read the synthetic labels manifest as a list of mappings."""
    labels: list[dict[str, Any]] = []
    with (root / "turns.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            value = json.loads(line)
            if isinstance(value, dict):
                labels.append(value)
    return labels


def _stamp() -> str:
    """Return the UTC time as an RFC 3339 string for mock updates."""
    return utcnow().isoformat().replace("+00:00", "Z")


def _clone(event: BookingEvent) -> BookingEvent:
    """Copy a booking event so callers cannot mutate the source store."""
    record = event.final_record
    return BookingEvent(
        event_id=event.event_id,
        call_id=event.call_id,
        state=event.state,
        final_record=copy.deepcopy(dict(record)) if isinstance(record, Mapping) else None,
        updated_at=event.updated_at,
    )


class MockBookingSystem:
    """In-memory booking system seeded only from synthetic fixture records.

    It exposes :meth:`list_booking_events` so a poller can consume it, and
    :meth:`update` to simulate a late booking-state change for a fixture call.
    No vendor API or network is involved.
    """

    def __init__(self, events: Iterable[BookingEvent] = ()) -> None:
        """Create a store preloaded with booking events."""
        self._lock = threading.RLock()
        self._events: list[BookingEvent] = [_clone(event) for event in events]

    @classmethod
    def seed_from_fixtures(cls, fixture_dir: str | Path | None = None) -> MockBookingSystem:
        """Seed one booking event per fixture call from ``turns.jsonl``.

        Args:
            fixture_dir: Optional fixture directory; defaults to
                ``tests/fixtures`` at the repository root.

        Returns:
            A mock system whose final records are only synthetic fixture slots.

        """
        root = _fixture_root(fixture_dir)
        by_call: dict[str, dict[str, Any]] = {}
        for label in _load_labels(root):
            call_id = label.get("call_id")
            if not isinstance(call_id, str) or not call_id.strip():
                continue
            current = by_call.setdefault(call_id, {"record": None, "outcome": None})
            record = label.get("expected_slots")
            if isinstance(record, dict) and record:
                current["record"] = record
            outcome = label.get("expected_outcome")
            if outcome:
                current["outcome"] = outcome

        events: list[BookingEvent] = []
        for call_id in sorted(by_call):
            info = by_call[call_id]
            record = info["record"]
            if isinstance(record, dict):
                try:
                    validate_booking(record)
                except Exception as exc:
                    logger.warning(
                        "connector_mock_seed_skipped_record", call_id=call_id, error=str(exc)
                    )
                    record = None
            outcome = info["outcome"]
            events.append(
                BookingEvent(
                    event_id=str(uuid5(NAMESPACE_URL, f"urn:voiceloop:booking:{call_id}:seed")),
                    call_id=call_id,
                    state=outcome_to_booking_state(str(outcome)) if outcome else "unknown",
                    final_record=copy.deepcopy(record) if isinstance(record, dict) else None,
                    updated_at=_stamp(),
                )
            )
        return cls(events)

    def update(
        self,
        call_id: str,
        state: str,
        final_record: Mapping[str, Any] | None = None,
    ) -> BookingEvent:
        """Append a late booking-state change for one call.

        Args:
            call_id: Fixture call id; the only join key.
            state: Raw booking state, e.g. ``edited_by_staff``.
            final_record: Optional synthetic booking slots; never invented here.

        Returns:
            The new booking event.

        Raises:
            ValueError: If the call id is empty or the record is not an object.
            schemas.booking.BookingValidationError: If the record is invalid.

        """
        if not isinstance(call_id, str) or not call_id.strip():
            raise ValueError("call_id must be a non-empty string.")
        record: dict[str, Any] | None = None
        if final_record is not None:
            if not isinstance(final_record, Mapping):
                raise ValueError("final_record must be an object when present.")
            record = copy.deepcopy(dict(final_record))
            validate_booking(record)
        event = BookingEvent(
            event_id=new_id(),
            call_id=call_id,
            state=state,
            final_record=record,
            updated_at=_stamp(),
        )
        with self._lock:
            self._events.append(event)
        return event

    def list_booking_events(self) -> Sequence[BookingEvent]:
        """Return a snapshot of all booking events in insertion order."""
        with self._lock:
            return tuple(_clone(event) for event in self._events)


class FixtureEventStore:
    """In-memory copies of fixture turn events with outcomes merged by call id."""

    def __init__(self) -> None:
        """Create an empty store."""
        self._lock = threading.RLock()
        self._events: dict[str, list[dict[str, Any]]] = {}

    @classmethod
    def load(cls, fixture_dir: str | Path | None = None) -> FixtureEventStore:
        """Load all fixture turn events, validating each with the shared schema.

        Args:
            fixture_dir: Optional fixture directory; defaults to
                ``tests/fixtures`` at the repository root.

        Returns:
            A store keyed by fixture call id.

        Raises:
            voiceloop_common.errors.ValidationError: If a fixture event is invalid.

        """
        root = _fixture_root(fixture_dir)
        store = cls()
        for label in _load_labels(root):
            relative = label.get("event_path")
            if not isinstance(relative, str) or not relative:
                continue
            path = Path(relative)
            if not path.is_absolute():
                path = _REPO_ROOT / relative
                if not path.exists():
                    path = root / relative
            event = json.loads(path.read_text(encoding="utf-8"))
            validate(event, "turn_event")
            call_id = str(event["call_id"])
            with store._lock:
                store._events.setdefault(call_id, []).append(event)
        return store

    def apply_outcome(self, call_id: str, outcome: Mapping[str, Any]) -> int:
        """Merge one outcome into every stored turn for a call.

        Args:
            call_id: Destination call; no other join key is used.
            outcome: Validated ``outcome`` object to attach.

        Returns:
            The number of updated fixture events.

        Raises:
            OutcomeMappingError: If a merged event fails contract validation.

        """
        merged = 0
        with self._lock:
            for index, event in enumerate(self._events.get(call_id, [])):
                updated = copy.deepcopy(event)
                updated["outcome"] = copy.deepcopy(dict(outcome))
                try:
                    validate(updated, "turn_event")
                except Exception as exc:
                    raise OutcomeMappingError(
                        f"Merged event for call {call_id} failed validation: {exc}"
                    ) from exc
                self._events[call_id][index] = updated
                merged += 1
        return merged

    def events_for(self, call_id: str) -> list[dict[str, Any]]:
        """Return deep copies of the stored turns for one call."""
        with self._lock:
            return copy.deepcopy(self._events.get(call_id, []))


class InMemoryOutcomeSink:
    """Fake PATCH sink that records payloads and updates fixture events."""

    def __init__(self, events: FixtureEventStore | None = None) -> None:
        """Create a sink, optionally linked to a fixture event store."""
        self._events = events
        self._lock = threading.Lock()
        self._deliveries: list[tuple[str, dict[str, Any]]] = []

    def patch_outcome(self, call_id: str, payload: Mapping[str, Any]) -> DeliveryReceipt:
        """Validate, record and optionally merge an outcome; never raises."""
        try:
            validate_outcome_payload(call_id, payload)
            snapshot = copy.deepcopy(dict(payload))
            if self._events is not None:
                self._events.apply_outcome(call_id, snapshot)
            with self._lock:
                self._deliveries.append((call_id, snapshot))
        except Exception as exc:
            logger.warning("connector_mock_patch_failed", call_id=call_id, error=str(exc))
            return DeliveryReceipt(ok=False, error=str(exc))
        return DeliveryReceipt(ok=True, status_code=202)

    def deliveries(self) -> list[tuple[str, dict[str, Any]]]:
        """Return recorded ``(call_id, payload)`` pairs in delivery order."""
        with self._lock:
            return copy.deepcopy(self._deliveries)
