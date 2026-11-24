"""Protocols and value types shared by outcome sources, sinks and the connector."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable


@dataclass(frozen=True)
class BookingEvent:
    """One booking-state change, joined to a call by ``call_id`` only.

    ``event_id`` is the source-side deduplication key. ``final_record`` is an
    optional booking-slot object; it is never invented by the connector.
    """

    event_id: str
    call_id: str
    state: str
    final_record: Mapping[str, Any] | None = None
    updated_at: str | None = None


@dataclass(frozen=True)
class DeliveryReceipt:
    """Result of one ``PATCH /v1/calls/{call_id}/outcome`` attempt."""

    ok: bool
    status_code: int | None = None
    error: str | None = None


@dataclass(frozen=True)
class SyncReport:
    """Counts from one connector pass; failures are counted, never raised."""

    source: str
    considered: int = 0
    delivered: int = 0
    skipped: int = 0
    failed: int = 0


@runtime_checkable
class BookingOutcomeSource(Protocol):
    """Where booking events come from: a webhook queue or a polling adapter."""

    @property
    def source_name(self) -> str:
        """Return the mode label used in logs and reports."""
        ...

    def collect(self) -> Sequence[BookingEvent]:
        """Return newly available events without blocking or raising."""
        ...

    def acknowledge(self, event_id: str) -> None:
        """Mark one delivered event as handled so it is not re-emitted."""
        ...


@runtime_checkable
class BookingSnapshotSource(Protocol):
    """Read side used by a real booking API adapter and by the mock system."""

    def list_booking_events(self) -> Sequence[BookingEvent]:
        """Return the source's current ordered list of booking events."""
        ...


@runtime_checkable
class OutcomeSink(Protocol):
    """Where validated outcome payloads go: ingest HTTP or an in-memory fake."""

    def patch_outcome(self, call_id: str, payload: Mapping[str, Any]) -> DeliveryReceipt:
        """Send an outcome body for one call and report success or failure."""
        ...
