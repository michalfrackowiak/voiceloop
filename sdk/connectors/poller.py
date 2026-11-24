"""Polling source: fetch booking events and deduplicate by source event id."""

from __future__ import annotations

import threading
from collections.abc import Sequence
from typing import Any

import structlog

from .interface import BookingEvent, BookingSnapshotSource

logger = structlog.get_logger("voiceloop.connectors.poller")


class PollingBookingSource:
    """Poll a booking snapshot source for events not yet acknowledged.

    A failed delivery is not acknowledged, so the next poll retries it. Polling
    is fail-open: a broken booking adapter yields an empty batch and a log line.
    """

    def __init__(self, booking: BookingSnapshotSource, *, max_events: int = 100) -> None:
        """Create a poller over a booking adapter.

        Args:
            booking: Any object exposing ``list_booking_events()``.
            max_events: Upper bound on events returned per poll.

        """
        self._booking = booking
        self._max_events = max(1, max_events)
        self._seen: set[str] = set()
        self._lock = threading.Lock()

    @property
    def source_name(self) -> str:
        """Return the mode label used in logs and reports."""
        return "poller"

    def collect(self) -> Sequence[BookingEvent]:
        """Return unacknowledged events, oldest first, without raising."""
        try:
            events = list(self._booking.list_booking_events())
        except Exception as exc:
            self._log("error", "connector_poller_fetch_failed", error=str(exc))
            return ()
        try:
            with self._lock:
                batch: list[BookingEvent] = []
                batch_ids: set[str] = set()
                for event in events:
                    if event.event_id in self._seen or event.event_id in batch_ids:
                        continue
                    batch.append(event)
                    batch_ids.add(event.event_id)
                    if len(batch) >= self._max_events:
                        break
                return tuple(batch)
        except Exception as exc:
            self._log("error", "connector_poller_collect_failed", error=str(exc))
            return ()

    def acknowledge(self, event_id: str) -> None:
        """Mark one delivered event as handled; safe to call repeatedly."""
        try:
            with self._lock:
                self._seen.add(event_id)
        except Exception as exc:
            self._log("warning", "connector_poller_ack_failed", error=str(exc))

    @staticmethod
    def _log(level: str, message: str, **fields: Any) -> None:
        """Emit a structured log line, swallowing logging failures."""
        try:
            event = getattr(logger, level, logger.warning)
            event(message, **fields)
        except Exception:
            pass
