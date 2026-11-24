"""In-process webhook source: vendors call ``receive`` when a booking changes."""

from __future__ import annotations

import threading
from collections import deque
from collections.abc import Mapping, Sequence
from typing import Any

import structlog

from .interface import BookingEvent
from .mapping import parse_booking_event

logger = structlog.get_logger("voiceloop.connectors.webhook")


class WebhookBookingSource:
    """Bounded queue fed by webhook deliveries; safe to call from any thread.

    Receiving is fail-open: malformed bodies are logged and rejected with
    ``False`` rather than raising into the caller.
    """

    def __init__(self, *, max_queue: int = 256) -> None:
        """Create an empty queue with a bounded capacity.

        Args:
            max_queue: Maximum queued events; the oldest is dropped when full.

        """
        self._max_queue = max(1, max_queue)
        self._queue: deque[BookingEvent] = deque()
        self._lock = threading.Lock()

    @property
    def source_name(self) -> str:
        """Return the mode label used in logs and reports."""
        return "webhook"

    def receive(self, payload: Mapping[str, Any] | BookingEvent) -> bool:
        """Accept one booking event or webhook body without raising.

        Args:
            payload: A :class:`BookingEvent`, or a mapping accepted by
                :func:`sdk.connectors.mapping.parse_booking_event`.

        Returns:
            ``True`` when queued, ``False`` when malformed or dropped.

        """
        try:
            event = payload if isinstance(payload, BookingEvent) else parse_booking_event(payload)
        except Exception as exc:
            self._log("warning", "connector_webhook_rejected", error=str(exc))
            return False
        with self._lock:
            if len(self._queue) >= self._max_queue:
                self._queue.popleft()
                self._log("warning", "connector_webhook_dropped_oldest", call_id=event.call_id)
            self._queue.append(event)
        return True

    def collect(self) -> Sequence[BookingEvent]:
        """Drain queued events, returning an empty tuple on failure."""
        try:
            with self._lock:
                events = tuple(self._queue)
                self._queue.clear()
            return events
        except Exception as exc:
            self._log("error", "connector_webhook_collect_failed", error=str(exc))
            return ()

    def acknowledge(self, event_id: str) -> None:
        """No-op: collected webhook events are already removed from the queue."""
        return None

    def pending_count(self) -> int:
        """Return the number of undelivered queued events."""
        with self._lock:
            return len(self._queue)

    @staticmethod
    def _log(level: str, message: str, **fields: Any) -> None:
        """Emit a structured log line, swallowing logging failures."""
        try:
            event = getattr(logger, level, logger.warning)
            event(message, **fields)
        except Exception:
            pass
