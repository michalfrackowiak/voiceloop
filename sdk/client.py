"""Fail-open turn logger: callers enqueue, a background worker delivers."""

from __future__ import annotations

import copy
import time
from collections import deque
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from enum import StrEnum
from threading import Condition, Event, Lock, Thread
from types import TracebackType
from typing import Any

import structlog

from voiceloop_common.schemas import validate

from .circuit_breaker import CircuitBreaker
from .metrics import CounterName, SdkMetrics
from .settings import SdkSettings
from .transport import HttpTransport, TurnTransport

Clock = Callable[[], float]


class EnqueueStatus(StrEnum):
    """Result of the caller-side enqueue step."""

    ACCEPTED = "accepted"
    ERROR = "error"


@dataclass(frozen=True)
class _QueueItem:
    """Internal queue entry with retry metadata."""

    event: Mapping[str, Any]
    attempts: int = 0
    not_before: float = 0.0


class SdkClient:
    """Non-blocking SDK client.

    Create one client at process startup and reuse it. ``enqueue`` performs no
    validation, logging, sleeps or network I/O; a daemon worker validates and
    sends in the background so a live call cannot be delayed or broken by the
    ingest service.
    """

    def __init__(
        self,
        settings: SdkSettings | None = None,
        transport: TurnTransport | None = None,
        *,
        clock: Clock = time.monotonic,
        start_worker: bool = True,
    ) -> None:
        """Create a client and optionally start its background worker.

        Args:
            settings: Typed SDK settings; defaults read ``VOICELOOP_SDK_*``.
            transport: Injectable transport; defaults to httpx against the
                configured ingest endpoint.
            clock: Injectable monotonic clock for retry/circuit timing.
            start_worker: Set false for deterministic queue-only tests.

        """
        self._settings = settings or SdkSettings()
        self._transport = (
            transport if transport is not None else HttpTransport(self._settings.endpoint)
        )
        self._clock = clock
        self._metrics = SdkMetrics()
        self._breaker = CircuitBreaker(
            failure_threshold=self._settings.circuit_breaker_failure_threshold,
            recovery_seconds=self._settings.circuit_breaker_recovery_seconds,
            clock=clock,
        )
        self._pending: deque[_QueueItem] = deque()
        self._retries: deque[_QueueItem] = deque()
        self._condition = Condition()
        self._stop = Event()
        self._active = False
        self._thread: Thread | None = None
        if start_worker:
            self._thread = Thread(target=self._run, name="voiceloop-sdk", daemon=True)
            self._thread.start()

    @property
    def metrics(self) -> SdkMetrics:
        """Return the counter collection; call ``snapshot()`` for one view."""
        return self._metrics

    @property
    def breaker(self) -> CircuitBreaker:
        """Return the circuit breaker for diagnostics and tests."""
        return self._breaker

    def enqueue(self, event: Mapping[str, Any]) -> EnqueueStatus:
        """Accept a turn event for background delivery without raising.

        The event is shallow-copied only. Validation and deep copying happen in
        the worker. When the bounded queue is full the oldest item is dropped
        and counted; the new event is still accepted.
        """
        try:
            return self._enqueue(event)
        except Exception:
            self._increment("enqueue_errors")
            return EnqueueStatus.ERROR

    def wait_until_idle(self, timeout_seconds: float = 5.0) -> bool:
        """Wait until pending and retrying events have been processed.

        Intended for tests and graceful shutdown, never for the call path.
        Returns true when the worker is idle, false on timeout.
        """
        deadline = self._safe_now() + max(0.0, timeout_seconds)
        with self._condition:
            while self._pending or self._retries or self._active:
                remaining = deadline - self._safe_now()
                if remaining <= 0:
                    return False
                self._condition.wait(timeout=min(remaining, 0.05))
            return True

    def pending_event_ids(self) -> list[str]:
        """Return queued event IDs, oldest first, for diagnostics and tests."""
        with self._condition:
            return [str(item.event.get("event_id")) for item in self._pending]

    def close(self, timeout_seconds: float = 5.0) -> None:
        """Stop the worker and close the transport; safe to call repeatedly."""
        with self._condition:
            self._stop.set()
            self._condition.notify_all()
        thread = self._thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=max(0.0, timeout_seconds))
        close = getattr(self._transport, "close", None)
        if callable(close):
            try:
                close()
            except Exception:
                self._increment("internal_errors")

    def __enter__(self) -> SdkClient:
        """Enter a context that closes the client on exit."""
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Stop the worker and transport when leaving a context."""
        self.close()

    def _enqueue(self, event: Mapping[str, Any]) -> EnqueueStatus:
        """Copy the top level of an event and append it to the bounded queue."""
        snapshot = dict(event)
        with self._condition:
            if len(self._pending) >= self._settings.queue_capacity:
                self._pending.popleft()
                self._increment("dropped_oldest")
            self._pending.append(_QueueItem(event=snapshot, not_before=self._safe_now()))
            self._increment("accepted")
            self._condition.notify()
        return EnqueueStatus.ACCEPTED

    def _run(self) -> None:
        """Consume queue items until stopped."""
        while not self._stop.is_set():
            item = self._take_item()
            if item is None:
                continue
            try:
                self._process(item)
            except Exception as exc:
                self._increment("internal_errors")
                self._log("error", "SDK failed to process a turn event", error=str(exc))
            finally:
                with self._condition:
                    self._active = False
                    self._condition.notify_all()

    def _take_item(self) -> _QueueItem | None:
        """Pop the next due item, waiting up to 50 ms for work or shutdown."""
        with self._condition:
            while not self._stop.is_set():
                now = self._safe_now()
                if self._pending:
                    item = self._pending.popleft()
                    self._active = True
                    return item
                due_index = next(
                    (index for index, item in enumerate(self._retries) if item.not_before <= now),
                    None,
                )
                if due_index is not None:
                    item = self._retries[due_index]
                    del self._retries[due_index]
                    self._active = True
                    return item
                wait_seconds = 0.05
                if self._retries:
                    soonest = min(item.not_before for item in self._retries)
                    wait_seconds = min(wait_seconds, max(0.001, soonest - now))
                self._condition.wait(timeout=wait_seconds)
            return None

    def _process(self, item: _QueueItem) -> None:
        """Validate and send one item, applying the breaker and retry policy."""
        if not self._breaker.allow():
            self._increment("shed_by_breaker")
            self._log("warning", "SDK circuit breaker open; shedding turn event")
            return
        try:
            event = copy.deepcopy(dict(item.event))
        except Exception as exc:
            self._increment("validation_failures")
            self._increment("dropped_invalid")
            self._log("warning", "SDK dropped an uncopyable turn event", error=str(exc))
            return
        try:
            validate(event, "turn_event")
        except Exception as exc:
            self._increment("validation_failures")
            self._increment("dropped_invalid")
            self._log("warning", "SDK dropped an invalid turn event", error=str(exc))
            return
        try:
            self._transport.send(event, timeout_seconds=self._settings.timeout_seconds)
        except Exception as exc:
            self._handle_failure(item, exc)
            return
        self._breaker.record_success()
        self._increment("sent")

    def _handle_failure(self, item: _QueueItem, exc: Exception) -> None:
        """Record a failed send and schedule a retry when attempts remain."""
        self._breaker.record_failure()
        self._increment("send_failures")
        self._log(
            "warning",
            "SDK failed to send a turn event",
            error=str(exc),
            attempts=item.attempts + 1,
        )
        if item.attempts >= self._settings.max_retries:
            self._increment("dropped_after_retries")
            return
        retry = _QueueItem(
            event=item.event,
            attempts=item.attempts + 1,
            not_before=self._safe_now() + self._settings.retry_backoff_seconds,
        )
        with self._condition:
            self._retries.append(retry)
            self._condition.notify()
        self._increment("retried")

    def _increment(self, name: CounterName) -> None:
        """Increment a counter, swallowing any counter failure."""
        try:
            self._metrics.increment(name)
        except Exception:
            pass

    def _safe_now(self) -> float:
        """Read the injected clock, falling back to the system clock on error."""
        try:
            return float(self._clock())
        except Exception:
            return time.monotonic()

    @staticmethod
    def _log(level: str, message: str, **fields: Any) -> None:
        """Emit a structured log event, swallowing any logging failure."""
        try:
            logger = structlog.get_logger("voiceloop.sdk")
            if level == "error":
                logger.error(message, **fields)
            else:
                logger.warning(message, **fields)
        except Exception:
            pass


_default_client: SdkClient | None = None
_default_lock = Lock()


def get_default_client() -> SdkClient:
    """Return the process-wide default client, creating it on first use."""
    global _default_client  # noqa: PLW0603 - deliberate process-wide SDK default
    with _default_lock:
        if _default_client is None:
            _default_client = SdkClient()
        return _default_client


def log_turn(event: Mapping[str, Any], *, client: SdkClient | None = None) -> EnqueueStatus:
    """Queue a turn event without validation, I/O or exceptions.

    This is the call-path entry point. It returns promptly even when ingest is
    unreachable; a background worker handles validation, retries and delivery.
    Callers must not rely on the return value for correctness: any delivery
    failure is reflected in :class:`SdkMetrics` and SDK logs instead.

    Args:
        event: Turn event conforming to ``turn_event.schema.json`` 1.0.0.
        client: Explicit client, normally created at startup. If omitted, the
            process-wide default client is used.

    Returns:
        ``ACCEPTED`` when the event was queued (possibly dropping an older
        event), or ``ERROR`` if even the enqueue failed.

    """
    try:
        return (client if client is not None else get_default_client()).enqueue(event)
    except Exception:
        return EnqueueStatus.ERROR
