"""Bounded fail-open buffer that never blocks a live call on storage."""

from __future__ import annotations

import logging
import queue
import time
from collections.abc import Mapping
from concurrent.futures import Future
from concurrent.futures import TimeoutError as FutureTimeout
from typing import Any

from voiceloop_common import new_id, utcnow

from .executors import DaemonExecutor, WriterBusy
from .metrics import FailOpenMetrics, HealthSnapshot
from .models import BufferedEvent, EnqueueResult, FlushResult, ShedReason
from .payload import PayloadError, PayloadTimeout, sanitize_payload
from .settings import FailOpenSettings
from .sinks import StorageSink
from .store import DiskStore, EventStore, MemoryStore, StoreTimeout

logger = logging.getLogger(__name__)


def _stamp() -> str:
    """Return the current UTC time as an RFC 3339 string."""
    return utcnow().isoformat().replace("+00:00", "Z")


class BoundedBuffer:
    """Accept events within a bounded timeout and drain them to storage later.

    ``enqueue`` never touches the live storage sink and never raises: callers
    get an explicit ``accepted``/``shed`` signal. Payload preparation, capacity
    waits and disk writes all share one deadline. ``flush`` runs sink writes in
    daemon threads so a blocked sink cannot extend its deadline.
    """

    def __init__(
        self,
        sink: StorageSink,
        settings: FailOpenSettings | None = None,
        *,
        metrics: FailOpenMetrics | None = None,
    ) -> None:
        """Create a buffer around a storage sink and typed settings."""
        self._settings = settings or FailOpenSettings()
        self._sink = sink
        self._metrics = metrics or FailOpenMetrics(
            capacity=self._settings.capacity,
            degraded_ratio=self._settings.degraded_ratio,
        )
        self._writer = DaemonExecutor(self._settings.writer_workers)
        self._store: EventStore = (
            DiskStore(
                self._settings.capacity,
                self._settings.disk_dir,
                writer_workers=self._settings.writer_workers,
                max_bytes=self._settings.max_payload_bytes + 4096,
            )
            if self._settings.backend == "disk"
            else MemoryStore(self._settings.capacity)
        )

    def enqueue(
        self,
        event: Mapping[str, Any],
        *,
        timeout_ms: int | None = None,
    ) -> EnqueueResult:
        """Buffer one event within one total deadline; shed on full/invalid.

        Args:
            event: JSON-compatible event mapping to buffer.
            timeout_ms: Optional per-call override of the total enqueue budget.

        Returns:
            A receipt with ``signal="shed"`` and a ``reason`` on load shedding.
            No exception is ever raised into the caller.

        """
        event_id: str | None = None
        wait_ms = self._settings.enqueue_timeout_ms if timeout_ms is None else max(0, timeout_ms)
        deadline = time.monotonic() + wait_ms / 1000.0
        try:
            if not isinstance(event, Mapping):
                self._metrics.record_dropped()
                return self._shed("invalid_event")
            payload = sanitize_payload(event, self._settings.max_payload_bytes, deadline)
            event_id = str(payload.get("event_id") or new_id())
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise PayloadTimeout("enqueue budget exhausted")
            item = BufferedEvent(
                event_id=event_id,
                payload=payload,
                enqueued_at=_stamp(),
            )
            self._store.put(item, remaining)
        except PayloadError:
            self._metrics.record_dropped()
            return self._shed("invalid_event", event_id)
        except (PayloadTimeout, StoreTimeout):
            self._metrics.record_dropped()
            return self._shed("internal_error", event_id)
        except queue.Full:
            self._metrics.record_dropped()
            return self._shed("queue_full", event_id)
        except Exception as exc:
            logger.warning("failopen_enqueue_error error_type=%s", type(exc).__name__)
            self._metrics.record_dropped()
            return self._shed("internal_error", event_id)
        self._metrics.record_accepted()
        self._metrics.record_buffered()
        return EnqueueResult(accepted=True, signal="accepted", event_id=item.event_id)

    def flush(
        self,
        *,
        max_events: int | None = None,
        timeout_s: float | None = None,
    ) -> FlushResult:
        """Drain buffered events to storage within a hard deadline.

        Failed writes are counted and requeued for the next call. One call
        attempts each event at most once, so a permanently down sink cannot
        spin inside a single flush. No exception is ever raised.

        Args:
            max_events: Optional cap on events attempted in this call.
            timeout_s: Optional override of the configured flush deadline.

        Returns:
            Counts of flushed events, remaining depth and storage failures.

        """
        try:
            return self._drain(max_events=max_events, timeout_s=timeout_s)
        except Exception as exc:
            logger.warning("failopen_flush_error error_type=%s", type(exc).__name__)
            return FlushResult(
                flushed=0,
                remaining=self.depth,
                timed_out=False,
                storage_failures=0,
            )

    def health(self) -> HealthSnapshot:
        """Return health status, current depth and cumulative counters."""
        return self._metrics.health(self.depth)

    def counters(self) -> dict[str, int]:
        """Return a copy of the current metric counters."""
        return self._metrics.counters()

    def close(self) -> None:
        """Release store resources; safe to call more than once."""
        try:
            self._store.close()
        except Exception as exc:
            logger.warning("failopen_close_error error_type=%s", type(exc).__name__)
        finally:
            self._writer.close()

    @property
    def depth(self) -> int:
        """Return the current number of buffered events."""
        try:
            return len(self._store)
        except Exception:
            return 0

    def _drain(
        self,
        *,
        max_events: int | None,
        timeout_s: float | None,
    ) -> FlushResult:
        """Perform one bounded drain, requeuing events that fail to write."""
        budget = self._settings.flush_timeout_s if timeout_s is None else timeout_s
        deadline = time.monotonic() + max(0.0, budget)
        attempted = 0
        flushed = 0
        failures = 0
        deferred: list[BufferedEvent] = []
        timed_out = False

        while max_events is None or attempted < max_events:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                timed_out = True
                break
            try:
                event = self._store.pop()
            except Exception as exc:
                logger.warning("failopen_store_pop_error error_type=%s", type(exc).__name__)
                break
            if event is None:
                break
            attempted += 1
            try:
                future = self._submit_write(event.payload)
            except WriterBusy:
                failures += 1
                self._metrics.record_storage_failure()
                deferred.append(event)
                timed_out = True
                break
            try:
                future.result(timeout=max(0.0, deadline - time.monotonic()))
            except FutureTimeout:
                failures += 1
                self._metrics.record_storage_failure()
                deferred.append(event)
                timed_out = True
                break
            except Exception:
                failures += 1
                self._metrics.record_storage_failure()
                deferred.append(event)
                continue
            flushed += 1
            self._metrics.record_flushed()
            self._ack(event)

        for event in deferred:
            try:
                self._store.restore(event)
            except Exception as exc:
                logger.warning("failopen_requeue_error error_type=%s", type(exc).__name__)
                self._metrics.record_dropped()
                self._ack(event)

        return FlushResult(
            flushed=flushed,
            remaining=self.depth,
            timed_out=timed_out,
            storage_failures=failures,
        )

    def _submit_write(self, payload: dict[str, Any]) -> Future[None]:
        """Run one sink write on a daemon worker with no implicit waiting."""
        return self._writer.submit(lambda: self._sink.write(payload))

    def _ack(self, event: BufferedEvent) -> None:
        """Acknowledge a popped event, ignoring local cleanup failures."""
        try:
            self._store.ack(event)
        except Exception as exc:
            logger.warning("failopen_store_ack_error error_type=%s", type(exc).__name__)

    def _shed(
        self,
        reason: ShedReason,
        event_id: str | None = None,
    ) -> EnqueueResult:
        """Build the explicit drop signal returned to a caller."""
        return EnqueueResult(
            accepted=False,
            signal="shed",
            event_id=event_id,
            reason=reason,
        )

    def __enter__(self) -> BoundedBuffer:
        """Return this buffer for use as a context manager."""
        return self

    def __exit__(self, *_: object) -> None:
        """Close the buffer on context-manager exit."""
        self.close()
