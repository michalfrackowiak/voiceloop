"""Bounded, non-blocking write queue with a fail-open background worker.

HTTP handlers validate and enqueue only. All repository and object-store calls
happen in the worker, so a storage outage can never add latency to a live call
or raise into the caller. Full queues shed the new item with a ``shed`` receipt
rather than blocking; ``P1-02`` can extend the buffer and backpressure signals.
"""

from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass
from typing import Any, Literal, cast

import structlog

from services.ingest.models import AudioState, Receipt
from services.ingest.repository import IngestRepository
from services.ingest.settings import IngestSettings
from services.ingest.storage import ObjectStore
from voiceloop_common import utcnow
from voiceloop_common.errors import StorageError

logger = structlog.get_logger(__name__)


@dataclass(frozen=True)
class Envelope:
    """One deferred write operation."""

    kind: Literal["turn", "outcome"]
    event_id: str
    call_id: str
    payload: dict[str, Any]


def _stamp() -> str:
    """Return the current UTC time as an RFC 3339 string."""
    return utcnow().isoformat().replace("+00:00", "Z")


class IngestQueue:
    """Idempotency ledger plus a bounded queue drained by a daemon worker."""

    def __init__(
        self,
        settings: IngestSettings,
        repository: IngestRepository,
        object_store: ObjectStore,
        *,
        autostart: bool = True,
    ) -> None:
        """Create an empty queue; ``autostart`` enables lazy background writes."""
        self._settings = settings
        self._repository = repository
        self._object_store = object_store
        self._autostart = autostart
        self._queue: queue.Queue[Envelope] = queue.Queue(maxsize=settings.queue_capacity)
        self._stats_lock = threading.Lock()
        self._counters = {"processed": 0, "failed": 0, "shed": 0, "duplicates": 0}
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()

    def enqueue(self, envelope: Envelope, receipt: Receipt) -> tuple[Receipt, bool]:
        """Claim an event id and enqueue its write without waiting for storage.

        Returns:
            The receipt to return to the caller and a flag that is True when an
            existing receipt was found (duplicate submission).

        """
        existing = self._repository.claim_event(envelope.event_id, receipt.model_dump())
        if existing is not None:
            self._increment("duplicates")
            logger.info(
                "ingest_duplicate",
                event_id=envelope.event_id,
                call_id=envelope.call_id,
                kind=envelope.kind,
            )
            return Receipt.model_validate(existing), True

        try:
            self._queue.put_nowait(envelope)
        except queue.Full:
            shed = receipt.model_copy(update={"status": "shed", "error": "queue_full"})
            self._repository.save_receipt(envelope.event_id, shed.model_dump())
            self._increment("shed")
            logger.warning(
                "ingest_shed",
                event_id=envelope.event_id,
                call_id=envelope.call_id,
                kind=envelope.kind,
            )
            return shed, False

        if self._autostart:
            self.start()
        return receipt, False

    def start(self) -> None:
        """Start the background worker once (idempotent)."""
        with self._stats_lock:
            if self._thread is not None and self._thread.is_alive():
                return
            self._stop.clear()
            self._thread = threading.Thread(
                target=self._run,
                name="voiceloop-ingest-worker",
                daemon=True,
            )
            self._thread.start()

    def stop(self, timeout: float | None = None) -> None:
        """Signal the worker to stop and wait up to ``timeout`` seconds."""
        self._stop.set()
        thread = self._thread
        if thread is not None and thread.is_alive():
            thread.join(
                timeout if timeout is not None else self._settings.worker_shutdown_timeout_s
            )

    def process_pending(self) -> int:
        """Drain the queue in the calling thread; used for deterministic tests."""
        processed = 0
        while True:
            try:
                envelope = self._queue.get_nowait()
            except queue.Empty:
                return processed
            try:
                self._process(envelope)
                processed += 1
            finally:
                self._queue.task_done()

    def flush(self, timeout: float | None = None) -> bool:
        """Wait for queued writes to finish; return False on timeout."""
        if timeout is None:
            timeout = self._settings.worker_shutdown_timeout_s
        deadline = time.monotonic() + timeout
        with self._queue.all_tasks_done:
            while self._queue.unfinished_tasks > 0:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return False
                self._queue.all_tasks_done.wait(min(0.01, remaining))
        return True

    def stats(self) -> dict[str, int]:
        """Return queue depth and cumulative counters for health checks."""
        with self._stats_lock:
            counts = dict(self._counters)
        return {
            "queued": self._queue.qsize(),
            **counts,
        }

    def _increment(self, name: str) -> None:
        """Increment one counter under the shared lock."""
        with self._stats_lock:
            self._counters[name] += 1

    def _run(self) -> None:
        """Worker loop; a failed write is recorded but never re-raised."""
        while not self._stop.is_set():
            try:
                envelope = self._queue.get(timeout=self._settings.worker_poll_interval_s)
            except queue.Empty:
                continue
            try:
                self._process(envelope)
            finally:
                self._queue.task_done()

    def _process(self, envelope: Envelope) -> None:
        """Persist one envelope and update its receipt, isolating all failures."""
        audio_state: AudioState | None = None
        error: str | None = None
        try:
            if envelope.kind == "turn":
                audio_state = self._persist_turn(envelope)
            else:
                self._repository.merge_outcome(
                    envelope.call_id,
                    envelope.payload,
                    source_event_id=envelope.event_id,
                )
            self._increment("processed")
        except Exception as exc:
            error = "storage_error" if isinstance(exc, StorageError) else type(exc).__name__
            self._increment("failed")
            logger.error(
                "ingest_write_failed",
                event_id=envelope.event_id,
                call_id=envelope.call_id,
                kind=envelope.kind,
                error=error,
            )

        updated = Receipt(
            event_id=envelope.event_id,
            call_id=envelope.call_id,
            kind=envelope.kind,
            status="failed" if error else "persisted",
            audio_state=audio_state,
            error=error,
        )
        try:
            self._repository.save_receipt(envelope.event_id, updated.model_dump())
        except Exception:
            logger.error("ingest_receipt_write_failed", event_id=envelope.event_id)

    def _persist_turn(self, envelope: Envelope) -> AudioState:
        """Write all metadata for a validated turn and return its audio state."""
        event = envelope.payload
        self._repository.save_call(event["call_id"], event.get("consent"))
        for kind, block_name in (("stt", "stt"), ("parser", "parser")):
            block = event.get(block_name) or {}
            if block.get("model_id") and block.get("model_version"):
                self._repository.save_model_version(
                    kind, cast(str, block["model_id"]), cast(str, block["model_version"])
                )
        audio_state = self._probe_audio(str(event["audio"]["uri"]))
        self._repository.save_turn(event, audio_state=audio_state)
        if "outcome" in event:
            self._repository.merge_outcome(
                event["call_id"], event["outcome"], source_event_id=envelope.event_id
            )
        return audio_state

    def _probe_audio(self, uri: str) -> AudioState:
        """Check object-store availability for a service-owned audio URI."""
        prefix = f"{self._object_store.scheme}://{self._object_store.bucket}/"
        if not uri.startswith(prefix):
            return "referenced"
        key = uri[len(prefix) :]
        try:
            return "stored" if self._object_store.exists(key) else "referenced"
        except Exception:
            logger.warning("ingest_audio_probe_failed", key=key)
            return "unavailable"
