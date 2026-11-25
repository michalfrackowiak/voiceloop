"""Bounded FIFO stores used by the fail-open ingest buffer."""

from __future__ import annotations

import json
import os
import queue
import shutil
import tempfile
import threading
import time
from concurrent.futures import Future
from concurrent.futures import TimeoutError as FutureTimeout
from pathlib import Path
from typing import Any, Protocol, cast

from voiceloop_common import new_id

from .executors import DaemonExecutor, WriterBusy
from .models import BufferedEvent
from .payload import PayloadError


class StoreTimeout(TimeoutError):
    """Raised when a store operation exceeds its caller-supplied budget."""


class _WriteCancelled(Exception):
    """Raised internally when an abandoned disk write finishes."""


class EventStore(Protocol):
    """Bounded FIFO store whose popped entries must be acknowledged."""

    def put(self, event: BufferedEvent, timeout: float) -> None:
        """Append an event, waiting at most ``timeout`` seconds for room."""
        ...

    def pop(self) -> BufferedEvent | None:
        """Remove and return the next event, or None when empty."""
        ...

    def ack(self, event: BufferedEvent) -> None:
        """Mark a previously popped event as processed."""
        ...

    def restore(self, event: BufferedEvent) -> None:
        """Return a popped event to the queue without a new write."""
        ...

    def __len__(self) -> int:
        """Return the number of buffered events."""
        ...

    def close(self) -> None:
        """Release any resources owned by the store."""
        ...


class MemoryStore:
    """In-process bounded FIFO backed by ``queue.Queue``."""

    def __init__(self, capacity: int) -> None:
        """Create an empty queue with the given maximum size."""
        self._queue: queue.Queue[BufferedEvent] = queue.Queue(maxsize=capacity)

    def put(self, event: BufferedEvent, timeout: float) -> None:
        """Append an event, waiting at most ``timeout`` seconds for room."""
        self._queue.put(event, timeout=max(0.0, timeout))

    def pop(self) -> BufferedEvent | None:
        """Remove and return the next event, or None when empty."""
        try:
            return self._queue.get_nowait()
        except queue.Empty:
            return None

    def ack(self, event: BufferedEvent) -> None:
        """Mark a popped event as processed."""
        self._queue.task_done()

    def restore(self, event: BufferedEvent) -> None:
        """Return a popped event to the queue without copying it."""
        self._queue.put_nowait(event)

    def __len__(self) -> int:
        """Return the number of buffered events."""
        return self._queue.qsize()

    def close(self) -> None:
        """No-op for the in-memory store."""
        return None


class DiskStore:
    """Disk-backed bounded FIFO with non-blocking, cancellable writes."""

    def __init__(
        self,
        capacity: int,
        directory: Path | None = None,
        *,
        writer_workers: int = 2,
        max_bytes: int = 64 * 1024 + 4096,
    ) -> None:
        """Create a bounded queue and a private temporary spill directory."""
        parent = directory or Path(tempfile.gettempdir())
        parent.mkdir(parents=True, exist_ok=True)
        self._root = Path(tempfile.mkdtemp(prefix="voiceloop-failopen-", dir=str(parent)))
        self._queue: queue.Queue[Path] = queue.Queue(maxsize=capacity)
        self._slots = threading.BoundedSemaphore(capacity)
        self._writer = DaemonExecutor(writer_workers)
        self._max_bytes = max_bytes
        self._closed = threading.Event()
        self._guard = threading.Lock()
        self._released: set[str] = set()

    def put(self, event: BufferedEvent, timeout: float) -> None:
        """Serialise and spill an event without exceeding ``timeout`` seconds."""
        if self._closed.is_set():
            raise StoreTimeout("disk store is closed")
        deadline = time.monotonic() + max(0.0, timeout)
        if not self._slots.acquire(timeout=max(0.0, deadline - time.monotonic())):
            raise queue.Full

        document = {
            "event_id": event.event_id,
            "payload": event.payload,
            "enqueued_at": event.enqueued_at,
        }
        document_text = json.dumps(document, sort_keys=True)
        if len(document_text.encode("utf-8")) > self._max_bytes:
            self._slots.release()
            raise PayloadError("serialised event exceeds disk budget")

        final = self._root / f"{new_id()}.json"
        temp = self._root / f".{final.name}.tmp"
        cancel = threading.Event()
        try:
            future: Future[Path] = self._writer.submit(
                lambda: self._write_file(document_text, temp, final, cancel)
            )
        except WriterBusy as exc:
            self._slots.release()
            raise StoreTimeout("no disk writer slot available") from exc

        try:
            future.result(timeout=max(0.0, deadline - time.monotonic()))
        except FutureTimeout as exc:
            cancel.set()
            temp.unlink(missing_ok=True)
            final.unlink(missing_ok=True)
            self._slots.release()
            raise StoreTimeout("disk write exceeded enqueue timeout") from exc
        except BaseException:
            cancel.set()
            temp.unlink(missing_ok=True)
            final.unlink(missing_ok=True)
            self._slots.release()
            raise

        if self._closed.is_set():
            cancel.set()
            final.unlink(missing_ok=True)
            self._slots.release()
            raise StoreTimeout("disk store closed during write")
        try:
            self._queue.put_nowait(final)
        except queue.Full:
            cancel.set()
            final.unlink(missing_ok=True)
            self._slots.release()
            raise

    def pop(self) -> BufferedEvent | None:
        """Read the next event from disk, skipping unreadable spill files."""
        while True:
            try:
                path = self._queue.get_nowait()
            except queue.Empty:
                return None
            try:
                document = json.loads(path.read_text(encoding="utf-8"))
                return BufferedEvent(
                    event_id=str(document["event_id"]),
                    payload=cast(dict[str, Any], document["payload"]),
                    enqueued_at=str(document["enqueued_at"]),
                    location=str(path),
                )
            except Exception:
                path.unlink(missing_ok=True)
                self._release_slot(None)

    def ack(self, event: BufferedEvent) -> None:
        """Delete the spill file and release the capacity slot exactly once."""
        location = event.location
        if location:
            Path(location).unlink(missing_ok=True)
        self._release_slot(location)

    def restore(self, event: BufferedEvent) -> None:
        """Requeue the original spill path, keeping its capacity slot."""
        if not event.location:
            self.put(
                BufferedEvent(
                    event_id=event.event_id,
                    payload=event.payload,
                    enqueued_at=event.enqueued_at,
                ),
                0.0,
            )
            return
        path = Path(event.location)
        if not path.exists():
            raise FileNotFoundError(event.location)
        self._queue.put_nowait(path)

    def __len__(self) -> int:
        """Return the number of buffered events."""
        return self._queue.qsize()

    def close(self) -> None:
        """Cancel pending writes and remove the private spill directory."""
        self._closed.set()
        shutil.rmtree(self._root, ignore_errors=True)
        with self._guard:
            self._released.clear()

    def _write_file(
        self,
        text: str,
        temp: Path,
        final: Path,
        cancel: threading.Event,
    ) -> Path:
        """Write atomically, cleaning up if the caller abandons the write."""
        try:
            if cancel.is_set() or self._closed.is_set():
                raise _WriteCancelled
            temp.write_text(text, encoding="utf-8")
            if cancel.is_set() or self._closed.is_set():
                raise _WriteCancelled
            os.replace(temp, final)
            if cancel.is_set() or self._closed.is_set():
                raise _WriteCancelled
            return final
        except BaseException:
            temp.unlink(missing_ok=True)
            final.unlink(missing_ok=True)
            raise

    def _release_slot(self, location: str | None) -> None:
        """Release one capacity slot, ignoring duplicate acknowledgements."""
        if location:
            with self._guard:
                if location in self._released:
                    return
                self._released.add(location)
        self._slots.release()
