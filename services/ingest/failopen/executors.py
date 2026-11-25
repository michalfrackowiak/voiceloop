"""Bounded daemon-thread execution so caller timeouts never wait on workers."""

from __future__ import annotations

import queue
import threading
from collections.abc import Callable
from concurrent.futures import Future
from typing import Any, TypeVar

T = TypeVar("T")


class WriterBusy(RuntimeError):
    """Raised when all writer slots are occupied."""


class DaemonExecutor:
    """Submit work to prestarted daemon threads without joining on close.

    A timed-out future may leave its worker running. Daemon threads and a
    non-blocking ``close`` keep that abandoned work from extending the
    caller's latency budget or interpreter shutdown.
    """

    def __init__(self, max_workers: int) -> None:
        """Start ``max_workers`` idle daemon workers and a bounded task queue."""
        self._tasks: queue.Queue[tuple[Callable[[], Any], Future[Any]]] = queue.Queue(
            maxsize=max_workers
        )
        for index in range(max_workers):
            thread = threading.Thread(
                target=self._work,
                name=f"voiceloop-failopen-writer-{index}",
                daemon=True,
            )
            thread.start()

    def submit(self, fn: Callable[[], T]) -> Future[T]:
        """Queue ``fn`` immediately or raise ``WriterBusy`` if all slots fill."""
        future: Future[T] = Future()
        try:
            self._tasks.put_nowait((fn, future))
        except queue.Full as exc:
            raise WriterBusy("all fail-open writer slots are busy") from exc
        return future

    def close(self) -> None:
        """No-op: idle daemon workers are deliberately not joined."""
        return None

    def _work(self) -> None:
        """Run queued tasks forever, publishing outcomes without blocking callers."""
        while True:
            fn, future = self._tasks.get()
            try:
                if future.set_running_or_notify_cancel():
                    try:
                        future.set_result(fn())
                    except BaseException as exc:
                        future.set_exception(exc)
            finally:
                self._tasks.task_done()
