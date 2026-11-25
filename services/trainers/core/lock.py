"""Single-job locking for training runs, with a documented multi-process seam.

The runner uses :class:`SingleJobLock` by default: a non-blocking in-process
thread lock that rejects a second job instead of queueing it. Multi-process
deployments must supply an implementation of :class:`FileJobLock` (an OS-level
file lock held for the lifetime of the context). No file or distributed lock is
implemented here, so the core package stays dependency- and platform-agnostic.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import AbstractContextManager, contextmanager
from pathlib import Path
from threading import Lock
from typing import Protocol

from voiceloop_common import VoiceLoopError


class JobBusyError(VoiceLoopError):
    """Raised when a training job is already running and no queueing is allowed."""


class JobLock(Protocol):
    """Context-manager lock accepted by :class:`~.runner.TrainerRunner`."""

    def held(self) -> AbstractContextManager[None]:
        """Acquire the lock, raising ``JobBusyError`` on contention, then release it."""
        ...


class FileJobLock(JobLock, Protocol):
    """Interface for a cross-process file lock (implementation belongs to infra).

    An implementation must open a well-known lock file at ``path`` and hold an
    *OS-level* exclusive lock (``fcntl.flock`` on POSIX, ``msvcrt.locking`` on
    Windows) for the duration of ``held()``. It must raise :class:`JobBusyError`
    immediately when another process holds the lock and release the OS lock in a
    ``finally`` block so a failed training job cannot wedge future runs. The
    lock file must not be deleted while held. Production orchestration should
    inject a real implementation; the in-process default is not cross-process
    safe.
    """

    @property
    def path(self) -> Path:
        """Return the lock file path shared by all competing processes."""
        ...


class SingleJobLock:
    """Non-blocking in-process lock: at most one training job per interpreter."""

    def __init__(self) -> None:
        """Create an unlocked in-process lock."""
        self._lock = Lock()

    @contextmanager
    def held(self) -> Iterator[None]:
        """Hold the lock, rejecting a concurrent job immediately.

        Yields:
            ``None`` while the lock is held.

        Raises:
            JobBusyError: If another thread already holds the lock.

        """
        if not self._lock.acquire(blocking=False):
            raise JobBusyError("a training job is already running")
        try:
            yield
        finally:
            self._lock.release()
