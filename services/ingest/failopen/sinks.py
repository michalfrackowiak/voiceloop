"""Storage sink contract and an in-memory fake for outage tests."""

from __future__ import annotations

import threading
from collections.abc import Mapping
from copy import deepcopy
from typing import Any, Protocol

from voiceloop_common.errors import StorageError, TransientError


class StorageSink(Protocol):
    """Persist one buffered event; raise on transient storage failure."""

    def write(self, event: Mapping[str, Any]) -> None:
        """Persist ``event`` or raise ``StorageError``/``TransientError``."""
        ...


class RecordingSink:
    """Thread-safe in-memory sink that can simulate a storage outage."""

    def __init__(
        self,
        *,
        down: bool = False,
        error: type[StorageError | TransientError] = TransientError,
    ) -> None:
        """Create an empty sink, optionally starting in an outage state."""
        self._lock = threading.Lock()
        self._down = down
        self._error = error
        self._written: list[dict[str, Any]] = []

    def set_down(
        self,
        down: bool,
        *,
        error: type[StorageError | TransientError] | None = None,
    ) -> None:
        """Enable or disable the outage and optionally change its error type."""
        with self._lock:
            self._down = down
            if error is not None:
                self._error = error

    @property
    def written(self) -> list[dict[str, Any]]:
        """Return copies of every event successfully persisted so far."""
        with self._lock:
            return [deepcopy(event) for event in self._written]

    def write(self, event: Mapping[str, Any]) -> None:
        """Record the event, or raise the configured error while down."""
        with self._lock:
            if self._down:
                raise self._error("simulated storage outage")
            self._written.append(deepcopy(dict(event)))
