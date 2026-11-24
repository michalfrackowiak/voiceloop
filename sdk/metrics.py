"""Thread-safe counters for the fail-open SDK."""

from __future__ import annotations

from dataclasses import dataclass
from threading import Lock
from typing import Literal

CounterName = Literal[
    "accepted",
    "enqueue_errors",
    "dropped_oldest",
    "shed_by_breaker",
    "sent",
    "send_failures",
    "validation_failures",
    "dropped_invalid",
    "retried",
    "dropped_after_retries",
    "internal_errors",
]

_COUNTER_NAMES: tuple[CounterName, ...] = (
    "accepted",
    "enqueue_errors",
    "dropped_oldest",
    "shed_by_breaker",
    "sent",
    "send_failures",
    "validation_failures",
    "dropped_invalid",
    "retried",
    "dropped_after_retries",
    "internal_errors",
)


@dataclass(frozen=True)
class SdkMetricsSnapshot:
    """Immutable copy of the SDK counters at one instant."""

    accepted: int = 0
    enqueue_errors: int = 0
    dropped_oldest: int = 0
    shed_by_breaker: int = 0
    sent: int = 0
    send_failures: int = 0
    validation_failures: int = 0
    dropped_invalid: int = 0
    retried: int = 0
    dropped_after_retries: int = 0
    internal_errors: int = 0


class SdkMetrics:
    """Collect SDK counters without raising into the caller."""

    def __init__(self) -> None:
        """Start every counter at zero."""
        self._lock = Lock()
        self._values: dict[CounterName, int] = {name: 0 for name in _COUNTER_NAMES}

    def increment(self, name: CounterName) -> None:
        """Increment one named counter.

        Args:
            name: Counter to increment; unknown names are ignored.

        """
        with self._lock:
            self._values[name] += 1

    def get(self, name: CounterName) -> int:
        """Return the current value of one counter."""
        with self._lock:
            return self._values[name]

    def snapshot(self) -> SdkMetricsSnapshot:
        """Return a consistent immutable view of all counters."""
        with self._lock:
            values = dict(self._values)
        return SdkMetricsSnapshot(
            accepted=values["accepted"],
            enqueue_errors=values["enqueue_errors"],
            dropped_oldest=values["dropped_oldest"],
            shed_by_breaker=values["shed_by_breaker"],
            sent=values["sent"],
            send_failures=values["send_failures"],
            validation_failures=values["validation_failures"],
            dropped_invalid=values["dropped_invalid"],
            retried=values["retried"],
            dropped_after_retries=values["dropped_after_retries"],
            internal_errors=values["internal_errors"],
        )
