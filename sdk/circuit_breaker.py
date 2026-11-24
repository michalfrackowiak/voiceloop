"""Small circuit breaker used by the SDK background worker."""

from __future__ import annotations

import time
from collections.abc import Callable
from enum import StrEnum
from threading import Lock

Clock = Callable[[], float]


class BreakerState(StrEnum):
    """States of the SDK send circuit breaker."""

    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitBreaker:
    """Shed sends after a run of failures, then allow one recovery probe."""

    def __init__(
        self,
        *,
        failure_threshold: int,
        recovery_seconds: float,
        clock: Clock = time.monotonic,
    ) -> None:
        """Create a breaker.

        Args:
            failure_threshold: Consecutive failures before the breaker opens.
            recovery_seconds: Time before a single probe is allowed through.
            clock: Injectable monotonic clock; defaults to ``time.monotonic``.

        """
        self._threshold = max(1, failure_threshold)
        self._recovery_seconds = max(0.0, recovery_seconds)
        self._clock = clock
        self._lock = Lock()
        self._state = BreakerState.CLOSED
        self._failures = 0
        self._opened_at = 0.0

    def allow(self) -> bool:
        """Return whether one send may proceed.

        Once the recovery window elapses the breaker becomes half-open and
        allows a single probe; further probes wait for that probe's result.
        """
        with self._lock:
            if self._state is BreakerState.CLOSED:
                return True
            if self._state is BreakerState.HALF_OPEN:
                return False
            if self._now() - self._opened_at < self._recovery_seconds:
                return False
            self._state = BreakerState.HALF_OPEN
            return True

    def record_success(self) -> None:
        """Record a successful send and reset the breaker to closed."""
        with self._lock:
            self._failures = 0
            self._state = BreakerState.CLOSED

    def record_failure(self) -> None:
        """Record a failed send, opening the breaker at the threshold."""
        with self._lock:
            self._failures += 1
            if self._state is BreakerState.HALF_OPEN or self._failures >= self._threshold:
                self._state = BreakerState.OPEN
                self._opened_at = self._now()

    @property
    def state(self) -> BreakerState:
        """Return the current breaker state."""
        with self._lock:
            return self._state

    def _now(self) -> float:
        """Read the clock, falling back to the system clock on injection errors."""
        try:
            value = float(self._clock())
        except Exception:
            return time.monotonic()
        return value
