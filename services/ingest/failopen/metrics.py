"""Thread-safe counters and a health snapshot for ingest backpressure."""

from __future__ import annotations

import math
import threading
from dataclasses import asdict, dataclass
from typing import Any, Literal

from voiceloop_common import utcnow

HealthStatus = Literal["healthy", "degraded"]


@dataclass(frozen=True)
class HealthSnapshot:
    """Point-in-time view of buffer health and cumulative counters."""

    status: HealthStatus
    depth: int
    capacity: int
    degraded_threshold: int
    accepted: int
    buffered: int
    flushed: int
    dropped: int
    storage_failures: int
    observed_at: str

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable dict suitable for a health endpoint."""
        return asdict(self)


class FailOpenMetrics:
    """Counter set for the fail-open buffer; every update is locked."""

    def __init__(self, *, capacity: int, degraded_ratio: float = 0.75) -> None:
        """Create zeroed counters and derive the degraded depth threshold."""
        self._capacity = max(1, capacity)
        bounded_ratio = min(1.0, max(0.0, degraded_ratio))
        self._degraded_threshold = max(1, math.ceil(self._capacity * bounded_ratio))
        self._lock = threading.Lock()
        self._counters = {
            "accepted": 0,
            "buffered": 0,
            "flushed": 0,
            "dropped": 0,
            "storage_failures": 0,
        }

    def record_accepted(self) -> None:
        """Count one successful enqueue attempt."""
        self._increment("accepted")

    def record_buffered(self) -> None:
        """Count one event placed in the buffer."""
        self._increment("buffered")

    def record_flushed(self, count: int = 1) -> None:
        """Count events successfully written to storage."""
        self._increment("flushed", count)

    def record_dropped(self, count: int = 1) -> None:
        """Count shed events (queue full or unusable payload)."""
        self._increment("dropped", count)

    def record_storage_failure(self, count: int = 1) -> None:
        """Count failed writes that will be retried on a later flush."""
        self._increment("storage_failures", count)

    def counters(self) -> dict[str, int]:
        """Return a copy of all cumulative counters."""
        with self._lock:
            return dict(self._counters)

    def health(self, depth: int) -> HealthSnapshot:
        """Return a snapshot; depth at or above the threshold means degraded."""
        with self._lock:
            values = dict(self._counters)
        current = max(0, depth)
        status: HealthStatus = "degraded" if current >= self._degraded_threshold else "healthy"
        return HealthSnapshot(
            status=status,
            depth=current,
            capacity=self._capacity,
            degraded_threshold=self._degraded_threshold,
            accepted=values["accepted"],
            buffered=values["buffered"],
            flushed=values["flushed"],
            dropped=values["dropped"],
            storage_failures=values["storage_failures"],
            observed_at=utcnow().isoformat().replace("+00:00", "Z"),
        )

    def _increment(self, name: str, count: int = 1) -> None:
        """Increment one counter under the lock; non-positive counts are ignored."""
        if count <= 0:
            return
        with self._lock:
            self._counters[name] += count
