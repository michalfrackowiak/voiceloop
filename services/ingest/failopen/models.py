"""Value types returned by the fail-open ingest buffer."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

ShedReason = Literal["none", "queue_full", "invalid_event", "internal_error"]


@dataclass(frozen=True)
class BufferedEvent:
    """One buffered event with enough provenance to retry it later."""

    event_id: str
    payload: dict[str, Any]
    enqueued_at: str
    location: str | None = None


@dataclass(frozen=True)
class EnqueueResult:
    """Fail-open receipt: accepted or shed with a machine-readable signal."""

    accepted: bool
    signal: Literal["accepted", "shed"]
    event_id: str | None
    reason: ShedReason = "none"


@dataclass(frozen=True)
class FlushResult:
    """Outcome of one bounded drain attempt against storage."""

    flushed: int
    remaining: int
    timed_out: bool
    storage_failures: int
