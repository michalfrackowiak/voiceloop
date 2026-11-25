"""Bounded, fail-open handoff of live turn *references* into shadow work.

This module is the only part of P3-07 designed to sit near the call path. A
caller passes an immutable :class:`ShadowTurn` (pseudonymous IDs plus an audio
URI -- never audio bytes, transcripts or guest data) to
:class:`ShadowMirror.mirror`. That method performs one in-memory bounded queue
handoff, checks a deadline from the injected clock, returns a receipt and
never calls a candidate, champion, network client or model runtime.

> [!IMPORTANT]
> **FAKE ONLY by default.** ``FakeClock``, ``FakeTurnSource``,
> ``BoundedMirrorSink`` and ``CollectingAuditSink`` are deterministic in-memory
> stand-ins. The protocols are the seams for real ingest queues, clocks and
> structured logging; none of those dependencies is imported here.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from typing import Any, Protocol

import structlog

from voiceloop_common import utcnow

from .settings import ShadowSettings

MIRROR_EVENT = "shadow.mirror"
DROP_EVENT = "shadow.mirror.drop"


class Clock(Protocol):
    """Injectable UTC clock; ``now`` must not sleep or perform I/O."""

    def now(self) -> datetime:
        """Return the current timezone-aware UTC time."""
        ...


class SystemClock:
    """Production seam using the shared fail-open UTC clock."""

    def now(self) -> datetime:
        """Return timezone-aware UTC time from ``voiceloop_common.utcnow``."""
        return utcnow()


class FakeClock:
    """Deterministic in-memory FAKE clock; ``advance`` never sleeps."""

    def __init__(self, start: datetime | None = None) -> None:
        """Initialise the clock at a fixed UTC instant."""
        self._now = start or datetime(2026, 1, 1, tzinfo=UTC)

    def now(self) -> datetime:
        """Return the current fake instant."""
        return self._now

    def advance(self, seconds: float) -> None:
        """Move the fake instant forward without any real waiting."""
        self._now += timedelta(seconds=seconds)


@dataclass(frozen=True)
class ShadowTurn:
    """Minimal, immutable observation from a live turn source.

    Only pseudonymous identifiers, an audio URI and a synthetic flag cross the
    shadow boundary. Text, parsed slots and raw audio bytes are deliberately
    absent so a misconfigured caller cannot leak them into shadow artifacts.
    """

    call_id: str
    turn_index: int
    audio_uri: str
    synthetic: bool
    observed_at: datetime


@dataclass(frozen=True)
class MirroredTurn:
    """Immutable copy of a turn reference handed to background shadow work."""

    call_id: str
    turn_index: int
    audio_uri: str
    synthetic: bool
    queued_at: datetime
    deadline: datetime


class TurnSource(Protocol):
    """Background-only seam supplying turns to be mirrored."""

    def next_turn(self) -> ShadowTurn | None:
        """Return the next turn, or None when the source is exhausted."""
        ...


class FakeTurnSource:
    """Deterministic in-memory FAKE turn source."""

    def __init__(self, turns: Sequence[ShadowTurn]) -> None:
        """Store an ordered list of synthetic turn references."""
        self._turns = tuple(turns)
        self._index = 0

    def next_turn(self) -> ShadowTurn | None:
        """Return the next stored turn or None at the end."""
        if self._index >= len(self._turns):
            return None
        turn = self._turns[self._index]
        self._index += 1
        return turn

    @property
    def emitted(self) -> int:
        """Return how many turns have been emitted."""
        return self._index


class MirrorSink(Protocol):
    """Non-blocking bounded handoff seam used by the call path."""

    def try_enqueue(self, turn: MirroredTurn, deadline: datetime) -> bool:
        """Accept the turn before the deadline, or reject it without waiting."""
        ...

    def dequeue(self) -> MirroredTurn | None:
        """Remove and return the oldest queued turn for background work."""
        ...

    def pending(self) -> int:
        """Return the current queue depth."""
        ...


class BoundedMirrorSink:
    """Deterministic FAKE bounded in-memory queue with O(1) enqueue.

    Rejection is immediate when the queue is full or the deadline has passed;
    this sink never blocks, sleeps, allocates unbounded memory or calls a
    model. Real queue adapters must preserve this non-blocking contract.
    """

    def __init__(self, clock: Clock, capacity: int) -> None:
        """Initialise the queue with an injected clock and fixed capacity."""
        self._clock = clock
        self._capacity = capacity
        self._queue: deque[MirroredTurn] = deque()

    def try_enqueue(self, turn: MirroredTurn, deadline: datetime) -> bool:
        """Accept a turn only if there is room and time remains."""
        if self._clock.now() >= deadline:
            return False
        if len(self._queue) >= self._capacity:
            return False
        self._queue.append(turn)
        return True

    def dequeue(self) -> MirroredTurn | None:
        """Pop the oldest turn, or return None when empty."""
        if not self._queue:
            return None
        return self._queue.popleft()

    def pending(self) -> int:
        """Return the number of queued turns."""
        return len(self._queue)


@dataclass(frozen=True)
class AuditEvent:
    """Allowlisted, PII-safe structured audit payload.

    Only identifiers, artifact URIs, model IDs, exception type names and
    numeric counters may appear. Transcripts, audio bytes, booking values and
    free-form exception bodies have no field here.
    """

    event: str
    task_id: str
    status: str
    observed_at: str
    call_id: str | None = None
    turn_index: int | None = None
    source: str | None = None
    artifact_uri: str | None = None
    model_id: str | None = None
    synthetic: bool = False
    error_type: str | None = None
    counts: Mapping[str, int] = field(default_factory=dict)

    def to_fields(self) -> dict[str, object]:
        """Return a flat dict of allowlisted structured-log fields."""
        return {
            "event": self.event,
            "task_id": self.task_id,
            "status": self.status,
            "observed_at": self.observed_at,
            "call_id": self.call_id,
            "turn_index": self.turn_index,
            "source": self.source,
            "artifact_uri": self.artifact_uri,
            "model_id": self.model_id,
            "synthetic": self.synthetic,
            "error_type": self.error_type,
            "counts": dict(self.counts),
        }


class AuditSink(Protocol):
    """Structured audit seam shared by the call path and background runner."""

    def record(self, event: AuditEvent) -> None:
        """Persist or emit one allowlisted audit event."""
        ...


class CollectingAuditSink:
    """Deterministic FAKE audit sink that records events in memory."""

    def __init__(self) -> None:
        """Initialise an empty event list."""
        self.events: list[AuditEvent] = []

    def record(self, event: AuditEvent) -> None:
        """Append one event to the in-memory list."""
        self.events.append(event)

    def events_named(self, name: str) -> list[AuditEvent]:
        """Return recorded events with the given name, in order."""
        return [event for event in self.events if event.event == name]


class StructlogAuditSink:
    """Adapter from :class:`AuditEvent` to the configured structlog logger."""

    def __init__(self, name: str = "release_shadow") -> None:
        """Bind a structlog logger to the given component name."""
        self._logger: Any = structlog.get_logger(name)

    def record(self, event: AuditEvent) -> None:
        """Emit the allowlisted fields as one structured event."""
        fields = event.to_fields()
        name = fields.pop("event")
        self._logger.info(name, **fields)


class MirrorReason(StrEnum):
    """Stable reason codes for a mirror receipt."""

    QUEUED = "queued"
    FULL = "full"
    DEADLINE_EXCEEDED = "deadline_exceeded"
    REJECTED = "rejected"
    SINK_ERROR = "sink_error"


@dataclass(frozen=True)
class MirrorReceipt:
    """Synchronous result of the bounded call-path handoff."""

    accepted: bool
    reason: MirrorReason
    observed_at: datetime
    deadline: datetime
    completed_at: datetime
    error_type: str | None = None

    @property
    def elapsed_seconds(self) -> float:
        """Return clock time consumed by the handoff (zero with a frozen clock)."""
        return max(0.0, (self.completed_at - self.observed_at).total_seconds())


@dataclass(frozen=True)
class MirrorStats:
    """Cumulative, numeric counters for call-path isolation evidence."""

    observed: int = 0
    accepted: int = 0
    dropped: int = 0
    sink_errors: int = 0
    max_elapsed_seconds: float = 0.0

    def to_counts(self) -> dict[str, int]:
        """Return integer counters for structured audit and reports."""
        return {
            "observed": self.observed,
            "accepted": self.accepted,
            "dropped": self.dropped,
            "sink_errors": self.sink_errors,
        }


class ShadowMirror:
    """Fail-open handoff that never awaits or invokes model work."""

    def __init__(
        self,
        settings: ShadowSettings,
        sink: MirrorSink,
        clock: Clock,
        audit: AuditSink,
    ) -> None:
        """Inject settings, bounded sink, clock and audit sink."""
        self._settings = settings
        self._sink = sink
        self._clock = clock
        self._audit = audit
        self._stats = MirrorStats()

    @property
    def stats(self) -> MirrorStats:
        """Return a snapshot of cumulative mirror counters."""
        return self._stats

    def mirror(self, turn: ShadowTurn) -> MirrorReceipt:
        """Enqueue a turn reference within the deadline; never raise or block.

        Args:
            turn: Immutable observation supplied by the caller. It is copied
                into :class:`MirroredTurn` and is never mutated.

        Returns:
            A receipt with acceptance, reason, deadline and elapsed time. All
            expected failures are reported as drops; nothing reaches the
            caller as an exception.

        """
        observed = self._safe_now()
        deadline = observed + self._settings.mirror_budget()
        self._stats = replace(self._stats, observed=self._stats.observed + 1)

        error_type: str | None = None
        accepted = False
        reason = MirrorReason.DEADLINE_EXCEEDED
        try:
            snapshot = MirroredTurn(
                call_id=turn.call_id,
                turn_index=turn.turn_index,
                audio_uri=turn.audio_uri,
                synthetic=turn.synthetic,
                queued_at=observed,
                deadline=deadline,
            )
            accepted = self._sink.try_enqueue(snapshot, deadline)
            completed = self._safe_now()
            if accepted:
                reason = MirrorReason.QUEUED
            elif completed >= deadline:
                # A sink that used the whole budget reports a deadline drop;
                # nothing was queued, so no deferred work was created.
                reason = MirrorReason.DEADLINE_EXCEEDED
            else:
                reason = (
                    MirrorReason.FULL
                    if self._safe_pending() >= self._settings.max_pending
                    else MirrorReason.REJECTED
                )
        except Exception as exc:
            completed = self._safe_now()
            accepted = False
            reason = MirrorReason.SINK_ERROR
            error_type = type(exc).__name__
            self._stats = replace(self._stats, sink_errors=self._stats.sink_errors + 1)

        elapsed = max(0.0, (completed - observed).total_seconds())
        self._stats = replace(
            self._stats,
            accepted=self._stats.accepted + int(accepted),
            dropped=self._stats.dropped + int(not accepted),
            max_elapsed_seconds=max(self._stats.max_elapsed_seconds, elapsed),
        )
        receipt = MirrorReceipt(
            accepted=accepted,
            reason=reason,
            observed_at=observed,
            deadline=deadline,
            completed_at=completed,
            error_type=error_type,
        )
        self._audit_safely(receipt, turn)
        return receipt

    def _safe_now(self) -> datetime:
        """Read the clock without propagating an injected-clock failure."""
        try:
            value = self._clock.now()
            if not isinstance(value, datetime):
                raise TypeError("clock did not return a datetime")
            return value
        except Exception:
            return datetime.now(UTC)

    def _safe_pending(self) -> int:
        """Read sink depth without propagating a sink failure."""
        try:
            return self._sink.pending()
        except Exception:
            return self._settings.max_pending

    def _audit_safely(self, receipt: MirrorReceipt, turn: ShadowTurn) -> None:
        """Emit one bounded audit event; audit failures never reach the caller."""
        event_name = MIRROR_EVENT if receipt.accepted else DROP_EVENT
        event = AuditEvent(
            event=event_name,
            task_id=self._settings.task_id,
            status="ok" if receipt.accepted else "dropped",
            observed_at=receipt.observed_at.isoformat().replace("+00:00", "Z"),
            call_id=turn.call_id,
            turn_index=turn.turn_index,
            source="mirror",
            synthetic=turn.synthetic,
            error_type=receipt.error_type,
            counts=self._stats.to_counts(),
        )
        try:
            self._audit.record(event)
        except Exception:
            return
