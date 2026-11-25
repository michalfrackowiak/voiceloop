"""P3-07: bounded, fail-open call-path mirroring."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from services.release.shadow import (
    BoundedMirrorSink,
    Clock,
    CollectingAuditSink,
    FakeClock,
    MirroredTurn,
    MirrorReason,
    MirrorSink,
    ShadowMirror,
    ShadowSettings,
    ShadowTurn,
)


@dataclass
class CapturingSink:
    """Fake sink that records what the call path handed over."""

    turns: list[MirroredTurn]

    def try_enqueue(self, turn: MirroredTurn, deadline: datetime) -> bool:
        """Record the snapshot and accept it."""
        self.turns.append(turn)
        return True

    def dequeue(self) -> MirroredTurn | None:
        """Return nothing (mirroring tests do not run background work)."""
        return None

    def pending(self) -> int:
        """Return the captured queue depth."""
        return len(self.turns)


class ClockAdvancingSink(BoundedMirrorSink):
    """Fake sink that consumes fake time without any real sleep."""

    def __init__(self, clock: FakeClock, capacity: int, delay: float) -> None:
        """Store the injected fake clock and the simulated delay."""
        super().__init__(clock=clock, capacity=capacity)
        self._fake_clock = clock
        self._delay = delay

    def try_enqueue(self, turn: MirroredTurn, deadline: datetime) -> bool:
        """Advance fake time past the deadline and reject the turn."""
        self._fake_clock.advance(self._delay)
        return False


class RaisingSink:
    """Fake sink that fails with a PII-shaped exception body."""

    def try_enqueue(self, turn: MirroredTurn, deadline: datetime) -> bool:
        """Raise to prove the call path fails open."""
        raise RuntimeError("sink secret guest-name@example.com card 4111111111111111")

    def dequeue(self) -> MirroredTurn | None:
        """Return nothing."""
        return None

    def pending(self) -> int:
        """Report zero pending turns."""
        return 0


def _turn(call_id: str = "call-test", turn_index: int = 0) -> ShadowTurn:
    """Build one synthetic turn reference."""
    return ShadowTurn(
        call_id=call_id,
        turn_index=turn_index,
        audio_uri="fake://audio/call-test/0.wav",
        synthetic=True,
        observed_at=datetime(2026, 1, 1),
    )


def _mirror(settings: ShadowSettings, sink: MirrorSink, clock: Clock) -> ShadowMirror:
    """Build a mirror over an injectable sink."""
    return ShadowMirror(settings, sink, clock, CollectingAuditSink())


def test_capacity_bound_drops_without_raising(shadow_settings: ShadowSettings) -> None:
    clock = FakeClock()
    settings = ShadowSettings(max_pending=1, mirror_budget_ms=2.0)
    sink = BoundedMirrorSink(clock=clock, capacity=1)
    audit = CollectingAuditSink()
    mirror = ShadowMirror(settings, sink, clock, audit)

    first = mirror.mirror(_turn("call-a", 0))
    second = mirror.mirror(_turn("call-a", 1))

    assert first.accepted and first.reason is MirrorReason.QUEUED
    assert not second.accepted and second.reason is MirrorReason.FULL
    assert sink.pending() == 1
    assert mirror.stats.observed == 2
    assert mirror.stats.accepted == 1
    assert mirror.stats.dropped == 1
    assert [event.event for event in audit.events] == ["shadow.mirror", "shadow.mirror.drop"]
    assert audit.events[-1].counts["dropped"] == 1
    assert clock.now() == FakeClock().now()


def test_deadline_is_enforced_with_a_fake_clock(shadow_settings: ShadowSettings) -> None:
    clock = FakeClock()
    settings = ShadowSettings(max_pending=4, mirror_budget_ms=1.0)
    sink = ClockAdvancingSink(clock, capacity=4, delay=5.0)

    receipt = _mirror(settings, sink, clock).mirror(_turn())

    assert not receipt.accepted
    assert receipt.reason is MirrorReason.DEADLINE_EXCEEDED
    assert receipt.elapsed_seconds > 0.001
    assert sink.pending() == 0


def test_sink_failure_is_reported_not_raised(shadow_settings: ShadowSettings) -> None:
    clock = FakeClock()
    settings = ShadowSettings(max_pending=4, mirror_budget_ms=2.0)
    audit = CollectingAuditSink()
    mirror = ShadowMirror(settings, RaisingSink(), clock, audit)

    receipt = mirror.mirror(_turn())

    assert not receipt.accepted
    assert receipt.reason is MirrorReason.SINK_ERROR
    assert receipt.error_type == "RuntimeError"
    assert mirror.stats.sink_errors == 1
    assert "4111111111111111" not in str([event.to_fields() for event in audit.events])
    assert "guest-name@example.com" not in str([event.to_fields() for event in audit.events])


def test_turn_is_copied_and_not_mutated(shadow_settings: ShadowSettings) -> None:
    clock = FakeClock()
    sink = CapturingSink([])
    mirror = _mirror(shadow_settings, sink, clock)
    turn = _turn()

    receipt = mirror.mirror(turn)

    assert receipt.accepted
    assert turn == _turn()
    assert sink.turns[0] is not turn
    assert set(MirroredTurn.__dataclass_fields__) == {
        "call_id",
        "turn_index",
        "audio_uri",
        "synthetic",
        "queued_at",
        "deadline",
    }
    assert sink.turns[0].audio_uri == turn.audio_uri
    assert sink.turns[0].deadline > sink.turns[0].queued_at
    assert shadow_settings.task_id == "P3-07"
