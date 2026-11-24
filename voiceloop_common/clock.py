"""Injectable UTC clock with fail-open behaviour for deterministic tests."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from datetime import UTC, datetime

Clock = Callable[[], datetime]


def _real_utcnow() -> datetime:
    """Return the current UTC time from the system clock."""
    return datetime.now(UTC)


_clock: Clock = _real_utcnow


def utcnow() -> datetime:
    """Return timezone-aware UTC time, falling back to the system clock on error.

    An injected clock that raises, returns a non-datetime, or returns a naive
    datetime is handled without propagating an exception: naive values are
    interpreted as UTC and any other failure falls back to the real clock.
    """
    try:
        now = _clock()
        if not isinstance(now, datetime):
            raise TypeError("injected clock did not return a datetime")
    except Exception:
        return _real_utcnow()
    if now.tzinfo is None:
        return now.replace(tzinfo=UTC)
    return now.astimezone(UTC)


def set_clock(clock: Clock | None) -> None:
    """Inject a clock factory, or restore the system clock when given None.

    Args:
        clock: Zero-argument callable returning a datetime. Non-callable
            values are ignored so a bad injection cannot break a caller.

    """
    global _clock  # noqa: PLW0603 - deliberate process-wide clock injection
    _clock = clock if callable(clock) else _real_utcnow


def frozen_clock(value: datetime) -> Clock:
    """Return a clock factory that always yields the same datetime.

    Args:
        value: Datetime to return; naive values are treated as UTC by utcnow().

    """

    def clock() -> datetime:
        """Return the fixed datetime."""
        return value

    return clock


@contextmanager
def use_clock(clock: Clock) -> Iterator[None]:
    """Temporarily inject a clock and restore the previous one on exit."""
    global _clock  # noqa: PLW0603 - deliberate process-wide clock injection
    previous = _clock
    set_clock(clock)
    try:
        yield
    finally:
        _clock = previous
