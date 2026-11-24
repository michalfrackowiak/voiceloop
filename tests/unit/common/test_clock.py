"""Tests for the injectable, fail-open UTC clock."""

from datetime import UTC, datetime, timedelta, timezone
from typing import Any, cast

from voiceloop_common import clock


def test_utcnow_returns_timezone_aware_utc() -> None:
    before = datetime.now(UTC)

    now = clock.utcnow()

    after = datetime.now(UTC)
    assert before <= now <= after
    assert now.tzinfo is not None
    assert now.utcoffset() == timedelta(0)


def test_frozen_clock_is_exact_and_context_manager_restores() -> None:
    fixed = datetime(2026, 1, 1, 12, 0, tzinfo=UTC)

    with clock.use_clock(clock.frozen_clock(fixed)):
        assert clock.utcnow() == fixed

    assert clock.utcnow() != fixed


def test_naive_injected_datetime_is_treated_as_utc() -> None:
    naive = datetime(2026, 1, 1, 12, 0)

    with clock.use_clock(lambda: naive):
        now = clock.utcnow()

    assert now == naive.replace(tzinfo=UTC)


def test_offset_injected_datetime_is_converted_to_utc() -> None:
    offset = timezone(timedelta(hours=2))
    value = datetime(2026, 1, 1, 14, 0, tzinfo=offset)

    with clock.use_clock(lambda: value):
        now = clock.utcnow()

    assert now.hour == 12
    assert now.tzinfo == UTC


def test_raising_clock_fails_open_to_system_clock() -> None:
    def broken_clock() -> datetime:
        raise RuntimeError("clock unavailable")

    with clock.use_clock(broken_clock):
        before = datetime.now(UTC)
        now = clock.utcnow()
        after = datetime.now(UTC)

    assert before <= now <= after


def test_wrong_type_clock_fails_open_to_system_clock() -> None:
    def wrong_clock() -> Any:
        return "not a datetime"

    with clock.use_clock(cast(clock.Clock, wrong_clock)):
        before = datetime.now(UTC)
        now = clock.utcnow()
        after = datetime.now(UTC)

    assert before <= now <= after


def test_non_callable_injection_is_ignored() -> None:
    before = datetime.now(UTC)

    clock.set_clock(cast(clock.Clock, "not callable"))
    now = clock.utcnow()

    after = datetime.now(UTC)
    assert before <= now <= after
