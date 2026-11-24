"""Tests for the SDK circuit breaker."""

from sdk import BreakerState, CircuitBreaker


def test_breaker_opens_after_threshold_and_allows_one_probe() -> None:
    now = [100.0]
    breaker = CircuitBreaker(
        failure_threshold=2,
        recovery_seconds=5.0,
        clock=lambda: now[0],
    )

    assert breaker.allow() is True
    breaker.record_failure()
    assert breaker.state is BreakerState.CLOSED

    breaker.record_failure()
    assert breaker.state is BreakerState.OPEN
    assert breaker.allow() is False

    now[0] += 5.0
    assert breaker.allow() is True
    assert breaker.state is BreakerState.HALF_OPEN
    assert breaker.allow() is False

    breaker.record_success()
    assert breaker.state is BreakerState.CLOSED
    assert breaker.allow() is True


def test_breaker_reopens_when_probe_fails() -> None:
    now = [0.0]
    breaker = CircuitBreaker(
        failure_threshold=1,
        recovery_seconds=1.0,
        clock=lambda: now[0],
    )

    breaker.record_failure()
    now[0] += 1.0
    assert breaker.allow() is True
    breaker.record_failure()

    assert breaker.state is BreakerState.OPEN
    assert breaker.allow() is False
