"""Fail-open behaviour tests for the SDK client."""

import time
from collections.abc import Callable, Mapping
from typing import Any, cast

from sdk import (
    BreakerState,
    EnqueueStatus,
    InMemoryTransport,
    SdkClient,
    SdkSettings,
    log_turn,
)

EventFactory = Callable[..., dict[str, Any]]


def test_unreachable_transport_adds_no_noticeable_caller_latency(make_event: EventFactory) -> None:
    transport = InMemoryTransport(delay_seconds=0.1)
    settings = SdkSettings(queue_capacity=1024, max_retries=0, timeout_seconds=0.05)
    with SdkClient(settings=settings, transport=transport) as client:
        client.enqueue(make_event(0))
        assert client.wait_until_idle(timeout_seconds=2.0)

        worst = 0.0
        total_started = time.perf_counter()
        statuses: list[EnqueueStatus] = []
        for number in range(1, 21):
            started = time.perf_counter()
            status = log_turn(make_event(number), client=client)
            worst = max(worst, time.perf_counter() - started)
            statuses.append(status)
        caller_elapsed = time.perf_counter() - total_started

        assert statuses == [EnqueueStatus.ACCEPTED] * 20
        assert worst < 0.05
        assert caller_elapsed / 20 < 0.005


def test_invalid_event_is_validated_in_background_and_counted(make_event: EventFactory) -> None:
    transport = InMemoryTransport()
    with SdkClient(transport=transport) as client:
        invalid = make_event(1)
        del invalid["stt"]

        assert log_turn(invalid, client=client) is EnqueueStatus.ACCEPTED
        assert client.wait_until_idle(timeout_seconds=2.0)

        snapshot = client.metrics.snapshot()
        assert snapshot.validation_failures == 1
        assert snapshot.dropped_invalid == 1
        assert snapshot.sent == 0
        assert transport.call_count == 0


def test_circuit_breaker_opens_and_sheds_later_sends(make_event: EventFactory) -> None:
    transport = InMemoryTransport(failures_remaining=100)
    settings = SdkSettings(circuit_breaker_failure_threshold=2, max_retries=0)
    with SdkClient(settings=settings, transport=transport) as client:
        for number in range(1, 4):
            assert client.enqueue(make_event(number)) is EnqueueStatus.ACCEPTED
        assert client.wait_until_idle(timeout_seconds=2.0)

        snapshot = client.metrics.snapshot()
        assert transport.call_count == 2
        assert snapshot.shed_by_breaker == 1
        assert snapshot.send_failures == 2
        assert client.breaker.state is BreakerState.OPEN


def test_retry_queue_flushes_when_transport_recovers(make_event: EventFactory) -> None:
    transport = InMemoryTransport(failures_remaining=1)
    settings = SdkSettings(max_retries=2, retry_backoff_seconds=0.0)
    with SdkClient(settings=settings, transport=transport) as client:
        assert client.enqueue(make_event(1)) is EnqueueStatus.ACCEPTED
        assert client.wait_until_idle(timeout_seconds=2.0)

        snapshot = client.metrics.snapshot()
        assert transport.call_count == 2
        assert len(transport.events) == 1
        assert snapshot.retried == 1
        assert snapshot.sent == 1
        assert snapshot.dropped_after_retries == 0


def test_bounded_queue_drops_oldest_and_counts_it(make_event: EventFactory) -> None:
    client = SdkClient(settings=SdkSettings(queue_capacity=2), start_worker=False)
    try:
        first = make_event(1)
        second = make_event(2)
        third = make_event(3)

        assert client.enqueue(first) is EnqueueStatus.ACCEPTED
        assert client.enqueue(second) is EnqueueStatus.ACCEPTED
        assert client.enqueue(third) is EnqueueStatus.ACCEPTED

        snapshot = client.metrics.snapshot()
        assert snapshot.dropped_oldest == 1
        assert snapshot.accepted == 3
        assert client.pending_event_ids() == [second["event_id"], third["event_id"]]
    finally:
        client.close()


def test_non_mapping_event_returns_error_without_raising() -> None:
    client = SdkClient(start_worker=False)
    try:
        bad_event = cast(Mapping[str, Any], object())
        assert log_turn(bad_event, client=client) is EnqueueStatus.ERROR
        assert client.metrics.snapshot().enqueue_errors == 1
    finally:
        client.close()
