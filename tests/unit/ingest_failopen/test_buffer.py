"""Bounded-buffer tests: outage buffering, load shedding and bounded waits."""

from __future__ import annotations

import time
from collections.abc import Mapping
from typing import Any, cast

import pytest

from services.ingest.failopen import BoundedBuffer, FailOpenSettings, RecordingSink
from voiceloop_common.errors import StorageError, TransientError


def _event(index: int) -> dict[str, Any]:
    """Return a small event with a stable id."""
    return {
        "event_id": f"evt-{index}",
        "call_id": f"call-{index}",
        "turn_index": index,
        "audio": {"uri": "s3://bucket/turn.wav"},
        "stt": {"text": "hello"},
    }


def test_outage_events_are_buffered_then_flushed_after_recovery() -> None:
    sink = RecordingSink(down=True, error=TransientError)
    buffer = BoundedBuffer(sink, FailOpenSettings(capacity=8))

    results = [buffer.enqueue(_event(i)) for i in range(3)]

    assert all(result.accepted and result.signal == "accepted" for result in results)
    assert buffer.depth == 3
    failed = buffer.flush()
    assert failed.flushed == 0
    assert failed.storage_failures == 3
    assert failed.remaining == 3
    assert sink.written == []
    assert buffer.counters() == {
        "accepted": 3,
        "buffered": 3,
        "flushed": 0,
        "dropped": 0,
        "storage_failures": 3,
    }

    sink.set_down(False)
    recovered = buffer.flush()

    assert recovered.flushed == 3
    assert recovered.storage_failures == 0
    assert recovered.remaining == 0
    assert not recovered.timed_out
    assert [event["event_id"] for event in sink.written] == ["evt-0", "evt-1", "evt-2"]
    assert buffer.depth == 0
    assert buffer.counters()["flushed"] == 3


@pytest.mark.parametrize("error", [TransientError, StorageError])
def test_storage_and_transient_errors_are_isolated(
    error: type[StorageError | TransientError],
) -> None:
    sink = RecordingSink(down=True, error=error)
    buffer = BoundedBuffer(sink, FailOpenSettings(capacity=4))

    assert buffer.enqueue(_event(0)).accepted
    result = buffer.flush()

    assert result.flushed == 0
    assert result.storage_failures == 1
    assert buffer.depth == 1
    sink.set_down(False)
    assert buffer.flush().flushed == 1


def test_full_queue_sheds_with_explicit_signal_and_health_degrades() -> None:
    sink = RecordingSink(down=True)
    buffer = BoundedBuffer(
        sink,
        FailOpenSettings(capacity=2, degraded_ratio=0.75),
    )

    assert buffer.enqueue(_event(0)).accepted
    assert buffer.enqueue(_event(1)).accepted
    shed = buffer.enqueue(_event(2))

    assert not shed.accepted
    assert shed.signal == "shed"
    assert shed.reason == "queue_full"
    assert shed.event_id == "evt-2"
    assert buffer.counters()["dropped"] == 1
    assert buffer.depth == 2
    snapshot = buffer.health()
    assert snapshot.status == "degraded"
    assert snapshot.depth == 2
    assert snapshot.degraded_threshold == 2
    assert snapshot.dropped == 1

    sink.set_down(False)
    assert buffer.flush().flushed == 2
    assert buffer.health().status == "healthy"


def test_enqueue_never_blocks_beyond_the_configured_timeout() -> None:
    sink = RecordingSink(down=True)
    buffer = BoundedBuffer(sink, FailOpenSettings(capacity=1, enqueue_timeout_ms=10))
    assert buffer.enqueue(_event(0)).accepted

    started = time.perf_counter()
    first = buffer.enqueue(_event(1))
    elapsed_default = time.perf_counter() - started

    assert not first.accepted and first.reason == "queue_full"
    assert elapsed_default < 0.2

    started = time.perf_counter()
    second = buffer.enqueue(_event(2), timeout_ms=25)
    elapsed_override = time.perf_counter() - started

    assert not second.accepted and second.reason == "queue_full"
    assert 0.01 <= elapsed_override < 0.3


def test_invalid_payload_and_unexpected_sink_error_do_not_raise() -> None:
    buffer = BoundedBuffer(RecordingSink(), FailOpenSettings(capacity=2))

    invalid = buffer.enqueue(cast(Any, None))

    assert not invalid.accepted
    assert invalid.reason == "invalid_event"

    class ExplodingSink:
        """Sink raising an error outside the storage hierarchy."""

        def write(self, event: Mapping[str, Any]) -> None:
            """Always fail."""
            raise RuntimeError("unexpected")

    broken = BoundedBuffer(ExplodingSink(), FailOpenSettings(capacity=2))
    assert broken.enqueue(_event(0)).accepted
    result = broken.flush()

    assert result.flushed == 0
    assert result.storage_failures == 1
    assert broken.depth == 1
