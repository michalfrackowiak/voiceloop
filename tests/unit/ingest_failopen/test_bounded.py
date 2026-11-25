"""Regression tests for end-to-end enqueue and flush deadlines."""

from __future__ import annotations

import threading
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal, cast

import pytest

from services.ingest.failopen import BoundedBuffer, FailOpenSettings, RecordingSink


def _event(index: int) -> dict[str, Any]:
    """Return a small event with a stable id."""
    return {"event_id": f"bound-{index}", "call_id": f"call-{index}", "stt": {"text": ""}}


def test_disk_enqueue_bounds_slow_write_text(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A blocked filesystem must not extend the caller's enqueue budget."""
    original_write_text = Path.write_text

    def slow_write_text(
        self: Path,
        data: str,
        *args: Any,
        **kwargs: Any,
    ) -> int:
        """Sleep past the enqueue deadline before performing the real write."""
        time.sleep(0.2)
        return original_write_text(self, data, *args, **kwargs)

    monkeypatch.setattr(Path, "write_text", slow_write_text)
    settings = FailOpenSettings(
        backend="disk",
        capacity=2,
        disk_dir=tmp_path,
        enqueue_timeout_ms=5,
        writer_workers=1,
    )

    with BoundedBuffer(RecordingSink(), settings) as buffer:
        started = time.perf_counter()
        result = buffer.enqueue(_event(0))
        elapsed = time.perf_counter() - started

        assert not result.accepted
        assert result.signal == "shed"
        assert result.reason in {"queue_full", "internal_error"}
        assert result.event_id == "bound-0"
        assert elapsed < 0.05
        assert buffer.depth == 0
        assert buffer.counters() == {
            "accepted": 0,
            "buffered": 0,
            "flushed": 0,
            "dropped": 1,
            "storage_failures": 0,
        }


@pytest.mark.parametrize("backend", ["memory", "disk"])
def test_hostile_payloads_are_rejected_with_honest_dropped_count(
    tmp_path: Path,
    backend: str,
) -> None:
    """Huge strings and cycles shed promptly instead of copying forever."""
    settings = FailOpenSettings(
        backend=cast(Literal["memory", "disk"], backend),
        capacity=4,
        disk_dir=tmp_path,
        max_payload_bytes=128,
        enqueue_timeout_ms=50,
    )
    cyclic: dict[str, Any] = {"event_id": "cyclic"}
    cyclic["self"] = cyclic

    with BoundedBuffer(RecordingSink(), settings) as buffer:
        huge = buffer.enqueue({"event_id": "huge", "blob": "x" * 4096})
        cycle = buffer.enqueue(cyclic)

        assert huge.reason == "invalid_event"
        assert cycle.reason == "invalid_event"
        assert not huge.accepted and not cycle.accepted
        assert buffer.counters()["accepted"] == 0
        assert buffer.counters()["buffered"] == 0
        assert buffer.counters()["dropped"] == 2


def test_flush_times_out_against_blocking_sink_and_requeues() -> None:
    """A blocked sink is abandoned at the deadline and retried next flush."""
    gate = threading.Event()

    class GatedSink:
        """Sink that blocks until the test releases its gate."""

        def __init__(self) -> None:
            """Create an empty write log."""
            self._lock = threading.Lock()
            self.written: list[dict[str, Any]] = []

        def write(self, event: Mapping[str, Any]) -> None:
            """Block on the gate, then record the event."""
            gate.wait(1.0)
            with self._lock:
                self.written.append(dict(event))

    sink = GatedSink()
    settings = FailOpenSettings(capacity=2, flush_timeout_s=0.01, writer_workers=2)
    buffer = BoundedBuffer(sink, settings)
    assert buffer.enqueue(_event(0)).accepted

    started = time.perf_counter()
    result = buffer.flush()
    elapsed = time.perf_counter() - started

    assert elapsed < 0.05
    assert result.timed_out is True
    assert result.flushed == 0
    assert result.storage_failures == 1
    assert result.remaining == 1
    assert buffer.counters()["storage_failures"] == 1
    assert buffer.depth == 1

    gate.set()
    time.sleep(0.05)
    recovered = buffer.flush()

    assert recovered.flushed == 1
    assert recovered.remaining == 0
    assert sink.written
