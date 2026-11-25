"""Disk-backed buffering tests using stdlib temp directories."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from services.ingest.failopen import BoundedBuffer, FailOpenSettings, RecordingSink
from voiceloop_common.errors import StorageError


def _event(index: int) -> dict[str, Any]:
    """Return a small event with a stable id."""
    return {"event_id": f"disk-{index}", "call_id": f"call-{index}", "stt": {"text": ""}}


def test_disk_buffer_survives_outage_and_cleans_up(tmp_path: Path) -> None:
    sink = RecordingSink(down=True, error=StorageError)
    settings = FailOpenSettings(backend="disk", capacity=4, disk_dir=tmp_path)

    with BoundedBuffer(sink, settings) as buffer:
        assert all(buffer.enqueue(_event(i)).accepted for i in range(2))

        roots = list(tmp_path.iterdir())
        assert len(roots) == 1
        assert len(list(roots[0].glob("*.json"))) == 2

        failed = buffer.flush()
        assert failed.flushed == 0
        assert failed.storage_failures == 2
        assert buffer.depth == 2

        sink.set_down(False)
        recovered = buffer.flush()
        assert recovered.flushed == 2
        assert recovered.remaining == 0
        assert [event["event_id"] for event in sink.written] == ["disk-0", "disk-1"]
        assert list(roots[0].glob("*.json")) == []

    assert list(tmp_path.iterdir()) == []


def test_disk_queue_capacity_is_enforced(tmp_path: Path) -> None:
    sink = RecordingSink(down=True)
    settings = FailOpenSettings(
        backend="disk",
        capacity=1,
        disk_dir=tmp_path,
        enqueue_timeout_ms=20,
    )

    with BoundedBuffer(sink, settings) as buffer:
        assert buffer.enqueue(_event(0)).accepted
        shed = buffer.enqueue(_event(1))

        assert not shed.accepted and shed.reason == "queue_full"
        assert buffer.depth == 1
        assert buffer.counters()["dropped"] == 1
