"""Metrics, settings and health-snapshot tests."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime

from pytest import MonkeyPatch

from services.ingest.failopen import (
    BoundedBuffer,
    FailOpenMetrics,
    FailOpenSettings,
    RecordingSink,
)
from voiceloop_common import frozen_clock, use_clock


def test_metrics_counters_and_health_thresholds() -> None:
    metrics = FailOpenMetrics(capacity=10, degraded_ratio=0.5)
    metrics.record_accepted()
    metrics.record_buffered()
    metrics.record_flushed(2)
    metrics.record_dropped()
    metrics.record_storage_failure()

    with use_clock(frozen_clock(datetime(2026, 1, 2, 3, 4, 5, tzinfo=UTC))):
        healthy = metrics.health(4)
        degraded = metrics.health(5)

    assert healthy.status == "healthy"
    assert degraded.status == "degraded"
    assert degraded.observed_at == "2026-01-02T03:04:05Z"
    assert degraded.to_dict()["degraded_threshold"] == 5
    assert degraded.accepted == 1
    assert degraded.buffered == 1
    assert degraded.flushed == 2
    assert degraded.dropped == 1
    assert degraded.storage_failures == 1
    assert metrics.counters()["flushed"] == 2


def test_metrics_updates_are_thread_safe() -> None:
    metrics = FailOpenMetrics(capacity=10)

    def worker(_: int) -> None:
        """Increment counters repeatedly from one worker."""
        for _ in range(100):
            metrics.record_accepted()
            metrics.record_flushed()

    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(worker, range(8)))

    assert metrics.counters()["accepted"] == 800
    assert metrics.counters()["flushed"] == 800


def test_failopen_settings_use_the_dedicated_env_prefix(monkeypatch: MonkeyPatch) -> None:
    monkeypatch.setenv("VOICELOOP_INGEST_FAILOPEN_CAPACITY", "7")
    monkeypatch.setenv("VOICELOOP_INGEST_FAILOPEN_ENQUEUE_TIMEOUT_MS", "12")
    monkeypatch.setenv("VOICELOOP_INGEST_FAILOPEN_BACKEND", "disk")
    monkeypatch.setenv("VOICELOOP_INGEST_FAILOPEN_DEGRADED_RATIO", "0.5")
    monkeypatch.setenv("VOICELOOP_INGEST_FAILOPEN_MAX_PAYLOAD_BYTES", "256")
    monkeypatch.setenv("VOICELOOP_INGEST_FAILOPEN_WRITER_WORKERS", "3")
    monkeypatch.setenv("VOICELOOP_INGEST_FAILOPEN_DISK_DIR", "C:/tmp/voiceloop-failopen-test")

    settings = FailOpenSettings()

    assert settings.capacity == 7
    assert settings.enqueue_timeout_ms == 12
    assert settings.backend == "disk"
    assert settings.degraded_ratio == 0.5
    assert settings.max_payload_bytes == 256
    assert settings.writer_workers == 3
    assert settings.disk_dir is not None
    assert settings.disk_dir.name == "voiceloop-failopen-test"


def test_buffer_health_exposes_depth_and_counters() -> None:
    buffer = BoundedBuffer(
        RecordingSink(down=True),
        FailOpenSettings(capacity=4, degraded_ratio=0.5),
    )

    assert buffer.health().status == "healthy"
    buffer.enqueue({"event_id": "h-1", "stt": {"text": ""}})
    snapshot = buffer.health()

    assert snapshot.status == "healthy"
    assert snapshot.depth == 1
    assert snapshot.accepted == 1
    assert snapshot.to_dict()["status"] == "healthy"
