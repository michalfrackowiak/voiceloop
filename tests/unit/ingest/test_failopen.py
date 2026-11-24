"""Load/failure tests proving ingest never blocks on storage."""

from __future__ import annotations

import time
from typing import Any
from uuid import uuid4

from fastapi.testclient import TestClient

from services.ingest.app import create_app
from services.ingest.repository import InMemoryRepository
from services.ingest.service import IngestService
from services.ingest.settings import IngestSettings
from services.ingest.storage import InMemoryObjectStore, ObjectStore
from voiceloop_common.errors import StorageError


class FailingRepository(InMemoryRepository):
    """Repository fake whose metadata writes always fail."""

    def save_call(self, call_id: str, consent: Any, seen_at: str | None = None) -> dict[str, Any]:
        raise StorageError("metadata store down")

    def save_turn(self, event: dict[str, Any], *, audio_state: str) -> dict[str, Any]:
        raise StorageError("metadata store down")

    def save_model_version(
        self, kind: str, model_id: str, model_version: str, seen_at: str | None = None
    ) -> None:
        raise StorageError("metadata store down")


class FailingObjectStore(InMemoryObjectStore):
    """Object-store fake whose reads always fail."""

    def exists(self, key: str) -> bool:
        raise StorageError("object store down")


def _event(index: int, *, audio_uri: str = "s3://voiceloop-local/missing.wav") -> dict[str, Any]:
    """Return a minimal schema-valid event with a unique id."""
    return {
        "event_id": str(uuid4()),
        "call_id": f"call-{index}",
        "turn_index": 0,
        "audio": {"uri": audio_uri},
        "stt": {"text": ""},
    }


def test_storage_outage_does_not_block_or_raise_into_callers() -> None:
    settings = IngestSettings(
        autostart_worker=True,
        worker_poll_interval_s=0.001,
        worker_shutdown_timeout_s=3.0,
        log_level="CRITICAL",
    )
    repo = FailingRepository()
    store = FailingObjectStore()
    service = IngestService(settings, repo, store, autostart_worker=True)
    app = create_app(settings, repo, store, autostart_worker=True, service=service)

    responses: list[Any] = []
    latencies: list[float] = []
    with TestClient(app) as client:
        started = time.perf_counter()
        for index in range(150):
            call_started = time.perf_counter()
            responses.append(client.post("/v1/turns", json=_event(index)))
            latencies.append(time.perf_counter() - call_started)
        elapsed = time.perf_counter() - started
        assert service.flush(timeout=2.0)
        stats = client.get("/healthz").json()

    assert all(response.status_code == 202 for response in responses)
    p95 = sorted(latencies)[int(0.95 * len(latencies))]
    assert p95 < 0.1, f"p95 ingest latency was {p95:.3f}s"
    assert max(latencies) < 0.5, f"worst ingest latency was {max(latencies):.3f}s"
    assert elapsed < 20.0, f"150 ingest calls took {elapsed:.2f}s"
    assert stats["failed"] == 150
    assert stats["queued"] == 0
    receipts = [response.json() for response in responses]
    assert all(receipt["status"] == "queued" for receipt in receipts)


def test_failed_receipts_record_storage_error_without_5xx() -> None:
    settings = IngestSettings(autostart_worker=True, worker_poll_interval_s=0.001)
    repo = FailingRepository()
    store = FailingObjectStore()
    service = IngestService(settings, repo, store, autostart_worker=True)
    app = create_app(settings, repo, store, autostart_worker=True, service=service)

    with TestClient(app) as client:
        event = _event(0)
        response = client.post("/v1/turns", json=event)
        assert service.flush(timeout=2.0)
        stored = repo.get_receipt(event["event_id"])

    assert response.status_code == 202
    assert stored is not None
    assert stored["status"] == "failed"
    assert stored["error"] == "storage_error"


def test_object_store_outage_is_isolated_from_metadata_writes() -> None:
    settings = IngestSettings(autostart_worker=False)
    repo = InMemoryRepository()
    store: ObjectStore = FailingObjectStore()
    service = IngestService(settings, repo, store, autostart_worker=False)
    app = create_app(settings, repo, store, autostart_worker=False, service=service)

    with TestClient(app) as client:
        event = _event(0, audio_uri="s3://voiceloop-local/turn.wav")
        response = client.post("/v1/turns", json=event)
        assert service.process_pending() == 1

    assert response.status_code == 202
    receipt = repo.get_receipt(event["event_id"])
    assert receipt is not None
    assert receipt["status"] == "persisted"
    turn = repo.get_turn(event["event_id"])
    assert turn is not None
    assert turn["audio_state"] == "unavailable"


def test_full_queue_sheds_with_defined_signal_instead_of_blocking() -> None:
    settings = IngestSettings(autostart_worker=False, queue_capacity=1)
    repo = InMemoryRepository()
    store = InMemoryObjectStore()
    service = IngestService(settings, repo, store, autostart_worker=False)
    app = create_app(settings, repo, store, autostart_worker=False, service=service)

    with TestClient(app) as client:
        first = client.post("/v1/turns", json=_event(0))
        second = client.post("/v1/turns", json=_event(1))
        assert service.stats()["shed"] == 1
        assert service.process_pending() == 1

    assert first.status_code == second.status_code == 202
    assert first.json()["status"] == "queued"
    assert second.json()["status"] == "shed"
    assert second.json()["error"] == "queue_full"
    assert len(repo.list_turns()) == 1
