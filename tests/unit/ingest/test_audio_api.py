"""Tests for the presigned audio upload path."""

from __future__ import annotations

from fastapi.testclient import TestClient

from services.ingest.app import create_app
from services.ingest.repository import InMemoryRepository
from services.ingest.settings import IngestSettings
from services.ingest.storage import InMemoryObjectStore


def test_presign_then_put_stores_bytes_outside_postgres(
    client: TestClient, object_store: InMemoryObjectStore, repository: InMemoryRepository
) -> None:
    upload_response = client.post(
        "/v1/audio/uploads",
        json={"call_id": "call-audio", "turn_index": 2, "filename": "turn-002.wav"},
    )

    assert upload_response.status_code == 201, upload_response.text
    upload = upload_response.json()
    assert upload["method"] == "PUT"
    assert upload["uri"].startswith("s3://voiceloop-local/")
    assert "X-Amz-Signature" in upload["url"]
    assert upload["expires_in_s"] > 0

    put_response = client.put(
        f"/v1/audio/objects/{upload['key']}",
        params={"upload_id": upload["upload_id"]},
        content=b"RIFF-synthetic-audio",
        headers={"Content-Type": "audio/wav"},
    )

    assert put_response.status_code == 201, put_response.text
    assert put_response.json()["size_bytes"] == len(b"RIFF-synthetic-audio")
    assert object_store.get(upload["key"]) == b"RIFF-synthetic-audio"
    assert repository.list_turns() == []
    assert repository.get_call("call-audio") is None


def test_upload_id_must_match_key(client: TestClient) -> None:
    upload = client.post(
        "/v1/audio/uploads", json={"call_id": "call-audio", "turn_index": 0}
    ).json()
    response = client.put(
        "/v1/audio/objects/wrong-key.wav",
        params={"upload_id": upload["upload_id"]},
        content=b"data",
    )

    assert response.status_code == 400
    body = response.json()
    assert body["code"] == "invalid_upload"
    assert body["details"]["field"] == "upload_id"


def test_upload_size_limit_is_enforced() -> None:
    settings = IngestSettings(autostart_worker=False, max_audio_bytes=4)
    app = create_app(settings, InMemoryRepository(), InMemoryObjectStore(), autostart_worker=False)

    with TestClient(app) as client:
        upload = client.post(
            "/v1/audio/uploads", json={"call_id": "call-audio", "turn_index": 0}
        ).json()
        response = client.put(
            f"/v1/audio/objects/{upload['key']}",
            params={"upload_id": upload["upload_id"]},
            content=b"too many bytes",
        )

    assert response.status_code == 413
    assert response.json()["code"] == "payload_too_large"


def test_presign_failure_is_structured_503() -> None:
    settings = IngestSettings(autostart_worker=False)
    store = InMemoryObjectStore(fail_presign=True)
    app = create_app(settings, InMemoryRepository(), store, autostart_worker=False)

    with TestClient(app) as client:
        response = client.post("/v1/audio/uploads", json={"call_id": "call-audio", "turn_index": 0})

    assert response.status_code == 503
    assert response.json() == {
        "code": "storage_error",
        "message": "Storage is temporarily unavailable.",
        "details": {},
    }


def test_audio_upload_request_is_validated(client: TestClient) -> None:
    response = client.post(
        "/v1/audio/uploads", json={"call_id": "", "turn_index": -1, "extra": True}
    )

    assert response.status_code == 422
    body = response.json()
    assert body["code"] == "validation_error"
    assert body["details"]["errors"]
    assert all("input" not in error for error in body["details"]["errors"])
