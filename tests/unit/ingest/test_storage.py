"""Tests for the in-memory object-store fake."""

from __future__ import annotations

import pytest

from services.ingest.storage import InMemoryObjectStore
from voiceloop_common.errors import StorageError


def test_key_and_uri_are_deterministic_and_safe() -> None:
    store = InMemoryObjectStore(bucket="fixtures", scheme="s3")

    key = store.key_for("call-0001", 3, "../turn 03.wav")

    assert key == "calls/call-0001/turns/000003/turn_03.wav"
    assert store.uri_for(key) == "s3://fixtures/calls/call-0001/turns/000003/turn_03.wav"


def test_presign_url_is_deterministic_scoped_and_expiring() -> None:
    store = InMemoryObjectStore(bucket="fixtures")
    key = store.key_for("call-1", 0, "turn.wav")

    first = store.presign_put(key, 900, "secret", "audio/wav")
    second = store.presign_put(key, 900, "secret", "audio/wav")
    other = store.presign_put(key, 60, "secret", "audio/wav")

    assert first == second
    assert first.url != other.url
    assert first.method == "PUT"
    assert first.uri == store.uri_for(key)
    assert first.expires_in_s == 900
    assert first.headers["Content-Type"] == "audio/wav"
    assert "X-Amz-Signature=" in first.url


def test_put_round_trip_and_failure_injection() -> None:
    store = InMemoryObjectStore()
    key = store.key_for("call-1", 0, "turn.wav")

    stored = store.put(key, b"RIFFfake", "audio/wav")

    assert stored.size_bytes == 8
    assert store.exists(key)
    assert store.get(key) == b"RIFFfake"
    assert store.get("missing") is None

    broken = InMemoryObjectStore(fail_put=True, fail_presign=True)
    with pytest.raises(StorageError):
        broken.put(key, b"data")
    with pytest.raises(StorageError):
        broken.presign_put(key, 900, "secret")
