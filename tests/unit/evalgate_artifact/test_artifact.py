"""Content-addressed fake artifact and read-only store behaviour."""

from __future__ import annotations

import stat
from pathlib import Path

import pytest

from services.evalgate.artifact.artifact import (
    ARTIFACT_URI_PREFIX,
    ReadOnlyArtifactStore,
    artifact_hash_from_uri,
    canonical_bytes,
    create_fake_artifact,
    sha256_hex,
)
from voiceloop_common import ValidationError


def test_artifact_is_content_addressed_and_immutable() -> None:
    """Equivalent descriptors hash equally; mutation and foreign URIs fail closed."""
    first = create_fake_artifact("healthy")
    second = create_fake_artifact("healthy")
    different = create_fake_artifact("empty")

    assert first.sha256 == second.sha256 == sha256_hex(first.descriptor_bytes())
    assert first.sha256 != different.sha256
    assert first.uri.startswith(f"{ARTIFACT_URI_PREFIX}{first.sha256}://")
    assert artifact_hash_from_uri(first.uri) == first.sha256
    assert first.descriptor["contains_audio_bytes"] is False
    assert first.descriptor["contains_pii"] is False
    with pytest.raises(TypeError):
        first.descriptor["behaviour"] = "empty"  # type: ignore[index]

    with pytest.raises(ValidationError, match="content-addressed"):
        artifact_hash_from_uri("fake://trainer/checkpoint")
    with pytest.raises(ValidationError, match="valid SHA-256"):
        artifact_hash_from_uri(f"{ARTIFACT_URI_PREFIX}short://fake")


def test_canonical_bytes_are_sorted_and_stable() -> None:
    """Hash inputs do not depend on Python dict insertion order."""
    assert canonical_bytes({"b": 2, "a": 1}) == canonical_bytes({"a": 1, "b": 2})


def test_read_only_store_roundtrip_and_tamper_detection(tmp_path: Path) -> None:
    """Stored descriptors reload only when their bytes still match the address."""
    artifact = create_fake_artifact("truncated")
    store = ReadOnlyArtifactStore(tmp_path / "artifacts")

    path = store.put(artifact)
    assert store.put(artifact) == path
    assert store.load(artifact.sha256) == artifact
    assert path.read_bytes() == artifact.descriptor_bytes()

    path.chmod(stat.S_IRUSR | stat.S_IWUSR)
    path.write_bytes(b'{"tampered":true}')
    with pytest.raises(ValidationError, match="do not match their content address"):
        store.load(artifact.sha256)

    with pytest.raises(ValidationError, match="valid SHA-256"):
        store.path_for("not-a-digest")
