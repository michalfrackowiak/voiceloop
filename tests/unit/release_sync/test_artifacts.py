"""P3-06: artifact byte hashing and download seam."""

from __future__ import annotations

from pathlib import Path

import pytest

from services.release.sync import (
    ArtifactRef,
    FakeArtifactStore,
    hash_payload,
    hash_tree,
    verify_artifact,
)
from voiceloop_common import StorageError, ValidationError


def test_hash_tree_is_order_independent_and_content_sensitive(tmp_path: Path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    first.mkdir()
    second.mkdir()
    (second / "b.bin").write_bytes(b"b")
    (second / "a.bin").write_bytes(b"a")
    (first / "a.bin").write_bytes(b"a")
    (first / "b.bin").write_bytes(b"b")

    assert hash_tree(first) == hash_tree(second)
    (first / "b.bin").write_bytes(b"changed")
    assert hash_tree(first) != hash_tree(second)


def test_fake_store_download_and_verify(tmp_path: Path) -> None:
    store = FakeArtifactStore()
    digest = store.put("fake://mlflow/v1", b"converted-artifact")

    assert digest == hash_payload(b"converted-artifact")
    destination = tmp_path / "download"
    artifact = store.download("fake://mlflow/v1", destination)

    assert artifact.root == destination
    assert (destination / "artifact.bin").read_bytes() == b"converted-artifact"
    assert verify_artifact(artifact, digest) == digest
    assert verify_artifact(artifact, digest.upper()) == digest


def test_hash_mismatch_is_rejected(tmp_path: Path) -> None:
    store = FakeArtifactStore()
    store.put("fake://mlflow/v1", b"payload")
    artifact = store.download("fake://mlflow/v1", tmp_path / "download")

    with pytest.raises(ValidationError):
        verify_artifact(artifact, "0" * 64)


def test_artifact_ref_validates_metadata() -> None:
    ref = ArtifactRef("fake://mlflow/v1", "A" * 64)
    assert ref.sha256 == "a" * 64

    with pytest.raises(ValidationError):
        ArtifactRef("fake://mlflow/v1", "not-a-hash")
    with pytest.raises(ValidationError):
        ArtifactRef("  ", "a" * 64)


def test_fake_store_failures_are_typed(tmp_path: Path) -> None:
    store = FakeArtifactStore()
    store.fail("fake://mlflow/broken")

    with pytest.raises(StorageError):
        store.download("fake://mlflow/broken", tmp_path / "broken")
    with pytest.raises(StorageError):
        store.download("fake://mlflow/missing", tmp_path / "missing")


def test_hash_tree_rejects_symlinks(tmp_path: Path) -> None:
    root = tmp_path / "download"
    root.mkdir()
    target = tmp_path / "outside.bin"
    target.write_bytes(b"outside")
    try:
        (root / "link.bin").symlink_to(target)
    except (OSError, NotImplementedError):
        pytest.skip("symlinks are unavailable in this environment")

    with pytest.raises(StorageError):
        hash_tree(root)
