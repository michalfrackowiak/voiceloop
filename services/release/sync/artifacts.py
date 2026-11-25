"""Artifact download seam and deterministic SHA-256 verification.

> [!IMPORTANT]
> **FAKE ONLY.** :class:`FakeArtifactStore` keeps bytes in memory and writes
> them into the temporary directory handed to it by the agent. It never
> contacts MLflow, S3 or any network service.

A real :class:`ArtifactStore` adapter would call
``mlflow.artifacts.download_artifacts(artifact_uri, dst_path=...)`` (or an
equivalent object-store client) and write the exact deployable bytes under
``destination``. ``mlflow`` is deliberately not imported here: the dependency
stays optional and the sync agent verifies whatever lands on disk itself.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from voiceloop_common import StorageError, ValidationError

_ARTIFACT_FILENAME = "artifact.bin"
_HEX_DIGITS = frozenset("0123456789abcdef")


@dataclass(frozen=True)
class ArtifactRef:
    """Registry metadata describing the deployable artifact for one version."""

    uri: str
    sha256: str

    def __post_init__(self) -> None:
        """Validate the URI and require a canonical SHA-256 hex digest."""
        _require_text(self.uri, "artifact URI")
        object.__setattr__(self, "sha256", _normalise_sha256(self.sha256))


@dataclass(frozen=True)
class DownloadedArtifact:
    """Artifact bytes materialised locally, ready for hashing and staging."""

    uri: str
    root: Path


class ArtifactStore(Protocol):
    """Download seam; one call materialises one artifact in a local temp dir."""

    def download(self, artifact_uri: str, destination: Path) -> DownloadedArtifact:
        """Write the exact artifact bytes under ``destination``.

        Args:
            artifact_uri: Registry-supplied URI of the deployable artifact.
            destination: Empty temporary directory owned by the caller.

        Returns:
            The local root that :func:`hash_tree` should verify.

        Raises:
            StorageError: If the bytes cannot be materialised.

        """
        ...


class FakeArtifactStore:
    """Deterministic in-memory FAKE artifact store; no network or MLflow."""

    def __init__(self, blobs: Mapping[str, bytes] | None = None) -> None:
        """Initialise from an optional URI-to-bytes mapping."""
        self._blobs: dict[str, bytes] = {uri: bytes(data) for uri, data in (blobs or {}).items()}
        self._failures: set[str] = set()

    def put(self, artifact_uri: str, content: bytes) -> str:
        """Store fake bytes and return their canonical SHA-256 digest."""
        _require_text(artifact_uri, "artifact URI")
        if not isinstance(content, bytes):
            raise ValidationError("artifact content must be bytes")
        self._blobs[artifact_uri] = bytes(content)
        return hash_payload(content)

    def fail(self, artifact_uri: str) -> None:
        """Configure this URI to fail on the next and all later downloads."""
        _require_text(artifact_uri, "artifact URI")
        self._failures.add(artifact_uri)

    def download(self, artifact_uri: str, destination: Path) -> DownloadedArtifact:
        """Write the stored bytes to ``destination/artifact.bin`` (FAKE)."""
        _require_text(artifact_uri, "artifact URI")
        if artifact_uri in self._failures:
            raise StorageError("fake artifact download failed")
        try:
            content = self._blobs[artifact_uri]
        except KeyError as exc:
            raise StorageError("fake artifact URI is not registered") from exc
        try:
            destination.mkdir(parents=True, exist_ok=True)
            (destination / _ARTIFACT_FILENAME).write_bytes(content)
        except OSError as exc:
            raise StorageError("fake artifact could not be written locally") from exc
        return DownloadedArtifact(uri=artifact_uri, root=destination)


def hash_payload(content: bytes) -> str:
    """Return the deterministic tree hash of one file named ``artifact.bin``."""
    digest = hashlib.sha256()
    _feed_digest(digest, _ARTIFACT_FILENAME, bytes(content))
    return digest.hexdigest()


def hash_tree(root: Path) -> str:
    """Hash a downloaded artifact directory in a path-stable way.

    Files are hashed in sorted POSIX-relative-path order. Each entry contributes
    its relative path, an 8-byte big-endian length and its raw bytes, so two
    directories with identical content hash identically regardless of creation
    order. Symlinks and unreadable files are rejected.
    """
    if not root.is_dir():
        raise StorageError("downloaded artifact root is not a directory")
    entries = sorted(root.rglob("*"), key=lambda path: path.relative_to(root).as_posix())
    digest = hashlib.sha256()
    for path in entries:
        if path.is_symlink():
            raise StorageError("symlinks are not allowed in downloaded artifacts")
        if not path.is_file():
            continue
        relative = path.relative_to(root).as_posix()
        try:
            data = path.read_bytes()
        except OSError as exc:
            raise StorageError("downloaded artifact file could not be read") from exc
        _feed_digest(digest, relative, data)
    return digest.hexdigest()


def verify_artifact(artifact: DownloadedArtifact, expected_sha256: str) -> str:
    """Verify downloaded bytes against registry metadata and return the hash.

    Raises:
        ValidationError: If the digest does not match the expected SHA-256.

    """
    actual = hash_tree(artifact.root)
    if actual != _normalise_sha256(expected_sha256):
        raise ValidationError("artifact content hash does not match registry metadata")
    return actual


def _feed_digest(digest: Any, relative_path: str, data: bytes) -> None:
    """Feed one length-delimited file into a SHA-256 digest."""
    digest.update(relative_path.encode("utf-8"))
    digest.update(b"\0")
    digest.update(len(data).to_bytes(8, "big"))
    digest.update(data)


def _normalise_sha256(value: str) -> str:
    """Return a lower-case SHA-256 hex digest or raise a validation error."""
    if not isinstance(value, str):
        raise ValidationError("SHA-256 hash must be a string")
    normalised = value.strip().lower()
    if len(normalised) != 64 or any(char not in _HEX_DIGITS for char in normalised):
        raise ValidationError("SHA-256 hash must be 64 hexadecimal characters")
    return normalised


def _require_text(value: str, label: str) -> None:
    """Reject blank or non-string identifiers and URIs."""
    if not isinstance(value, str) or not value.strip():
        raise ValidationError(f"{label} must be a non-blank string")
