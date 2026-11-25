"""Content-addressed, read-only fake artifact stand-ins.

A :class:`FakeArtifact` stores only a small synthetic descriptor (runtime label,
failure behaviour and conversion recipe). It contains no audio bytes, no
transcripts and no PII. Its SHA-256 covers the canonical descriptor bytes, and
the descriptor is immutable after construction. The artifact URI always embeds
that hash, so an evaluator cannot silently substitute a training checkpoint.
"""

from __future__ import annotations

import hashlib
import json
import re
import stat
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any

from voiceloop_common import ConfigError, ValidationError

from .settings import ArtifactBehaviour

ARTIFACT_URI_PREFIX = "artifact+sha256:"
_ARTIFACT_SCHEMA_VERSION = "internal-1.0.0"
_DIGEST = re.compile(r"[0-9a-f]{64}\Z")


def canonical_bytes(payload: Mapping[str, Any]) -> bytes:
    """Return deterministic UTF-8 JSON bytes for a flat descriptor mapping."""
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode(
        "utf-8"
    )


def sha256_hex(data: bytes) -> str:
    """Return the SHA-256 hex digest of ``data``."""
    return hashlib.sha256(data).hexdigest()


def artifact_hash_from_uri(uri: str) -> str:
    """Extract and validate the artifact digest from a content-addressed URI.

    Args:
        uri: URI shaped ``artifact+sha256:<64 hex digits>://<name>``.

    Returns:
        The validated lowercase hex digest.

    Raises:
        ValidationError: If the URI is not a content-addressed artifact URI.

    """
    if not isinstance(uri, str) or not uri.startswith(ARTIFACT_URI_PREFIX):
        raise ValidationError("artifact URI must be content-addressed with artifact+sha256:")
    digest = uri[len(ARTIFACT_URI_PREFIX) :].split("://", 1)[0]
    if not _DIGEST.fullmatch(digest):
        raise ValidationError("artifact URI does not contain a valid SHA-256 digest")
    return digest


@dataclass(frozen=True)
class FakeArtifact:
    """Immutable, content-addressed descriptor for a fake deployable artifact."""

    runtime_label: str
    behaviour: ArtifactBehaviour
    conversion_recipe: str
    sha256: str = field(init=False)
    _descriptor: Mapping[str, Any] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        """Compute the content hash and freeze the descriptor mapping."""
        if not self.runtime_label.strip() or not self.conversion_recipe.strip():
            raise ValidationError("artifact runtime label and conversion recipe must be non-blank")
        descriptor: dict[str, Any] = {
            "artifact_schema_version": _ARTIFACT_SCHEMA_VERSION,
            "runtime_label": self.runtime_label,
            "behaviour": self.behaviour,
            "conversion_recipe": self.conversion_recipe,
            "synthetic": True,
            "contains_audio_bytes": False,
            "contains_pii": False,
        }
        object.__setattr__(self, "_descriptor", MappingProxyType(descriptor))
        object.__setattr__(self, "sha256", sha256_hex(canonical_bytes(descriptor)))

    @property
    def uri(self) -> str:
        """Return the exact content-addressed URI used by the gate."""
        return f"{ARTIFACT_URI_PREFIX}{self.sha256}://fake/{self.behaviour}"

    @property
    def descriptor(self) -> Mapping[str, Any]:
        """Return a read-only view of the descriptor; mutation raises TypeError."""
        return self._descriptor

    def descriptor_bytes(self) -> bytes:
        """Return the canonical bytes covered by :attr:`sha256`."""
        return canonical_bytes(dict(self._descriptor))


def create_fake_artifact(
    behaviour: ArtifactBehaviour,
    runtime_label: str = "fake-ctranslate2",
) -> FakeArtifact:
    """Build one fake artifact descriptor for a deterministic behaviour.

    Args:
        behaviour: Injected decode behaviour (``healthy`` or a failure case).
        runtime_label: Synthetic runtime name recorded in the descriptor.

    Returns:
        A frozen, content-addressed fake artifact.

    """
    return FakeArtifact(
        runtime_label=runtime_label,
        behaviour=behaviour,
        conversion_recipe="synthetic-merge-then-convert:fixture-only",
    )


class ReadOnlyArtifactStore:
    """Write-once, content-addressed store for fake artifact descriptors."""

    def __init__(self, directory: str | Path) -> None:
        """Initialise the store with a caller-supplied output directory."""
        if not str(directory).strip():
            raise ConfigError("artifact store directory must be non-blank")
        self.directory = Path(directory).resolve()

    def path_for(self, digest: str) -> Path:
        """Return the on-disk path for one validated artifact digest."""
        artifact_hash_from_uri(f"{ARTIFACT_URI_PREFIX}{digest}://fake")
        return self.directory / f"{digest}.json"

    def put(self, artifact: FakeArtifact) -> Path:
        """Persist a descriptor once and make it read-only.

        Re-putting an identical artifact is a no-op; re-putting different bytes
        under the same digest fails closed.

        Args:
            artifact: Fake artifact to store.

        Returns:
            Path of the stored descriptor.

        Raises:
            ValidationError: If the stored bytes would not match the digest.

        """
        self.directory.mkdir(parents=True, exist_ok=True)
        path = self.path_for(artifact.sha256)
        payload = artifact.descriptor_bytes()
        if path.exists():
            if path.read_bytes() != payload:
                raise ValidationError("content-addressed artifact was modified after creation")
            return path
        with path.open("xb") as handle:
            handle.write(payload)
        try:
            path.chmod(stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
        except OSError:  # pragma: no cover - Windows/ACL-dependent hardening only
            pass
        return path

    def load(self, digest: str) -> FakeArtifact:
        """Load and re-verify a stored artifact by digest.

        Args:
            digest: Artifact SHA-256 hex digest.

        Returns:
            The reconstructed immutable artifact.

        Raises:
            ValidationError: If stored bytes no longer hash to ``digest``.

        """
        path = self.path_for(digest)
        try:
            payload = path.read_bytes()
        except OSError as exc:
            raise ValidationError(f"cannot read stored artifact: {path}") from exc
        if sha256_hex(payload) != digest:
            raise ValidationError("stored artifact bytes do not match their content address")
        try:
            decoded = json.loads(payload.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValidationError("stored artifact descriptor is not valid JSON") from exc
        if not isinstance(decoded, dict):
            raise ValidationError("stored artifact descriptor must be an object")
        if decoded.get("artifact_schema_version") != _ARTIFACT_SCHEMA_VERSION:
            raise ValidationError("unsupported stored artifact schema version")
        artifact = FakeArtifact(
            runtime_label=str(decoded["runtime_label"]),
            behaviour=decoded["behaviour"],
            conversion_recipe=str(decoded["conversion_recipe"]),
        )
        if artifact.sha256 != digest:
            raise ValidationError("reconstructed artifact hash does not match its address")
        return artifact
