"""Object-store boundary for audio bytes, with an in-memory fake."""

from __future__ import annotations

import hashlib
import hmac
import re
from dataclasses import dataclass
from typing import Protocol
from urllib.parse import quote

from voiceloop_common.errors import StorageError

_SAFE_NAME = re.compile(r"[^A-Za-z0-9._-]+")


@dataclass(frozen=True)
class PresignedUpload:
    """A short-lived URL that lets a caller PUT audio straight to object storage."""

    upload_id: str
    key: str
    uri: str
    url: str
    method: str
    headers: dict[str, str]
    expires_in_s: int


@dataclass(frozen=True)
class StoredAudio:
    """Metadata for audio bytes accepted by the object store."""

    key: str
    uri: str
    size_bytes: int
    content_type: str | None


class ObjectStore(Protocol):
    """Storage interface shared by the in-memory fake and a future S3 adapter."""

    bucket: str
    scheme: str

    def key_for(self, call_id: str, turn_index: int, filename: str) -> str:
        """Build a collision-free object key for one turn's audio."""
        ...

    def uri_for(self, key: str) -> str:
        """Return the canonical URI for an object key."""
        ...

    def presign_put(
        self, key: str, ttl_s: int, secret: str, content_type: str | None = None
    ) -> PresignedUpload:
        """Create a presigned PUT URL for an object key."""
        ...

    def put(self, key: str, data: bytes, content_type: str | None = None) -> StoredAudio:
        """Store audio bytes and return their metadata."""
        ...

    def get(self, key: str) -> bytes | None:
        """Return stored bytes or None when the key is absent."""
        ...

    def exists(self, key: str) -> bool:
        """Report whether an object exists without returning its bytes."""
        ...


class InMemoryObjectStore:
    """Thread-safe in-memory fake with deterministic presigned URLs."""

    def __init__(
        self,
        bucket: str = "voiceloop-local",
        scheme: str = "s3",
        *,
        fail_put: bool = False,
        fail_presign: bool = False,
    ) -> None:
        """Create an empty store; failure flags simulate storage outages."""
        self.bucket = bucket
        self.scheme = scheme
        self.fail_put = fail_put
        self.fail_presign = fail_presign
        self._objects: dict[str, tuple[bytes, str | None]] = {}

    def key_for(self, call_id: str, turn_index: int, filename: str) -> str:
        """Build a collision-free object key for one turn's audio."""
        safe_call = _SAFE_NAME.sub("_", call_id).strip("_.") or "call"
        basename = filename.replace("\\", "/").rsplit("/", 1)[-1]
        safe_file = _SAFE_NAME.sub("_", basename).strip("_.") or "turn.wav"
        return f"calls/{safe_call}/turns/{turn_index:06d}/{safe_file}"

    def uri_for(self, key: str) -> str:
        """Return the canonical URI for an object key."""
        return f"{self.scheme}://{self.bucket}/{key}"

    def presign_put(
        self, key: str, ttl_s: int, secret: str, content_type: str | None = None
    ) -> PresignedUpload:
        """Create a deterministic presigned PUT URL (no network or real secret)."""
        if self.fail_presign:
            raise StorageError("Object store could not sign an upload URL")
        payload = f"{self.bucket}\n{key}\n{ttl_s}".encode()
        signature = hmac.new(secret.encode("utf-8"), payload, hashlib.sha256).hexdigest()[:24]
        url = (
            f"https://{self.bucket}.s3.local/{quote(key, safe='/')}"
            f"?X-Amz-Expires={ttl_s}&X-Amz-Signature={signature}"
        )
        headers = {"Content-Type": content_type or "audio/wav"}
        return PresignedUpload(
            upload_id="",
            key=key,
            uri=self.uri_for(key),
            url=url,
            method="PUT",
            headers=headers,
            expires_in_s=ttl_s,
        )

    def put(self, key: str, data: bytes, content_type: str | None = None) -> StoredAudio:
        """Store audio bytes in memory and return metadata only."""
        if self.fail_put:
            raise StorageError("Object store rejected the upload")
        self._objects[key] = (bytes(data), content_type)
        return StoredAudio(
            key=key,
            uri=self.uri_for(key),
            size_bytes=len(data),
            content_type=content_type,
        )

    def get(self, key: str) -> bytes | None:
        """Return stored bytes or None when the key is absent."""
        stored = self._objects.get(key)
        return stored[0] if stored is not None else None

    def exists(self, key: str) -> bool:
        """Report whether an object exists without returning its bytes."""
        return key in self._objects
