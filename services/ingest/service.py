"""Ingest service orchestrating validation receipts, queueing and audio uploads."""

from __future__ import annotations

from typing import Any

from services.ingest.errors import IngestHTTPError
from services.ingest.models import AudioUpload, AudioUploadRequest, Receipt, StoredAudio
from services.ingest.queue import Envelope, IngestQueue
from services.ingest.repository import IngestRepository
from services.ingest.settings import IngestSettings
from services.ingest.storage import ObjectStore
from voiceloop_common import new_id


class IngestService:
    """Application service shared by all ingest HTTP endpoints."""

    def __init__(
        self,
        settings: IngestSettings,
        repository: IngestRepository,
        object_store: ObjectStore,
        *,
        autostart_worker: bool = True,
    ) -> None:
        """Wire settings, storage backends and the write queue together."""
        self.settings = settings
        self.repository = repository
        self.object_store = object_store
        self.queue = IngestQueue(
            settings,
            repository,
            object_store,
            autostart=autostart_worker,
        )
        self._uploads: dict[str, tuple[str, str | None]] = {}

    def enqueue_turn(self, event: dict[str, Any]) -> tuple[Receipt, bool]:
        """Queue a validated turn event and return its receipt and duplicate flag."""
        envelope = Envelope(
            kind="turn",
            event_id=str(event["event_id"]),
            call_id=str(event["call_id"]),
            payload=event,
        )
        receipt = Receipt(
            event_id=envelope.event_id,
            call_id=envelope.call_id,
            kind="turn",
            status="queued",
            audio_state="pending",
        )
        return self.queue.enqueue(envelope, receipt)

    def enqueue_outcome(
        self, call_id: str, event_id: str, outcome: dict[str, Any]
    ) -> tuple[Receipt, bool]:
        """Queue a late outcome merge and return its receipt and duplicate flag."""
        envelope = Envelope(kind="outcome", event_id=event_id, call_id=call_id, payload=outcome)
        receipt = Receipt(event_id=event_id, call_id=call_id, kind="outcome", status="queued")
        return self.queue.enqueue(envelope, receipt)

    def issue_upload(self, request: AudioUploadRequest) -> AudioUpload:
        """Issue a presigned PUT URL for one turn's audio object."""
        key = self.object_store.key_for(request.call_id, request.turn_index, request.filename)
        signed = self.object_store.presign_put(
            key,
            self.settings.presign_ttl_s,
            self.settings.presign_secret.get_secret_value(),
            request.content_type,
        )
        upload_id = new_id()
        self._uploads[upload_id] = (key, request.content_type)
        return AudioUpload(
            upload_id=upload_id,
            key=key,
            uri=signed.uri,
            url=signed.url,
            headers=signed.headers,
            expires_in_s=signed.expires_in_s,
        )

    def store_audio(self, upload_id: str, key: str, data: bytes) -> StoredAudio:
        """Accept bytes for a previously issued upload (local fake path)."""
        pending = self._uploads.get(upload_id)
        if pending is None or pending[0] != key:
            raise IngestHTTPError(
                400,
                "invalid_upload",
                "The upload id does not match the requested object key.",
                {"field": "upload_id"},
            )
        if len(data) > self.settings.max_audio_bytes:
            raise IngestHTTPError(
                413,
                "payload_too_large",
                "Audio upload exceeds the configured size limit.",
                {"field": "body", "limit_bytes": self.settings.max_audio_bytes},
            )
        stored = self.object_store.put(key, data, pending[1])
        return StoredAudio(
            key=stored.key,
            uri=stored.uri,
            size_bytes=stored.size_bytes,
            content_type=stored.content_type,
        )

    def start(self) -> None:
        """Start the background write worker."""
        self.queue.start()

    def stop(self) -> None:
        """Stop the background write worker."""
        self.queue.stop()

    def process_pending(self) -> int:
        """Drain queued writes synchronously (used by tests and shutdown hooks)."""
        return self.queue.process_pending()

    def flush(self, timeout: float | None = None) -> bool:
        """Wait for queued writes to finish; return False on timeout."""
        return self.queue.flush(timeout)

    def stats(self) -> dict[str, int]:
        """Return queue depth and write counters for health checks."""
        return self.queue.stats()
