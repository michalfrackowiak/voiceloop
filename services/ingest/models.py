"""Request and response models for the ingest HTTP API."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

AudioState = Literal["pending", "stored", "referenced", "unavailable"]


class Receipt(BaseModel):
    """Idempotency receipt returned for every accepted ingest operation."""

    event_id: str
    call_id: str
    kind: Literal["turn", "outcome"] = "turn"
    status: Literal["queued", "persisted", "failed", "shed"] = "queued"
    audio_state: AudioState | None = None
    error: str | None = None


class OutcomeRequest(BaseModel):
    """Late-arriving outcome payload merged into a call by ``call_id``."""

    model_config = ConfigDict(extra="forbid")

    status: str
    final_record: dict[str, Any] | None = None
    updated_at: str | None = None
    event_id: str | None = None


class AudioUploadRequest(BaseModel):
    """Request for a presigned PUT URL targeting an object-store key."""

    model_config = ConfigDict(extra="forbid")

    call_id: str = Field(min_length=1, max_length=128)
    turn_index: int = Field(ge=0)
    filename: str = Field(default="turn.wav", min_length=1, max_length=128)
    content_type: str | None = Field(default=None, max_length=128)
    size_bytes: int | None = Field(default=None, ge=0)


class AudioUpload(BaseModel):
    """Presigned upload instructions; audio bytes never pass through Postgres."""

    upload_id: str
    key: str
    uri: str
    method: Literal["PUT"] = "PUT"
    url: str
    headers: dict[str, str]
    expires_in_s: int


class StoredAudio(BaseModel):
    """Metadata returned after audio bytes are accepted by the object store."""

    key: str
    uri: str
    size_bytes: int
    content_type: str | None = None
