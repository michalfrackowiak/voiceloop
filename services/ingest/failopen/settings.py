"""Typed settings for the fail-open ingest buffer."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import Field

from voiceloop_common import BaseSettings, settings_config


class FailOpenSettings(BaseSettings):
    """Environment-backed configuration read from ``VOICELOOP_INGEST_FAILOPEN_*``.

    Defaults favour the call path: a small queue wait, bounded flush time, and
    an in-memory store. The disk backend uses a private temporary directory and
    adds no dependencies beyond the standard library.
    """

    model_config = settings_config("VOICELOOP_INGEST_FAILOPEN_")

    backend: Literal["memory", "disk"] = "memory"
    capacity: int = Field(default=1024, ge=1)
    enqueue_timeout_ms: int = Field(default=25, ge=0, le=1000)
    max_payload_bytes: int = Field(default=64 * 1024, ge=1)
    writer_workers: int = Field(default=2, ge=1, le=8)
    flush_timeout_s: float = Field(default=2.0, gt=0)
    degraded_ratio: float = Field(default=0.75, ge=0.0, le=1.0)
    disk_dir: Path | None = None
