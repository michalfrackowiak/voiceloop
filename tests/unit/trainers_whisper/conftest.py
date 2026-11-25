"""Shared synthetic fixtures for the Whisper trainer stand-in tests."""

from __future__ import annotations

from typing import Any

import pytest

from services.trainers.core import TrainerSettings
from services.trainers.whisper import (
    WhisperTrainingExample,
    sha256_text,
)

DATASET_HASH = "a" * 64
SAMPLE_TEXT = "reserve a standard room"
AUDIO_ROOT = "fake://audio"


def make_example(
    example_id: str,
    *,
    call_id: str = "call-train-a",
    sample_rate_hz: int = 16000,
    replay: bool = False,
    text: str = SAMPLE_TEXT,
) -> WhisperTrainingExample:
    """Build a digest-only synthetic example with no guest text on the wire."""
    audio_uri = f"{AUDIO_ROOT}/{call_id}/{example_id}.wav"
    return WhisperTrainingExample(
        example_id=example_id,
        call_id=call_id,
        audio_uri_sha256=sha256_text(audio_uri),
        target_text_sha256=sha256_text(text),
        duration_ms=300,
        sample_rate_hz=sample_rate_hz,
        replay=replay,
        synthetic=True,
        source="replay" if replay else "primary",
    )


@pytest.fixture
def manifest() -> dict[str, Any]:
    """Return a minimal P2-05-style manifest acceptable to the trainer contract."""
    return {"content_hash": DATASET_HASH, "dataset_id": "fixture-dataset", "examples": 4}


@pytest.fixture
def core_settings() -> TrainerSettings:
    """Return core job settings with a fake model and content-addressed URI prefix."""
    return TrainerSettings(
        model="fake-whisper-base",
        seed=7,
        artifact_uri_prefix="fake://whisper-artifacts",
    )


@pytest.fixture
def primary_examples() -> tuple[WhisperTrainingExample, ...]:
    """Return three non-sealed primary examples, including one 8 kHz row."""
    return (
        make_example("ex-1", call_id="call-train-a"),
        make_example("ex-2", call_id="call-train-b", sample_rate_hz=8000),
        make_example("ex-3", call_id="call-train-c"),
    )


@pytest.fixture
def replay_pool() -> tuple[WhisperTrainingExample, ...]:
    """Return earlier verified examples from calls absent from the primary set."""
    return (
        make_example("rp-1", call_id="call-replay-a", text="two nights please"),
        make_example("rp-2", call_id="call-replay-b", sample_rate_hz=8000),
        make_example("rp-3", call_id="call-replay-c"),
        make_example("rp-4", call_id="call-replay-d"),
    )
