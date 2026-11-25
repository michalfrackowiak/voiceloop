"""Shared synthetic fixtures for P3-02 tests (no real audio, PII or golden data)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from services.datasets import DatasetExample, export_nemo
from services.trainers.core import TrainerSettings
from services.trainers.nemo import (
    DEFAULT_CHECKPOINT_URI,
    NemoManifestRow,
    NemoSettings,
)

DATASET_HASH = "d" * 64
EVENT_IDS = (
    "11111111-1111-4111-8111-111111111111",
    "22222222-2222-4222-8222-222222222222",
    "33333333-3333-4333-8333-333333333333",
)
CALL_IDS = ("synthetic-nemo-call-a", "synthetic-nemo-call-b", "synthetic-nemo-call-c")
TEXTS = (
    "Check in on the twelfth for Ada Fixture, two guests.",
    "Make that a deluxe room.",
    "And three nights, not two.",
)
DURATIONS = (0.24, 0.32, 0.28)


def make_dataset_example(index: int) -> DatasetExample:
    """Build one synthetic STT example in the exact P2-05 NeMo export shape."""
    return DatasetExample(
        example_id=f"synthetic-nemo-{index}:stt",
        kind="stt",
        call_id=CALL_IDS[index],
        turn_index=index,
        audio_uri=f"synthetic://audio/nemo-{index}.wav",
        source_text=TEXTS[index],
        target_text=TEXTS[index],
        slots={},
        dialogue_context={},
        routing="both_fine",
        corrected_transcript=TEXTS[index],
        hypothesis=TEXTS[index],
        duration_ms=int(DURATIONS[index] * 1000),
        synthetic=True,
    )


@pytest.fixture
def nemo_export(tmp_path: Path) -> Path:
    """Export synthetic examples with the read-only P2-05 NeMo exporter."""
    return export_nemo(
        tuple(make_dataset_example(index) for index in range(3)),
        tmp_path / "dataset.nemo.jsonl",
    )


@pytest.fixture
def rows() -> tuple[NemoManifestRow, ...]:
    """Return synthetic rows matching the exported fixture."""
    return tuple(
        NemoManifestRow(
            audio_filepath=f"synthetic://audio/nemo-{index}.wav",
            duration=DURATIONS[index],
            text=TEXTS[index],
        )
        for index in range(3)
    )


@pytest.fixture
def nemo_settings(nemo_export: Path) -> NemoSettings:
    """Return settings pointing at the local synthetic P2-05 export."""
    return NemoSettings(
        manifest_path=str(nemo_export),
        checkpoint_uri=DEFAULT_CHECKPOINT_URI,
    )


@pytest.fixture
def core_settings() -> TrainerSettings:
    """Return deterministic core settings with explicitly injected lineage."""
    return TrainerSettings(
        model="fixture-parakeet-base",
        seed=7,
        artifact_uri_prefix="fake://nemo-artifacts",
        image_digest="registry.example/nemo@sha256:" + "a" * 64,
        git_sha="b" * 40,
    )


@pytest.fixture
def manifest() -> dict[str, Any]:
    """Return a minimal contract-valid dataset manifest."""
    return {"dataset_id": "synthetic-nemo-fixture", "content_hash": DATASET_HASH}
