"""Shared fixtures and a contract-valid dummy trainer for core tests."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pytest

from services.trainers.core import (
    TrainerPlugin,
    TrainerResult,
    TrainerSettings,
    build_trainer_result,
)

DATASET_HASH = "c" * 64


@pytest.fixture
def manifest() -> dict[str, Any]:
    """Return a minimal trainer-contract-valid dataset manifest."""
    return {"dataset_id": "synthetic-parser", "content_hash": DATASET_HASH}


@pytest.fixture
def settings() -> TrainerSettings:
    """Return deterministic fake settings with real-shaped lineage values."""
    return TrainerSettings(
        image_digest="registry.example/parser@sha256:" + "a" * 64,
        git_sha="b" * 40,
        artifact_uri_prefix="fake://trainer-artifacts",
    )


class DummyTrainer(TrainerPlugin):
    """Minimal plugin that returns a contract-valid fake adapter result."""

    @property
    def name(self) -> str:
        """Return the registered dummy plugin name."""
        return "dummy"

    def train(self, manifest: Mapping[str, Any], settings: TrainerSettings) -> TrainerResult:
        """Return a fake merged-adapter URI and one training metric."""
        return build_trainer_result(
            manifest,
            settings,
            artifact_uri=f"{settings.artifact_uri_prefix}/dummy-adapter",
            metrics={"train_loss": 0.25},
        )
