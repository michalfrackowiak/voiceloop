"""Tests for the trainer plugin ABC and contract result helper."""

from __future__ import annotations

from collections.abc import Mapping
from inspect import isabstract
from typing import Any

import pytest

from services.trainers.core import (
    TRAINER_CONTRACT_SCHEMA,
    TRAINER_CONTRACT_VERSION,
    TrainerPlugin,
    TrainerSettings,
    build_trainer_result,
)
from voiceloop_common import ValidationError, validate


def test_plugin_base_is_abstract() -> None:
    """The plugin interface cannot be instantiated without name and train."""
    assert isabstract(TrainerPlugin)


def test_builder_produces_contract_valid_lineage(
    manifest: dict[str, Any], settings: TrainerSettings
) -> None:
    """The composed result validates and carries every lineage field."""
    result = build_trainer_result(
        manifest,
        settings,
        artifact_uri="fake://artifacts/merged-adapter",
        metrics={"train_loss": 0.1},
        run_id="run-1",
    )

    validate(result, TRAINER_CONTRACT_SCHEMA)
    assert result["schema_version"] == TRAINER_CONTRACT_VERSION
    assert result["contract_part"] == "output"
    assert result["run_id"] == "run-1"
    assert result["artifact_uri"] == "fake://artifacts/merged-adapter"
    assert result["metrics"] == {"train_loss": 0.1}
    assert result["lineage"] == {
        "dataset_hash": manifest["content_hash"],
        "git_sha": settings.git_sha,
        "image_digest": settings.image_digest,
    }


def test_builder_generates_run_id(manifest: dict[str, Any], settings: TrainerSettings) -> None:
    """A run ID is generated when the plugin does not supply one."""
    first = build_trainer_result(manifest, settings, artifact_uri="fake://a")
    second = build_trainer_result(manifest, settings, artifact_uri="fake://a")
    assert first["run_id"] and second["run_id"]
    assert first["run_id"] != second["run_id"]


def test_builder_rejects_manifest_without_hash(settings: TrainerSettings) -> None:
    """No result may be produced without a dataset hash for lineage."""
    empty_manifest: Mapping[str, Any] = {"dataset_id": "no-hash"}
    with pytest.raises(ValidationError, match="content_hash"):
        build_trainer_result(empty_manifest, settings, artifact_uri="fake://a")
