"""Tests for typed trainer settings and the contract config projection."""

from __future__ import annotations

import pytest
from pydantic import ValidationError as PydanticValidationError

from services.trainers.core import TrainerSettings


def test_settings_read_only_prefixed_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Every setting is sourced from VOICELOOP_TRAINERS_* variables."""
    digest = "registry.example/parser@sha256:" + "d" * 64
    monkeypatch.setenv("VOICELOOP_TRAINERS_IMAGE_DIGEST", digest)
    monkeypatch.setenv("VOICELOOP_TRAINERS_GIT_SHA", "e" * 40)
    monkeypatch.setenv("VOICELOOP_TRAINERS_ARTIFACT_URI_PREFIX", "s3://bucket/models")
    monkeypatch.setenv("VOICELOOP_TRAINERS_MODEL", "base-model-from-settings")
    monkeypatch.setenv("VOICELOOP_TRAINERS_SEED", "7")
    monkeypatch.setenv("VOICELOOP_TRAINERS_PARAMS", '{"epochs": 2, "lora_alpha": 8}')

    settings = TrainerSettings()
    assert settings.image_digest == digest
    assert settings.git_sha == "e" * 40
    assert settings.artifact_uri_prefix == "s3://bucket/models"
    assert settings.as_contract_config("parser-lora") == {
        "trainer": "parser-lora",
        "model": "base-model-from-settings",
        "seed": 7,
        "params": {"epochs": 2, "lora_alpha": 8},
    }


def test_contract_config_omits_unset_optional_fields() -> None:
    """A minimal config keeps only trainer and seed when no model is set."""
    settings = TrainerSettings(seed=3)
    assert settings.as_contract_config("dummy") == {"trainer": "dummy", "seed": 3}


def test_unpinned_image_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    """A mutable tag cannot satisfy the pinned-image requirement."""
    monkeypatch.setenv("VOICELOOP_TRAINERS_IMAGE_DIGEST", "registry.example/parser:latest")
    with pytest.raises(PydanticValidationError, match="image_digest"):
        TrainerSettings()
