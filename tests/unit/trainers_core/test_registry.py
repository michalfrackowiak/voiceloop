"""Tests for named trainer plugin registration and lookup."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pytest

from services.trainers.core import (
    DuplicatePluginError,
    PluginNotFoundError,
    TrainerPlugin,
    TrainerRegistry,
    TrainerRegistryError,
    TrainerResult,
    TrainerSettings,
    build_trainer_result,
)

from .conftest import DummyTrainer


def test_register_get_and_list() -> None:
    """Plugins are retrievable by name and listed deterministically."""
    registry = TrainerRegistry()
    dummy = DummyTrainer()
    registry.register(dummy)
    registry.register(_OtherTrainer())
    assert registry.get("dummy") is dummy
    assert registry.list() == ["dummy", "other"]


def test_duplicate_registration_is_rejected() -> None:
    """A name cannot be silently overwritten by default."""
    registry = TrainerRegistry()
    registry.register(DummyTrainer())
    with pytest.raises(DuplicatePluginError):
        registry.register(DummyTrainer())


def test_replace_can_override_a_name() -> None:
    """Explicit replacement swaps the plugin instance."""
    registry = TrainerRegistry()
    first = DummyTrainer()
    second = DummyTrainer()
    registry.register(first)
    registry.register(second, replace=True)
    assert registry.get("dummy") is second


def test_unknown_plugin_raises() -> None:
    """Lookup failures use the shared VoiceLoop error hierarchy."""
    registry = TrainerRegistry()
    with pytest.raises(PluginNotFoundError):
        registry.get("missing")


def test_empty_plugin_name_is_rejected() -> None:
    """An unnamed plugin cannot be registered."""
    registry = TrainerRegistry()
    with pytest.raises(TrainerRegistryError, match="non-empty"):

        class NamelessTrainer(TrainerPlugin):
            """Plugin with an empty name."""

            @property
            def name(self) -> str:
                """Return an empty name."""
                return ""

            def train(
                self, manifest: Mapping[str, Any], settings: TrainerSettings
            ) -> TrainerResult:
                """Never called in this test."""
                raise AssertionError("unexpected train call")

        registry.register(NamelessTrainer())


class _OtherTrainer(TrainerPlugin):
    """Second plugin used to check sorted listings."""

    @property
    def name(self) -> str:
        """Return the registered name."""
        return "other"

    def train(self, manifest: Mapping[str, Any], settings: TrainerSettings) -> TrainerResult:
        """Return an unused result."""
        return build_trainer_result(manifest, settings, artifact_uri="fake://other")
