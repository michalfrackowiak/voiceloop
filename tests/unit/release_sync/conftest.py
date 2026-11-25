"""Deterministic fake-only fixtures for the P3-06 sync agent."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import pytest

from services.release.sync import (
    AliasSyncAgent,
    ArtifactRef,
    CollectingLogger,
    FakeArtifactStore,
    FakeChampionRegistry,
    FakeServingStack,
    HealthProbe,
    ReleaseSyncSettings,
    ServingStateProbe,
)


@dataclass
class World:
    """One wired agent plus access to every injected fake."""

    registry: FakeChampionRegistry
    store: FakeArtifactStore
    serving: FakeServingStack
    probe: HealthProbe
    logger: CollectingLogger
    agent: AliasSyncAgent

    def publish(self, version: str, content: bytes = b"synthetic-artifact") -> str:
        """Bind a fake champion version and return its content hash."""
        uri = f"fake://mlflow/{version}"
        digest = self.store.put(uri, content)
        self.registry.bind(version, ArtifactRef(uri, digest))
        return digest


@pytest.fixture
def world_factory(tmp_path: Path) -> Callable[..., World]:
    """Return a builder for isolated fake worlds rooted at ``tmp_path``."""

    def build(
        *,
        stt_version: str | None = "v1",
        parser_version: str | None = "v1",
        probe: HealthProbe | None = None,
        temp_root: Path | None = tmp_path,
    ) -> World:
        registry = FakeChampionRegistry()
        store = FakeArtifactStore()
        serving = FakeServingStack(stt_version=stt_version, parser_version=parser_version)
        state_probe = probe or ServingStateProbe(serving)
        logger = CollectingLogger()
        settings = ReleaseSyncSettings(temp_root=temp_root)
        agent = AliasSyncAgent(
            settings=settings,
            registry=registry,
            artifacts=store,
            serving=serving,
            probe=state_probe,
            logger=logger,
        )
        return World(registry, store, serving, state_probe, logger, agent)

    return build


@pytest.fixture
def world(world_factory: Callable[..., World]) -> World:
    """Return the default fake world with both targets serving ``v1``."""
    return world_factory()
