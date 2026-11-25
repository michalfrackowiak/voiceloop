"""P3-06: deterministic post-swap health checks."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from services.release.sync import (
    DownloadedArtifact,
    FakeHealthProbe,
    FakeServingStack,
    HealthResult,
    HealthStatus,
    ServingStateProbe,
    ServingTarget,
)
from tests.unit.release_sync.conftest import World


def test_scripted_probe_returns_queued_then_healthy() -> None:
    unhealthy = HealthResult(HealthStatus.UNHEALTHY, ("stt.version",))
    probe = FakeHealthProbe([unhealthy])

    assert probe.check(expected_version="v2") == unhealthy
    assert probe.check(expected_version="v2").healthy
    assert probe.calls == 2
    assert probe.expected_versions == ["v2", "v2"]


def test_state_probe_sees_committed_versions(world_factory: Callable[..., World]) -> None:
    world = world_factory()
    assert isinstance(world.probe, ServingStateProbe)
    assert world.probe.check(expected_version="v1").healthy

    world.publish("v2")
    assert world.agent.tick().outcome.value == "swapped"

    assert world.probe.check(expected_version="v2").healthy
    assert world.probe.check(expected_version="v3") == HealthResult(
        HealthStatus.UNHEALTHY, ("parser.version", "stt.version")
    )


def test_state_probe_detects_staged_residue() -> None:
    serving = FakeServingStack("v1", "v1")
    probe = ServingStateProbe(serving)
    artifact = DownloadedArtifact("fake://v2", Path("tmp"))
    serving.fake_adapters[ServingTarget.STT].prepare("v2", artifact, "d" * 64)

    result = probe.check(expected_version="v1")

    assert result.status is HealthStatus.UNHEALTHY
    assert result.failed_checks == ("stt.staged",)


def test_forced_failure_check_ids() -> None:
    probe = ServingStateProbe(FakeServingStack("v1", "v1"), ["parser.adapter"])

    assert probe.check(expected_version="v1") == HealthResult(
        HealthStatus.UNHEALTHY, ("parser.adapter",)
    )
