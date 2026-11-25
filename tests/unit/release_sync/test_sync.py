"""P3-06: end-to-end fake-stack sync, idempotency and auto-revert."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from services.release.sync import (
    AliasSyncAgent,
    ArtifactRef,
    ChampionSnapshot,
    FakeHealthProbe,
    HealthResult,
    HealthStatus,
    ReleaseSyncSettings,
    ServingStateProbe,
    ServingTarget,
)
from tests.unit.release_sync.conftest import World


def _versions(world: World) -> dict[ServingTarget, str | None]:
    """Return active versions for both fake targets."""
    return {state.target: state.current_version for state in world.serving.states()}


def _last_event(world: World) -> dict[str, object]:
    """Return the structured fields of the latest tick event."""
    events = world.logger.events_named("release_sync.tick")
    assert events
    return events[-1].fields


def test_first_change_swaps_both_targets_and_logs_structured_fields(
    world_factory: Callable[..., World],
) -> None:
    world = world_factory()
    digest = world.publish("v2", b"converted-v2")

    result = world.agent.tick()

    assert result.outcome.value == "swapped"
    assert result.changed
    assert result.alias_version == "v2"
    assert result.artifact_hash == digest
    assert result.previous_version == {ServingTarget.STT: "v1", ServingTarget.PARSER: "v1"}
    assert result.current_version == {ServingTarget.STT: "v2", ServingTarget.PARSER: "v2"}
    assert _versions(world) == {ServingTarget.STT: "v2", ServingTarget.PARSER: "v2"}

    event = _last_event(world)
    assert event["outcome"] == "swapped"
    assert event["alias_version"] == "v2"
    assert event["artifact_hash"] == digest
    assert event["previous_version"] == {"stt": "v1", "parser": "v1"}
    assert event["current_version"] == {"stt": "v2", "parser": "v2"}
    assert event["task_id"] == "P3-06"
    assert "observed_at" in event


def test_unchanged_alias_is_idempotent(world_factory: Callable[..., World]) -> None:
    world = world_factory()
    world.publish("v2")
    assert world.agent.tick().outcome.value == "swapped"

    prepare_count = len(world.serving.fake_adapters[ServingTarget.STT].prepare_calls)
    result = world.agent.tick()

    assert result.outcome.value == "unchanged"
    assert not result.changed
    assert len(world.serving.fake_adapters[ServingTarget.STT].prepare_calls) == prepare_count
    assert world.registry.poll_count == 2
    assert isinstance(world.probe, ServingStateProbe)
    assert world.probe.calls == 1


def test_second_alias_change_swaps_again(world_factory: Callable[..., World]) -> None:
    world = world_factory()
    world.publish("v2")
    world.agent.tick()
    world.publish("v3", b"converted-v3")

    result = world.agent.tick()

    assert result.outcome.value == "swapped"
    assert result.alias_version == "v3"
    assert _versions(world) == {ServingTarget.STT: "v3", ServingTarget.PARSER: "v3"}


def test_bad_hash_fails_before_any_serving_call(world_factory: Callable[..., World]) -> None:
    world = world_factory()
    expected = world.publish("v2", b"good-artifact")
    world.store.put("fake://mlflow/v2", b"corrupted-artifact")

    result = world.agent.tick()

    assert result.outcome.value == "failed"
    assert result.reason == "artifact hash verification failed"
    assert result.error_type == "ValidationError"
    assert result.alias_version == "v2"
    assert result.artifact_hash is None
    assert expected != result.artifact_hash
    assert _versions(world) == {ServingTarget.STT: "v1", ServingTarget.PARSER: "v1"}
    assert world.serving.fake_adapters[ServingTarget.STT].prepare_calls == []


def test_missing_artifact_uri_is_reported(world_factory: Callable[..., World]) -> None:
    world = world_factory()
    digest = world.store.put("fake://mlflow/exists", b"payload")
    world.registry.bind("v2", ArtifactRef("fake://mlflow/missing", digest))

    result = world.agent.tick()

    assert result.outcome.value == "failed"
    assert result.reason == "artifact download failed"
    assert _versions(world) == {ServingTarget.STT: "v1", ServingTarget.PARSER: "v1"}


def test_bound_alias_without_metadata_is_failed(world_factory: Callable[..., World]) -> None:
    world = world_factory()
    world.registry.bind("v2", None)

    result = world.agent.tick()

    assert result.outcome.value == "failed"
    assert result.reason == "bound champion has no artifact metadata"
    assert all(adapter.prepare_calls == [] for adapter in world.serving.fake_adapters.values())


def test_unbound_alias_is_skipped_without_serving_changes(
    world_factory: Callable[..., World],
) -> None:
    world = world_factory()

    skipped = world.agent.tick()

    assert skipped.outcome.value == "skipped"
    assert skipped.alias_state.value == "unbound"
    assert _versions(world) == {ServingTarget.STT: "v1", ServingTarget.PARSER: "v1"}

    world.publish("v2")
    assert world.agent.tick().outcome.value == "swapped"


def test_prepare_failure_reverts_and_reports(world_factory: Callable[..., World]) -> None:
    world = world_factory()
    world.publish("v2")
    world.serving.fake_adapters[ServingTarget.PARSER].fail_prepare = True

    result = world.agent.tick()

    assert result.outcome.value == "failed"
    assert result.reason == "serving prepare failed"
    assert result.error_type == "RuntimeError"
    assert result.previous_version == result.current_version
    assert _versions(world) == {ServingTarget.STT: "v1", ServingTarget.PARSER: "v1"}
    assert _last_event(world)["outcome"] == "failed"


def test_commit_failure_reverts_and_reports(world_factory: Callable[..., World]) -> None:
    world = world_factory()
    world.publish("v2")
    world.serving.fake_adapters[ServingTarget.PARSER].fail_commit = True

    result = world.agent.tick()

    assert result.outcome.value == "failed"
    assert result.reason == "serving commit failed"
    assert _versions(world) == {ServingTarget.STT: "v1", ServingTarget.PARSER: "v1"}
    assert all(adapter.rollback_calls == 1 for adapter in world.serving.fake_adapters.values())


def test_failed_health_check_auto_reverts_exact_previous_versions(
    world_factory: Callable[..., World],
) -> None:
    unhealthy = HealthResult(HealthStatus.UNHEALTHY, ("stt.version",))
    world = world_factory(probe=FakeHealthProbe([unhealthy]))
    digest = world.publish("v2", b"broken-artifact")

    result = world.agent.tick()

    assert result.outcome.value == "reverted"
    assert result.reason == "health check failed: stt.version"
    assert result.artifact_hash == digest
    assert result.current_version == {ServingTarget.STT: "v1", ServingTarget.PARSER: "v1"}
    assert _versions(world) == {ServingTarget.STT: "v1", ServingTarget.PARSER: "v1"}
    assert all(adapter.rollback_calls == 1 for adapter in world.serving.fake_adapters.values())
    assert _last_event(world)["rollback_errors"] == []


def test_health_probe_error_auto_reverts(world_factory: Callable[..., World]) -> None:
    class RaisingProbe:
        def check(self, expected_version: str | None = None) -> HealthResult:
            """Simulate a probe that crashes after the swap."""
            raise RuntimeError("probe unavailable")

    world = world_factory(probe=RaisingProbe())
    world.publish("v2")

    result = world.agent.tick()

    assert result.outcome.value == "failed"
    assert result.reason == "health probe error"
    assert result.error_type == "RuntimeError"
    assert _versions(world) == {ServingTarget.STT: "v1", ServingTarget.PARSER: "v1"}


def test_rollback_error_is_reported_without_raising(world_factory: Callable[..., World]) -> None:
    unhealthy = HealthResult(HealthStatus.UNHEALTHY)
    world = world_factory(probe=FakeHealthProbe([unhealthy]))
    world.publish("v2")
    world.serving.fake_adapters[ServingTarget.PARSER].fail_rollback = True

    result = world.agent.tick()

    assert result.outcome.value == "rollback_failed"
    assert result.rollback_errors == ("parser:RuntimeError",)
    assert _last_event(world)["rollback_errors"] == ["parser:RuntimeError"]


def test_poll_failure_is_reported_without_raising(world_factory: Callable[..., World]) -> None:
    class RaisingRegistry:
        def poll(self) -> ChampionSnapshot:
            """Simulate an unavailable registry."""
            raise RuntimeError("registry unavailable")

    world = world_factory()
    world.agent = AliasSyncAgent(
        settings=ReleaseSyncSettings(temp_root=Path("unused")),
        registry=RaisingRegistry(),
        artifacts=world.store,
        serving=world.serving,
        probe=world.probe,
        logger=world.logger,
    )

    result = world.agent.tick()

    assert result.outcome.value == "failed"
    assert result.reason == "registry poll failed"
    assert result.error_type == "RuntimeError"
    assert result.alias_state.value == "unknown"
    assert _versions(world) == {ServingTarget.STT: "v1", ServingTarget.PARSER: "v1"}


def test_artifact_temp_directory_is_removed(
    world_factory: Callable[..., World], tmp_path: Path
) -> None:
    world = world_factory(temp_root=tmp_path)
    world.publish("v2")

    assert world.agent.tick().outcome.value == "swapped"
    assert list(tmp_path.iterdir()) == []


def test_reset_allows_operator_retry_of_the_same_alias(
    world_factory: Callable[..., World], tmp_path: Path
) -> None:
    world = world_factory()
    probe = ServingStateProbe(world.serving, ["parser.adapter"])
    world.agent = AliasSyncAgent(
        settings=ReleaseSyncSettings(temp_root=tmp_path),
        registry=world.registry,
        artifacts=world.store,
        serving=world.serving,
        probe=probe,
        logger=world.logger,
    )
    world.publish("v2")

    assert world.agent.tick().outcome.value == "reverted"
    assert world.agent.tick().outcome.value == "unchanged"
    world.agent.reset()
    assert world.agent.tick().outcome.value == "reverted"
