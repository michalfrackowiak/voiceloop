"""P3-07: deferred candidate/champion execution and call-path isolation proof."""

from __future__ import annotations

import json
from collections.abc import Callable
from datetime import timedelta

from services.release.shadow import ModelSource, ResultStatus
from tests.unit.release_shadow.conftest import World

PII_TRAP = "guest-name@example.com card 4111111111111111"


def test_slow_failing_candidate_is_deferred_outside_the_call_path(
    world_factory: Callable[..., World],
) -> None:
    world = world_factory(
        candidate_fail=True,
        candidate_steps=30.0,
        champion_steps=10.0,
        error_message=PII_TRAP,
    )
    start = world.clock.now()
    world.add_turn("call-1", 0)

    receipt = world.mirror.mirror(turn=world.add_turn("call-2", 0))

    assert receipt.accepted
    assert receipt.elapsed_seconds <= world.settings.mirror_budget_ms / 1000.0
    assert world.clock.now() == start
    assert world.candidate.calls == []
    assert world.champion.calls == []
    assert world.sink.pending() == 1

    assert world.drain() == 1

    results = world.runner.results
    assert len(world.candidate.calls) == 1
    assert len(world.champion.calls) == 1
    assert world.clock.now() - start == timedelta(seconds=40)
    assert [result.source for result in results] == [
        ModelSource.CANDIDATE,
        ModelSource.CHAMPION,
    ]
    assert [result.status for result in results] == [
        ResultStatus.FAILED,
        ResultStatus.OK,
    ]
    assert results[0].error_type == "RuntimeError"
    assert world.audit.events_named("shadow.run")[0].counts["failures"] == 1


def test_result_audit_events_carry_source_artifact_time_and_synthetic_flag(
    world_factory: Callable[..., World],
) -> None:
    world = world_factory(candidate_fail=True, error_message=PII_TRAP)
    turn_zero = world.add_turn("call-1", 0, synthetic=True)
    turn_one = world.add_turn("call-1", 1, synthetic=True)
    world.mirror.mirror(turn_zero)
    world.mirror.mirror(turn_one)
    world.drain()

    events = world.audit.events_named("shadow.result")
    assert len(events) == 4
    for event, result in zip(events, world.runner.results, strict=True):
        assert event.source == result.source.value
        assert event.artifact_uri == result.artifact_uri
        assert event.model_id == result.model_id
        assert event.synthetic is True
        assert "T" in event.observed_at
        assert event.counts["field_count"] == len(result.field_digests)

    payload = json.dumps([event.to_fields() for event in world.audit.events])
    assert "4111111111111111" not in payload
    assert "guest-name@example.com" not in payload


def test_drain_is_bounded_per_batch(world_factory: Callable[..., World]) -> None:
    world = world_factory(max_turns_per_drain=2)
    for index in range(3):
        world.mirror.mirror(world.add_turn("call-1", index))

    assert world.drain() == 2
    assert world.sink.pending() == 1
    assert world.drain() == 1
    assert world.sink.pending() == 0
    assert len(world.runner.results) == 6


def test_results_store_digests_not_raw_booking_values(
    world_factory: Callable[..., World],
) -> None:
    world = world_factory()
    fields = {"guest_name": "Ada Fixture", "contact": {"email": "ada.fixture@example.com"}}
    world.mirror.mirror(world.add_turn("call-1", 0, fields=fields, champion_fields=fields))
    world.drain()

    payload = json.dumps([result.__dict__ for result in world.runner.results], default=str)
    assert "Ada Fixture" not in payload
    assert "ada.fixture@example.com" not in payload
    assert all(
        len(value) == 64
        for result in world.runner.results
        for value in result.field_digests.values()
    )
