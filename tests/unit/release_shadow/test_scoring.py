"""P3-07: outcome-based scoring joined by call_id only."""

from __future__ import annotations

from collections.abc import Callable, Sequence

from services.release.shadow import (
    ComparisonData,
    ModelResult,
    ModelSource,
    OutcomeRecord,
    score_comparison,
)
from tests.unit.release_shadow.conftest import World


def _score(world: World, results: Sequence[ModelResult] | None = None) -> ComparisonData:
    """Score the world's fake results with its injected outcome source."""
    return score_comparison(
        settings=world.settings,
        results=world.runner.results if results is None else results,
        outcome_source=world.outcomes,
        mirror_stats=world.stats,
        candidate=world.candidate,
        champion=world.champion,
    )


def test_outcomes_join_only_by_call_id(world_factory: Callable[..., World]) -> None:
    world = world_factory()
    world.mirror.mirror(
        world.add_turn("call-a", 0, fields={"guests": 2}, champion_fields={"guests": 2})
    )
    world.mirror.mirror(world.add_turn("call-b", 0, fields={"guests": 2}))
    world.drain()
    world.add_outcome("call-a", "confirmed", {"guests": 2})

    data = _score(world)

    assert world.outcomes.lookups == ["call-a", "call-b"]
    assert data.sample_counts["calls_with_outcome"] == 1
    assert data.sample_counts["calls_missing_outcome"] == 1
    assert data.sample_counts["calls_scored"] == 1
    assert data.sample_counts["turns_observed"] == 2

    raw_name = "Ada Fixture"
    record = OutcomeRecord.from_payload(
        "call-a",
        {"status": "confirmed", "final_record": {"guest_name": raw_name}},
        world.settings.ignored_outcome_fields(),
    )
    assert raw_name not in str(record)
    assert raw_name not in record.field_digests.values()
    assert "call_id" not in record.field_digests
    assert "status" not in record.field_digests


def test_deltas_and_sample_counts_with_perfect_candidate(
    world_factory: Callable[..., World],
) -> None:
    world = world_factory()
    full = {"guests": 2, "room_type": "standard"}
    world.mirror.mirror(world.add_turn("call-a", 0, fields=full, champion_fields=full))
    world.mirror.mirror(
        world.add_turn("call-b", 0, fields=full, champion_fields={"room_type": "standard"})
    )
    world.drain()
    world.add_outcome("call-a", "confirmed", full)
    world.add_outcome("call-b", "edited_by_staff", full)

    data = _score(world)

    assert data.metrics["candidate"]["outcome_call_accuracy"] == 1.0
    assert data.metrics["champion"]["outcome_call_accuracy"] == 0.5
    assert data.deltas["outcome_call_accuracy"] == 0.5
    assert data.metrics["candidate"]["outcome_error_rate"] == 0.0
    assert data.metrics["champion"]["outcome_error_rate"] == 0.5
    assert data.deltas["outcome_error_rate"] == -0.5
    assert data.metrics["candidate"]["outcome_field_accuracy"] == 1.0
    assert data.metrics["champion"]["outcome_field_accuracy"] == 0.75
    assert data.sample_counts == {
        "turns_observed": 2,
        "turns_mirrored": 2,
        "turns_dropped": 0,
        "sink_errors": 0,
        "results_candidate": 2,
        "results_champion": 2,
        "failed_results": 0,
        "calls_observed": 2,
        "calls_with_outcome": 2,
        "calls_scored": 2,
        "calls_missing_outcome": 0,
        "calls_excluded_by_status": 0,
        "calls_empty_final_record": 0,
        "calls_missing_model_result": 0,
    }
    assert [call.call_id for call in data.scored_calls] == ["call-a", "call-b"]


def test_excluded_status_and_missing_model_result_do_not_divide_by_zero(
    world_factory: Callable[..., World],
) -> None:
    world = world_factory()
    world.mirror.mirror(world.add_turn("call-a", 0, fields={"guests": 2}))
    world.mirror.mirror(world.add_turn("call-b", 0, fields={"guests": 2}))
    world.drain()
    world.add_outcome("call-a", "abandoned", {"guests": 2})
    world.add_outcome("call-b", "confirmed", {"guests": 2})

    only_candidate = [
        result for result in world.runner.results if result.source is ModelSource.CANDIDATE
    ]
    data = _score(world, results=only_candidate)

    assert data.sample_counts["calls_excluded_by_status"] == 1
    assert data.sample_counts["calls_missing_model_result"] == 1
    assert data.sample_counts["calls_scored"] == 0
    assert data.metrics["candidate"]["outcome_call_accuracy"] == 0.0
    assert data.metrics["champion"]["outcome_error_rate"] == 0.0
    assert data.deltas == {
        "outcome_call_accuracy": 0.0,
        "outcome_error_rate": 0.0,
        "outcome_field_accuracy": 0.0,
    }


def test_latest_turn_per_call_wins(world_factory: Callable[..., World]) -> None:
    world = world_factory()
    world.mirror.mirror(
        world.add_turn("call-a", 0, fields={"guests": 2}, champion_fields={"guests": 2})
    )
    world.mirror.mirror(
        world.add_turn("call-a", 1, fields={"guests": 4}, champion_fields={"guests": 2})
    )
    world.drain()
    world.add_outcome("call-a", "confirmed", {"guests": 4})

    data = _score(world)

    assert data.sample_counts["calls_scored"] == 1
    assert data.metrics["candidate"]["outcome_call_accuracy"] == 1.0
    assert data.metrics["champion"]["outcome_call_accuracy"] == 0.0
    assert data.deltas["outcome_call_accuracy"] == 1.0


def test_empty_final_record_is_not_scored(world_factory: Callable[..., World]) -> None:
    world = world_factory()
    world.mirror.mirror(world.add_turn("call-a", 0, fields={}))
    world.drain()
    world.add_outcome("call-a", "confirmed", {})

    data = _score(world)

    assert data.sample_counts["calls_empty_final_record"] == 1
    assert data.sample_counts["calls_scored"] == 0
    assert data.metrics["candidate"]["outcome_field_accuracy"] == 0.0
