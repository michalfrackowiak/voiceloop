"""P3-07: internal comparison artifact, promotion label and PII safety."""

from __future__ import annotations

import copy
import json
from collections.abc import Callable
from pathlib import Path

import pytest

from services.release.shadow import (
    CollectingReportSink,
    ComparisonData,
    FileReportSink,
    build_shadow_report,
    validate_shadow_report,
    write_shadow_report,
)
from tests.unit.release_shadow.conftest import World
from voiceloop_common import ValidationError, load_schema

PII_FIELDS = {
    "guest_name": "Ada Fixture",
    "contact": {
        "email": "ada.fixture@example.com",
        "phone": "+1 555 010 0001",
    },
    "card_number": "4111111111111111",
}


def _comparison(world: World) -> ComparisonData:
    """Score one scorable call with PII-shaped slot values."""
    world.mirror.mirror(
        world.add_turn(
            "call-a",
            0,
            fields=PII_FIELDS,
            champion_fields={"guest_name": "Ada Fixture"},
            synthetic=True,
        )
    )
    world.drain()
    world.add_outcome("call-a", "confirmed", PII_FIELDS)
    return build_data(world)


def build_data(world: World) -> ComparisonData:
    """Build comparison data from the world's recorded results."""
    from services.release.shadow import score_comparison

    return score_comparison(
        settings=world.settings,
        results=world.runner.results,
        outcome_source=world.outcomes,
        mirror_stats=world.stats,
        candidate=world.candidate,
        champion=world.champion,
        synthetic=True,
    )


def test_report_is_labelled_and_contains_no_raw_guest_data(
    world_factory: Callable[..., World],
) -> None:
    world = world_factory(error_message="secret card 4111111111111111")
    data = _comparison(world)

    report = build_shadow_report(data, "reports/release/shadow/comparison.json")
    payload = json.dumps(report)

    for secret in (
        "Ada Fixture",
        "ada.fixture@example.com",
        "+1 555 010 0001",
        "4111111111111111",
    ):
        assert secret not in payload
    assert report["promotion_decision"] == "not_evaluated"
    assert report["promotion"]["alias_writes"] == 0
    assert report["promotion"]["champion_flips"] == 0
    assert report["promotion"]["gate_edits"] == 0
    assert report["promotion"]["thresholds_changed"] is False
    assert "SYNTHETIC" in report["status"]
    assert report["provenance"]["synthetic"] is True
    assert report["provenance"]["real_traffic"] is False
    assert report["provenance"]["audio_bytes_logged"] is False
    assert report["provenance"]["join_key"] == "call_id"
    assert report["shadow_evidence"]["turns_observed"] == 1
    assert report["deltas"]["outcome_call_accuracy"] == 1.0
    assert report["isolation"]["within_budget"] is True
    assert report["metrics"]["candidate"]["outcome_error_rate"] == 0.0


def test_shadow_evidence_matches_committed_schema_shape(
    world_factory: Callable[..., World],
) -> None:
    world = world_factory()
    data = _comparison(world)
    report = build_shadow_report(data, "fake://shadow/comparison.json")
    required = load_schema("eval_report")["$defs"]["ShadowEvidence"]["required"]

    evidence = report["shadow_evidence"]
    assert all(key in evidence for key in required)
    assert isinstance(evidence["report_uri"], str) and evidence["report_uri"]
    assert isinstance(evidence["turns_observed"], int)
    assert isinstance(evidence["outcome_error_rate"], (int, float))


def test_validation_rejects_promotion_claims_and_disallowed_fields(
    world_factory: Callable[..., World],
) -> None:
    world = world_factory()
    report = build_shadow_report(_comparison(world), "fake://shadow/comparison.json")

    claimed = copy.deepcopy(report)
    claimed["promotion_decision"] = "pass"
    with pytest.raises(ValidationError):
        validate_shadow_report(claimed)

    aliased = copy.deepcopy(report)
    aliased["promotion"]["alias_writes"] = 1
    with pytest.raises(ValidationError):
        validate_shadow_report(aliased)

    leaked = copy.deepcopy(report)
    leaked["metrics"]["candidate"]["transcript"] = "Ada Fixture"
    with pytest.raises(ValidationError):
        validate_shadow_report(leaked)

    negative = copy.deepcopy(report)
    negative["sample_counts"]["turns_observed"] = -1
    with pytest.raises(ValidationError):
        validate_shadow_report(negative)


def test_report_sinks_write_and_collect(
    tmp_path: Path, world_factory: Callable[..., World]
) -> None:
    world = world_factory()
    report = build_shadow_report(_comparison(world), "fake://shadow/comparison.json")
    destination = tmp_path / "nested" / "comparison.json"

    written = FileReportSink().write(report, str(destination))
    reloaded = json.loads(destination.read_text(encoding="utf-8"))
    collector = CollectingReportSink()
    collected_uri = collector.write(report, "fake://shadow/collected.json")

    assert written == str(destination.resolve())
    assert reloaded == report
    assert collected_uri == "fake://shadow/collected.json"
    assert collector.reports == [report]
    assert write_shadow_report(report, destination).exists()
