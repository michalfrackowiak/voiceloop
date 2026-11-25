"""Injectable seam and fixture-backed fake source tests."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from services.review_adapter.metrics.events import (
    BLIND_REVIEWER_ID,
    PREFILLED_REVIEWER_ID,
    FixedClock,
    FixtureReviewResultSource,
    JsonFileReportSink,
    ResultPairSource,
    SequenceClock,
    SequenceReviewResultSource,
    StoreReviewResultSource,
    parse_utc,
    rfc3339,
)
from services.review_adapter.metrics.timing import elapsed_seconds
from services.review_adapter.results import InMemoryReviewStore
from tests.unit.review_adapter_metrics.conftest import (
    FIXTURES_PATH,
    make_result,
)
from voiceloop_common import ValidationError


def test_clocks_are_injected_and_deterministic() -> None:
    """Fake clocks return only supplied timestamps, never the system clock."""
    fixed = FixedClock("2026-11-25T10:00:00Z")
    sequence = SequenceClock("2026-11-25T10:00:00Z", step_seconds=72)

    assert fixed() == datetime(2026, 11, 25, 10, 0, tzinfo=UTC)
    assert fixed() == datetime(2026, 11, 25, 10, 0, tzinfo=UTC)
    assert sequence() == datetime(2026, 11, 25, 10, 0, tzinfo=UTC)
    assert sequence() == datetime(2026, 11, 25, 10, 1, 12, tzinfo=UTC)
    assert rfc3339(parse_utc("2026-11-25T10:00:00Z")) == "2026-11-25T10:00:00Z"


def test_fixture_source_replays_every_item_as_two_fake_variants(
    fixture_source: FixtureReviewResultSource, repo_root: Path
) -> None:
    """Every committed synthetic turn yields exactly one blind and one prefilled fake."""
    results = tuple(fixture_source.iter_results())
    expected_rows = sum(
        bool(line.strip())
        for line in (repo_root / FIXTURES_PATH).read_text(encoding="utf-8").splitlines()
    )

    assert len(results) == expected_rows * 2
    assert sum(bool(result.blind) for result in results) == expected_rows
    assert sum(not result.blind for result in results) == expected_rows
    assert len({result.event_id for result in results}) == expected_rows
    assert {result.tool_name for result in results} == {"fake_in_memory"}
    assert all("fake_reviewer_stand_in" in result.warnings for result in results)
    assert all(elapsed_seconds(result) == 72.0 for result in results)
    assert {r.reviewer_id for r in results if r.blind} == {BLIND_REVIEWER_ID}
    assert {r.reviewer_id for r in results if not r.blind} == {PREFILLED_REVIEWER_ID}


def test_store_and_sequence_sources_are_frozen() -> None:
    """Store adapters preserve stable result order without reading new inputs."""
    result = make_result()
    store = InMemoryReviewStore()
    store.save(result)
    stored = tuple(StoreReviewResultSource(store).iter_results())
    frozen = tuple(SequenceReviewResultSource(stored).iter_results())

    assert stored == (result,)
    assert frozen == (result,)


def test_result_pair_source_never_invents_a_partner() -> None:
    """An event without exactly one result of each variant stays unpaired."""
    blind = make_result("event-1", blind=True)
    prefilled = make_result("event-1", blind=False)
    lone = make_result("event-2", blind=True)
    source = ResultPairSource(SequenceReviewResultSource((blind, prefilled, lone)))

    pairs = tuple(source.iter_pairs())

    assert len(pairs) == 1
    assert pairs[0].event_id == "event-1"
    assert pairs[0].blind.reviewer_id == blind.reviewer_id
    assert pairs[0].prefilled.reviewer_id == prefilled.reviewer_id


def test_json_sink_writes_sorted_utf8_json(tmp_path: Path) -> None:
    """The only file-writing sink produces deterministic, sorted JSON."""
    sink = JsonFileReportSink(tmp_path / "nested" / "report.json")

    destination = sink.write({"z": 1, "a": "value"})

    assert destination == str(tmp_path / "nested" / "report.json")
    assert (tmp_path / "nested" / "report.json").read_text(encoding="utf-8") == (
        '{\n  "a": "value",\n  "z": 1\n}\n'
    )


def test_parse_utc_rejects_non_timestamps() -> None:
    """Malformed fake timestamps fail closed rather than reaching the wall clock."""
    with pytest.raises(ValidationError, match="invalid fake timestamp"):
        parse_utc("not-a-timestamp")


def test_sequence_clock_does_not_use_wall_clock() -> None:
    """Sequence steps are relative to the injected start only."""
    start = datetime(2030, 1, 1, tzinfo=UTC)
    clock = SequenceClock(start, step_seconds=0.5)

    assert clock() == start
    assert clock() == start + timedelta(seconds=0.5)
