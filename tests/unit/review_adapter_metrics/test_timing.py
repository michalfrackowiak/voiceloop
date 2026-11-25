"""Per-item timing tests using only injected fake timestamps."""

from __future__ import annotations

import pytest

from services.review_adapter.metrics.timing import (
    compute_item_times,
    elapsed_seconds,
    summarize_times,
)
from tests.unit.review_adapter_metrics.conftest import make_result
from voiceloop_common import ValidationError


def test_elapsed_seconds_comes_from_recorded_timestamps() -> None:
    """Elapsed time is computed from start/completion fields, not the wall clock."""
    result = make_result(duration_seconds=12.5)

    assert elapsed_seconds(result) == 12.5


def test_elapsed_seconds_rejects_reversed_timestamps() -> None:
    """A completion before its start cannot yield a negative review time."""
    result = make_result(duration_seconds=-1)

    with pytest.raises(ValidationError, match="precedes"):
        elapsed_seconds(result)


def test_compute_and_summarize_times() -> None:
    """Per-item records retain pseudonymous lineage; summary stats are deterministic."""
    results = (
        make_result("event-1", duration_seconds=10),
        make_result("event-2", duration_seconds=20),
        make_result("event-3", duration_seconds=30),
    )

    items = compute_item_times(results)
    summary = summarize_times(items)

    assert [(item.event_id, item.seconds) for item in items] == [
        ("event-1", 10.0),
        ("event-2", 20.0),
        ("event-3", 30.0),
    ]
    assert summary.count == 3
    assert summary.total_seconds == 60.0
    assert summary.mean_seconds == 20.0
    assert summary.median_seconds == 20.0
    assert summary.min_seconds == 10.0
    assert summary.max_seconds == 30.0


def test_empty_timing_summary_has_null_statistics() -> None:
    """No fake results means no descriptive statistics."""
    summary = summarize_times(())

    assert summary.count == 0
    assert summary.total_seconds == 0.0
    assert summary.mean_seconds is None
    assert summary.median_seconds is None
