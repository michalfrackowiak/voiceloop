"""Deterministic hypothesis-to-correction edit distance tests."""

from __future__ import annotations

from services.review_adapter.metrics.edit_distance import (
    compute_edit_distances,
    levenshtein,
    summarize_edit_distances,
)
from tests.unit.review_adapter_metrics.conftest import make_result


def test_levenshtein_known_values() -> None:
    """Unit-cost insert, delete and substitute operations match reference cases."""
    assert levenshtein("kitten", "sitting") == 3
    assert levenshtein("", "abc") == 3
    assert levenshtein("abc", "") == 3
    assert levenshtein("", "") == 0
    assert levenshtein("café", "cafe") == 1
    assert levenshtein("same", "same") == 0


def test_compute_and_summarize_edit_distances() -> None:
    """Per-item distances keep lengths and lineage but never transcript text."""
    results = (
        make_result("event-1", hypothesis="kitten", corrected_transcript="sitting"),
        make_result("event-2", hypothesis="", corrected_transcript="abc"),
    )

    items = compute_edit_distances(results)
    summary = summarize_edit_distances(items)

    assert [(item.event_id, item.distance) for item in items] == [
        ("event-1", 3),
        ("event-2", 3),
    ]
    assert items[0].hypothesis_length == 6
    assert items[0].corrected_length == 7
    assert summary.count == 2
    assert summary.total_distance == 6
    assert summary.mean_distance == 3.0
    assert summary.median_distance == 3.0
    assert summary.min_distance == 3
    assert summary.max_distance == 3
    assert not hasattr(items[0], "corrected_transcript")


def test_empty_distance_summary_has_null_statistics() -> None:
    """No fake results means no descriptive statistics."""
    summary = summarize_edit_distances(())

    assert summary.count == 0
    assert summary.total_distance == 0
    assert summary.mean_distance is None
