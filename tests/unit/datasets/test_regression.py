"""Tests for regression-set accumulation and persistence."""

from __future__ import annotations

import json
from pathlib import Path

from services.datasets import RegressionStore, ReviewedItem, is_reviewed_failure


def test_fixture_failure_labels_accumulate_every_reviewed_failure(
    reviewed_items: tuple[ReviewedItem, ...],
) -> None:
    """Ten fixture failures across five calls enter the regression set once."""
    store = RegressionStore()
    records = store.accumulate(reviewed_items, created_at="2026-11-25T00:00:00Z")
    assert len(records) == 10
    assert store.call_ids() == tuple(f"call-{number:04d}" for number in range(2, 7))
    assert {record.routing for record in records} == {
        "stt_wrong",
        "parser_wrong",
        "unusable_audio",
    }
    assert [record.source_id for record in records] == sorted(
        record.source_id for record in records
    )


def test_repeated_review_and_clean_turns_do_not_duplicate(
    reviewed_items: tuple[ReviewedItem, ...],
) -> None:
    """Deduping is by call and turn; non-failures are not regression rows."""
    store = RegressionStore()
    first = store.accumulate(reviewed_items, created_at="2026-11-25T00:00:00Z")
    second = store.accumulate(reviewed_items, created_at="2026-11-26T00:00:00Z")
    assert second == first
    assert is_reviewed_failure("both_fine") is False
    assert is_reviewed_failure("clean") is False
    assert is_reviewed_failure("parser_wrong") is True
    added = [
        item for item in reviewed_items if is_reviewed_failure(item.failure_type or item.routing)
    ]
    assert all(item.failure_type not in ("clean", "both_fine") for item in added)


def test_jsonl_round_trip_preserves_records(
    reviewed_items: tuple[ReviewedItem, ...], tmp_path: Path
) -> None:
    """A written regression set loads back byte-equivalently as records."""
    destination = tmp_path / "regression.jsonl"
    store = RegressionStore()
    store.accumulate(reviewed_items, created_at="2026-11-25T00:00:00Z")
    store.write(destination)
    loaded = RegressionStore.load(destination)
    assert loaded.records() == store.records()
    assert len(destination.read_text(encoding="utf-8").splitlines()) == 10
    assert all(
        set(json.loads(line)) >= {"call_id", "turn_index", "routing", "slots"}
        for line in destination.read_text(encoding="utf-8").splitlines()
    )
