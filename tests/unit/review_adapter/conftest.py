"""Shared fixtures for P2-03; all source data is read-only synthetic fixtures."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from schemas.booking import load_booking_schema
from services.review_adapter.interface import ROUTING_LABELS
from services.review_adapter.tool import ScriptedReview
from services.triage.queue import (
    FAILURE_SLICE,
    LowConfidenceSpan,
    QueueItem,
    TurnRecord,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
FIXTURES = REPO_ROOT / "tests" / "fixtures"
CONTRACT = REPO_ROOT / "schemas" / "review_adapter.openapi.yaml"
CLOCK = lambda: datetime(2026, 11, 25, 10, 0, tzinfo=UTC)  # noqa: E731 - deterministic clock
FIXTURE_REVIEWER = "fixture-reviewer"


def fixture_row(case_id: str) -> dict[str, Any]:
    """Load one read-only label row by its named case ID."""
    for line in (FIXTURES / "turns.jsonl").read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        if row.get("case_id") == case_id:
            return row
    raise AssertionError(f"no fixture case {case_id!r}")


def fixture_event(case_id: str) -> dict[str, Any]:
    """Load the read-only turn event referenced by a fixture label row."""
    row = fixture_row(case_id)
    return json.loads((REPO_ROOT / row["event_path"]).read_text(encoding="utf-8"))


def queue_item(case_id: str, *, blind: bool = False) -> QueueItem:
    """Build a real P2-02 queue item from a fixture event, without using its label."""
    event = fixture_event(case_id)
    record = TurnRecord.from_event(event)
    return QueueItem(
        event_id=record.event_id,
        call_id=record.call_id,
        turn_index=record.turn_index,
        audio_uri=record.audio_uri,
        hypothesis=record.hypothesis,
        score=0.5,
        slice=FAILURE_SLICE,
        cluster_id=f"cluster-{case_id}",
        cluster_size=1,
        no_prefill=blind,
        training_consent=record.training_consent,
        signals=(),
        low_confidence_spans=tuple(
            LowConfidenceSpan(span.start_ms, span.end_ms, span.text, span.confidence)
            for span in record.low_confidence_spans(0.60)
        ),
    )


def scripted_review(case_id: str, **overrides: Any) -> ScriptedReview:
    """Build fake reviewer answers from read-only fixture expectations."""
    row = fixture_row(case_id)
    routing = row["failure_type"]
    if routing == "clean":
        routing = "both_fine"
    assert routing in ROUTING_LABELS
    values: dict[str, Any] = {
        "corrected_transcript": row["expected_transcript"],
        "slots": row["expected_slots"],
        "routing": routing,
        "reviewer_id": FIXTURE_REVIEWER,
        "warnings": ("fake_reviewer_stand_in",),
        "duration_seconds": 72.0,
    }
    values.update(overrides)
    return ScriptedReview(**values)


@pytest.fixture
def booking_schema() -> dict[str, Any]:
    """Return the read-only booking schema used to generate stage 2 forms."""
    return load_booking_schema()
