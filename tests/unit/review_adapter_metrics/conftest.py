"""Shared fixtures for P2-04; every source is a read-only synthetic fixture fake."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

from services.review_adapter.metrics.events import (
    FixtureReviewResultSource,
    SequenceClock,
)
from services.review_adapter.metrics.settings import ReviewMetricsSettings
from services.review_adapter.results import StoredReviewResult

REPO_ROOT = Path(__file__).resolve().parents[3]
GOLDEN_DIR = "services/datasets/golden/artifacts"
FIXTURES_PATH = "tests/fixtures/turns.jsonl"


def make_result(
    event_id: str = "event-1",
    *,
    call_id: str = "call-1",
    turn_index: int = 0,
    routing: str = "both_fine",
    hypothesis: str = "kitten",
    corrected_transcript: str = "sitting",
    slots: dict[str, Any] | None = None,
    blind: bool = False,
    reviewer_id: str = "reviewer-fixture",
    started_at: datetime | None = None,
    duration_seconds: float = 72.0,
) -> StoredReviewResult:
    """Build a P2-03-style fake result with fully injected timestamps."""
    start = started_at or datetime(2026, 11, 25, 10, 0, tzinfo=UTC)
    return StoredReviewResult(
        schema_version="1.0.0",
        item_id=f"item:{event_id}",
        event_id=event_id,
        call_id=call_id,
        turn_index=turn_index,
        audio_uri=f"memory://{event_id}.wav",
        hypothesis=hypothesis,
        corrected_transcript=corrected_transcript,
        slots=slots if slots is not None else {},
        routing=routing,
        reviewer_id=reviewer_id,
        started_at=start,
        completed_at=start + timedelta(seconds=duration_seconds),
        warnings=("fake_reviewer_stand_in",),
        blind=blind,
        tool_name="fake_in_memory",
    )


@pytest.fixture
def repo_root() -> Path:
    """Return the repository root for read-only input paths."""
    return REPO_ROOT


@pytest.fixture
def metrics_settings(tmp_path: Path) -> ReviewMetricsSettings:
    """Return fixture-only settings whose only write is a temporary report."""
    return ReviewMetricsSettings(output_path=str(tmp_path / "report.json"))


@pytest.fixture
def fixture_source(
    metrics_settings: ReviewMetricsSettings,
) -> FixtureReviewResultSource:
    """Replay committed fixture labels through P2-03 fakes with a fake clock."""
    return FixtureReviewResultSource(
        metrics_settings,
        SequenceClock(
            metrics_settings.start_at,
            step_seconds=metrics_settings.scripted_duration_seconds,
        ),
    )
