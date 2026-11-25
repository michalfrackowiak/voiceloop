"""Shared deterministic windows, settings and fakes for P4-02 tests."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from infra.pipelines.drift import (
    DriftSettings,
    OutcomeWindow,
    TriageScoreWindow,
)
from voiceloop_common import Clock, frozen_clock

FIXED_NOW = datetime(2026, 11, 25, 12, 0, tzinfo=UTC)

BASELINE_SCORES = (0.05, 0.06, 0.05, 0.07, 0.06, 0.05)
DRIFTED_SCORES = (0.85, 0.86, 0.84, 0.87, 0.85, 0.86)
BASELINE_OUTCOMES = (
    "confirmed",
    "confirmed",
    "confirmed",
    "confirmed",
    "edited_by_staff",
    "confirmed",
)
DRIFTED_OUTCOMES = (
    "abandoned",
    "abandoned",
    "edited_by_staff",
    "abandoned",
    "transferred",
    "abandoned",
)


@pytest.fixture
def drift_settings() -> DriftSettings:
    """Return small-window, local-safe settings."""
    return DriftSettings(
        min_window_samples=4,
        min_new_verified=5,
        score_buckets=10,
        score_mean_delta_threshold=0.05,
        score_tv_distance_threshold=0.20,
        outcome_rate_delta_threshold=0.10,
    )


@pytest.fixture
def fixed_clock() -> Clock:
    """Return a frozen UTC clock."""
    return frozen_clock(FIXED_NOW)


@pytest.fixture
def drifted_score_window() -> TriageScoreWindow:
    """Return a synthetic score window with an explicit positive shift."""
    return TriageScoreWindow(baseline=BASELINE_SCORES, current=DRIFTED_SCORES)


@pytest.fixture
def stable_score_window() -> TriageScoreWindow:
    """Return a synthetic score window with no shift."""
    return TriageScoreWindow(baseline=BASELINE_SCORES, current=BASELINE_SCORES)


@pytest.fixture
def drifted_outcome_window() -> OutcomeWindow:
    """Return a synthetic outcome window with an explicit rate shift."""
    return OutcomeWindow(baseline=BASELINE_OUTCOMES, current=DRIFTED_OUTCOMES)


@pytest.fixture
def stable_outcome_window() -> OutcomeWindow:
    """Return a synthetic outcome window with no shift."""
    return OutcomeWindow(baseline=BASELINE_OUTCOMES, current=BASELINE_OUTCOMES)
