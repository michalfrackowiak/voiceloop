"""Shared deterministic fixtures for dashboard tests."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from infra.dashboards.data import DashboardData, Metric
from infra.dashboards.settings import DashboardSettings
from voiceloop_common import Clock, frozen_clock

FIXED_NOW = datetime(2026, 11, 25, 12, 0, tzinfo=UTC)
FIXTURES = "tests/fixtures/turns.jsonl"
SNAPSHOT = "infra/dashboards/examples/synthetic-snapshot.json"


@pytest.fixture
def dashboard_settings() -> DashboardSettings:
    """Return local-safe settings using the committed synthetic sources."""
    return DashboardSettings(
        fixtures_path=FIXTURES,
        snapshot_path=SNAPSHOT,
        format="html",
        stale_after_minutes=1440,
    )


@pytest.fixture
def fixed_clock() -> Clock:
    """Return a deterministic UTC clock."""
    return frozen_clock(FIXED_NOW)


def metric_named(data: DashboardData, label: str) -> Metric:
    """Find exactly one metric by its display label."""
    matches = [
        metric for section in data.sections for metric in section.metrics if metric.label == label
    ]
    assert len(matches) == 1, f"expected one metric labelled {label!r}, got {len(matches)}"
    return matches[0]
