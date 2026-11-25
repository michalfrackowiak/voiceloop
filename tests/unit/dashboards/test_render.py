"""Tests for deterministic HTML and Markdown dashboard rendering."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import cast

import pytest

from infra.dashboards.data import DashboardData, FakeQueueDepthProvider, collect_dashboard
from infra.dashboards.render import SYNTHETIC_BANNER, render_dashboard
from infra.dashboards.settings import DashboardFormat, DashboardSettings
from voiceloop_common import ValidationError

NOW = datetime(2026, 11, 25, 12, 0, tzinfo=UTC)


@pytest.fixture
def sample_data(dashboard_settings: DashboardSettings) -> DashboardData:
    """Return deterministic data with an injected source needing escaping."""
    return collect_dashboard(
        settings=dashboard_settings,
        now=NOW,
        queue_provider=FakeQueueDepthProvider(
            depth=2, source="<script>alert(1)</script>|queue", observed_at=None
        ),
    )


def test_html_render_includes_all_provenance_labels(sample_data: DashboardData) -> None:
    rendered = render_dashboard(sample_data)
    assert rendered.startswith("<!DOCTYPE html>")
    assert SYNTHETIC_BANNER in rendered
    assert "2026-11-25T12:00:00Z" in rendered
    for header in ("Source", "Observed at", "Synthetic", "Stale"):
        assert f"<th>{header}</th>" in rendered
    assert "<td>Failure rate (fixture labels)</td>" in rendered
    assert "tests/fixtures/turns.jsonl" in rendered
    assert "unknown (no timestamp)" in rendered
    assert "10/36 (27.8%)" in rendered


def test_html_render_escapes_injected_labels(sample_data: DashboardData) -> None:
    rendered = render_dashboard(sample_data)
    assert "<script>alert(1)</script>" not in rendered
    assert "&lt;script&gt;alert(1)&lt;/script&gt;|queue" in rendered


def test_markdown_render_escapes_table_separators(sample_data: DashboardData) -> None:
    rendered = render_dashboard(sample_data, format="markdown")
    assert rendered.startswith("# VoiceLoop local dashboard")
    assert f"> **{SYNTHETIC_BANNER}**" in rendered
    assert "| Source | Observed at | Synthetic | Stale | Note |" in rendered
    assert "<script>alert(1)</script>\\|queue" in rendered
    assert "10/36 (27.8%)" in rendered


def test_render_is_deterministic(sample_data: DashboardData) -> None:
    assert render_dashboard(sample_data) == render_dashboard(sample_data)


def test_render_rejects_unknown_format(sample_data: DashboardData) -> None:
    with pytest.raises(ValidationError, match="unsupported dashboard format"):
        render_dashboard(sample_data, format=cast(DashboardFormat, "xml"))
