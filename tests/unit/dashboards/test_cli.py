"""Tests for the offline dashboard CLI."""

from __future__ import annotations

from io import StringIO
from pathlib import Path

from infra.dashboards.cli import main
from infra.dashboards.data import (
    FakeModelStateProvider,
    FakeQueueDepthProvider,
    FakeReviewThroughputProvider,
    PromotionRecord,
)
from infra.dashboards.render import SYNTHETIC_BANNER
from infra.dashboards.settings import DashboardSettings


def test_cli_writes_default_html_from_local_sources(tmp_path: Path) -> None:
    output = tmp_path / "dashboard.html"
    stdout = StringIO()
    assert main(["--output", str(output)], stdout=stdout) == 0
    rendered = output.read_text(encoding="utf-8")
    assert rendered.startswith("<!DOCTYPE html>")
    assert SYNTHETIC_BANNER in rendered
    assert "tests/fixtures/turns.jsonl" in rendered
    assert "infra/dashboards/examples/synthetic-snapshot.json" in rendered
    assert "unknown / no version recorded" in rendered
    assert "unknown / no promotion recorded" in rendered
    assert "10/36 (27.8%)" in rendered
    assert f"Wrote dashboard: {output}" in stdout.getvalue()


def test_cli_uses_injected_fakes_and_markdown_format(tmp_path: Path) -> None:
    output = tmp_path / "nested" / "dashboard.md"
    promotion = PromotionRecord(
        version="test-v2",
        promoted_at="2026-11-24T09:00:00Z",
        source="fake://release/aliases",
    )
    stdout = StringIO()
    code = main(
        [
            "--output",
            str(output),
            "--format",
            "markdown",
            "--title",
            "Test dashboard",
        ],
        settings=DashboardSettings(format="html"),
        failure_provider=None,
        queue_provider=FakeQueueDepthProvider(depth=12, source="fake://queue"),
        review_provider=FakeReviewThroughputProvider(completed=4, source="fake://review"),
        model_provider=FakeModelStateProvider(
            version_in_service="test-v1", promotion=promotion, source="fake://release"
        ),
        stdout=stdout,
    )
    assert code == 0
    rendered = output.read_text(encoding="utf-8")
    assert rendered.startswith("# Test dashboard")
    assert "| Queue depth | 12 | fake://queue |" in rendered
    assert "| Model version in service | test-v1 | fake://release |" in rendered
    assert "test-v2" in rendered
    assert "fake://release/aliases" in rendered


def test_cli_returns_error_code_for_missing_fixture(tmp_path: Path) -> None:
    output = tmp_path / "not-written.html"
    stdout = StringIO()
    settings = DashboardSettings(fixtures_path="tests/fixtures/does-not-exist.jsonl")
    assert main(["--output", str(output)], settings=settings, stdout=stdout) == 2
    assert "ERROR: ConfigError" in stdout.getvalue()
    assert not output.exists()
