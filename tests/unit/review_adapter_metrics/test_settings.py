"""Typed settings tests for the fixture-only review metrics harness."""

from __future__ import annotations

import pytest

from services.review_adapter.metrics.settings import (
    EXPECTED_GOLDEN_CONTENT_HASH,
    SCRIPTED_DURATION_SECONDS,
    ReviewMetricsSettings,
)


def test_env_prefix_overrides(monkeypatch: pytest.MonkeyPatch) -> None:
    """Settings read VOICELOOP_REVIEW_METRICS_ variables only."""
    monkeypatch.setenv("VOICELOOP_REVIEW_METRICS_OUTPUT_PATH", "tmp/report.json")
    monkeypatch.setenv("VOICELOOP_REVIEW_METRICS_START_AT", "2026-11-26T09:30:00Z")
    monkeypatch.setenv("VOICELOOP_REVIEW_METRICS_SCRIPTED_DURATION_SECONDS", "12.5")

    settings = ReviewMetricsSettings()

    assert settings.output_path == "tmp/report.json"
    assert settings.start_at == "2026-11-26T09:30:00Z"
    assert settings.scripted_duration_seconds == 12.5
    assert settings.expected_golden_content_hash == EXPECTED_GOLDEN_CONTENT_HASH


def test_defaults_are_synthetic_fixture_only() -> None:
    """Defaults point only at committed fixtures and a fake report destination."""
    settings = ReviewMetricsSettings()

    assert settings.fixtures_path == "tests/fixtures/turns.jsonl"
    assert settings.golden_artifact_dir == "services/datasets/golden/artifacts"
    assert settings.golden_version == "golden.v1"
    assert settings.scripted_duration_seconds == SCRIPTED_DURATION_SECONDS
    assert settings.output_path == "reports/review_metrics/report.json"


def test_invalid_scripted_duration_rejected() -> None:
    """A non-positive fake duration cannot produce a valid timing report."""
    with pytest.raises(ValueError):
        ReviewMetricsSettings(scripted_duration_seconds=0)
