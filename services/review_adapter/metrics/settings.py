"""Typed environment-backed settings for the P2-04 review UX metrics harness."""

from __future__ import annotations

from pydantic import Field

from voiceloop_common import BaseSettings, settings_config

EXPECTED_GOLDEN_CONTENT_HASH = "662cdaf43ee2d5fde03faf5e69fde196114d4f312406d94c00bf53714d03da8e"
"""Content hash committed by the sealed synthetic golden stand-in (P1-06)."""

SCRIPTED_DURATION_SECONDS = 72.0
"""Duration from the committed P2-03 scripted review helper, not measured human time."""


class ReviewMetricsSettings(BaseSettings):
    """Configuration for fixture-only review metrics; every field reads a prefixed env var.

    All inputs are read-only synthetic fixtures. ``output_path`` is the only path
    this harness writes and may be overridden by the caller or CLI. No real
    reviewer, review tool, audio byte, credential or wall-clock setting exists.
    """

    model_config = settings_config("VOICELOOP_REVIEW_METRICS_")

    fixtures_path: str = "tests/fixtures/turns.jsonl"
    golden_artifact_dir: str = "services/datasets/golden/artifacts"
    golden_version: str = "golden.v1"
    expected_golden_content_hash: str = EXPECTED_GOLDEN_CONTENT_HASH
    output_path: str = "reports/review_metrics/report.json"
    start_at: str = "2026-11-25T10:00:00Z"
    scripted_duration_seconds: float = Field(default=SCRIPTED_DURATION_SECONDS, gt=0)
