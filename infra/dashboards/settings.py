"""Typed settings for the local, offline dashboard renderer (P2-11)."""

from __future__ import annotations

from typing import Literal

from pydantic import Field, field_validator

from voiceloop_common import BaseSettings, settings_config

DashboardFormat = Literal["html", "markdown"]


class DashboardSettings(BaseSettings):
    """Environment-backed dashboard settings, prefix ``VOICELOOP_DASHBOARDS_``.

    Defaults are deliberately local and synthetic: a read-only fixture manifest
    and a committed example snapshot. No service URL, credential, Prometheus
    endpoint or Grafana dependency exists here. ``stale_after_minutes`` is a
    display-only freshness hint, never a promotion or gate threshold.
    """

    model_config = settings_config("VOICELOOP_DASHBOARDS_")

    title: str = "VoiceLoop local dashboard"
    fixtures_path: str = "tests/fixtures/turns.jsonl"
    snapshot_path: str = "infra/dashboards/examples/synthetic-snapshot.json"
    format: DashboardFormat = "html"
    stale_after_minutes: int = Field(
        default=1440,
        ge=1,
        description="Age above which a metric is labelled stale in the rendered artifact.",
    )

    @field_validator("title", "fixtures_path", "snapshot_path")
    @classmethod
    def _non_empty_text(cls, value: str) -> str:
        """Reject empty text for fields used as labels or paths."""
        if not value.strip():
            raise ValueError("must be a non-empty string")
        return value
