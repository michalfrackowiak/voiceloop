"""Typed settings for the P3-07 shadow-mode rehearsal.

Every field is read from an environment variable prefixed with
``VOICELOOP_RELEASE_SHADOW_``. No credentials, model names, serving addresses
or promotion thresholds live here: real adapters get those from their own
deployment configuration. Shadow mode is evidence collection only (P3-08 owns
the promotion gate).
"""

from __future__ import annotations

from datetime import timedelta

from pydantic import Field, field_validator

from voiceloop_common import BaseSettings, settings_config

DEFAULT_TASK_ID = "P3-07"
DEFAULT_SCORABLE_STATUSES = "confirmed,edited_by_staff"
DEFAULT_IGNORED_OUTCOME_FIELDS = "call_id,status,updated_at"
SYNTHETIC_STATUS_LABEL = "SYNTHETIC/FIXTURE STAND-IN - NOT PRODUCTION SHADOW EVIDENCE"


class ShadowSettings(BaseSettings):
    """Environment-backed configuration for one bounded shadow run."""

    model_config = settings_config("VOICELOOP_RELEASE_SHADOW_")

    task_id: str = DEFAULT_TASK_ID
    max_pending: int = Field(default=64, ge=1)
    mirror_budget_ms: float = Field(default=2.0, gt=0)
    max_turns_per_drain: int = Field(default=128, ge=1)
    scorable_statuses_csv: str = DEFAULT_SCORABLE_STATUSES
    ignored_outcome_fields_csv: str = DEFAULT_IGNORED_OUTCOME_FIELDS
    fixtures_path: str = "tests/fixtures/turns.jsonl"
    output_path: str = "reports/release/shadow/comparison.json"
    report_status_label: str = SYNTHETIC_STATUS_LABEL
    log_level: str = "INFO"

    @field_validator(
        "task_id", "scorable_statuses_csv", "fixtures_path", "output_path", "report_status_label"
    )
    @classmethod
    def _require_text(cls, value: str) -> str:
        """Reject blank identifiers, paths and labels."""
        if not value.strip():
            raise ValueError("must be a non-blank string")
        return value

    @field_validator("log_level")
    @classmethod
    def _require_log_level(cls, value: str) -> str:
        """Reject blank log levels without imposing a tool choice."""
        if not value.strip():
            raise ValueError("log level must be a non-blank string")
        return value

    def scorable_statuses(self) -> frozenset[str]:
        """Return the outcome statuses eligible for booking-field scoring."""
        return frozenset(
            value.strip() for value in self.scorable_statuses_csv.split(",") if value.strip()
        )

    def ignored_outcome_fields(self) -> frozenset[str]:
        """Return final-record fields that are not comparable booking slots."""
        return frozenset(
            value.strip() for value in self.ignored_outcome_fields_csv.split(",") if value.strip()
        )

    def mirror_budget(self) -> timedelta:
        """Return the call-path enqueue deadline as a timedelta."""
        return timedelta(milliseconds=self.mirror_budget_ms)
