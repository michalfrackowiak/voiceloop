"""Typed settings for the P4-01 synthetic canary rehearsal.

Every field is read from an environment variable prefixed with
``VOICELOOP_CANARY_``. No credentials, model names, serving addresses or
promotion thresholds live here: a canary never promotes, never claims a
champion and never changes a gate rule (``context.md`` §2 invariant 4 and §14).
The traffic fraction and evaluation window are explicitly bounded so a
misconfiguration cannot silently turn a canary into a full rollout.
"""

from __future__ import annotations

from datetime import timedelta

from pydantic import Field, field_validator

from voiceloop_common import BaseSettings, settings_config

DEFAULT_TASK_ID = "P4-01"
DEFAULT_SPLIT_SALT = "voiceloop-canary-v1"
DEFAULT_SCORABLE_STATUSES = "confirmed,edited_by_staff"
DEFAULT_IGNORED_OUTCOME_FIELDS = "call_id,status,updated_at"
DEFAULT_ROLLBACK_ACTOR_ID = "canary-auto-rollback"
DEFAULT_FIXTURES_PATH = "tests/fixtures/turns.jsonl"
DEFAULT_AUDIT_PATH = "reports/release/canary/rollback_audit.jsonl"
SYNTHETIC_LABEL = "SYNTHETIC/FIXTURE STAND-IN - NOT LIVE CANARY EVIDENCE"

# Explicit engineering bounds, not promotion thresholds (§14). The cap keeps
# this module a canary: a caller cannot configure a 100% canary by mistake.
MAX_CANARY_FRACTION = 0.20
MAX_WINDOW_SECONDS = 3600.0
MAX_CALLS_PER_TICK = 128
MAX_PAIRS = 10_000


class CanarySettings(BaseSettings):
    """Environment-backed configuration for one bounded canary rehearsal."""

    model_config = settings_config("VOICELOOP_CANARY_")

    task_id: str = DEFAULT_TASK_ID
    canary_fraction: float = Field(default=0.05, ge=0.0, le=MAX_CANARY_FRACTION)
    split_salt: str = DEFAULT_SPLIT_SALT
    window_seconds: float = Field(default=300.0, gt=0.0, le=MAX_WINDOW_SECONDS)
    min_paired_calls: int = Field(default=3, ge=1)
    max_calls_per_tick: int = Field(default=16, ge=1, le=MAX_CALLS_PER_TICK)
    max_pairs: int = Field(default=256, ge=1, le=MAX_PAIRS)
    scorable_statuses_csv: str = DEFAULT_SCORABLE_STATUSES
    ignored_outcome_fields_csv: str = DEFAULT_IGNORED_OUTCOME_FIELDS
    rollback_actor_id: str = DEFAULT_ROLLBACK_ACTOR_ID
    fixtures_path: str = DEFAULT_FIXTURES_PATH
    rollback_audit_path: str = DEFAULT_AUDIT_PATH
    evidence_label: str = SYNTHETIC_LABEL
    log_level: str = "INFO"

    @field_validator(
        "task_id",
        "split_salt",
        "scorable_statuses_csv",
        "ignored_outcome_fields_csv",
        "rollback_actor_id",
        "fixtures_path",
        "rollback_audit_path",
        "evidence_label",
        "log_level",
    )
    @classmethod
    def _require_text(cls, value: str) -> str:
        """Reject blank identifiers, paths and evidence labels."""
        if not value.strip():
            raise ValueError("must be a non-blank string")
        return value

    def scorable_statuses(self) -> frozenset[str]:
        """Return call statuses whose final booking record may be scored."""
        return frozenset(
            value.strip() for value in self.scorable_statuses_csv.split(",") if value.strip()
        )

    def ignored_outcome_fields(self) -> frozenset[str]:
        """Return final-record keys that are not comparable booking fields."""
        return frozenset(
            value.strip() for value in self.ignored_outcome_fields_csv.split(",") if value.strip()
        )

    def window(self) -> timedelta:
        """Return the rollback evaluation window as a timedelta."""
        return timedelta(seconds=self.window_seconds)
