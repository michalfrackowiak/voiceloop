"""Typed settings for the P4-02 synthetic drift/retraining policy."""

from __future__ import annotations

from typing import Final

from pydantic import Field, field_validator

from voiceloop_common import BaseSettings, settings_config

SYNTHETIC_LABEL: Final = "SYNTHETIC/FIXTURE STAND-IN - NOT PRODUCTION DRIFT EVIDENCE"

EXPECTED_FIXTURE_SHA256: Final = "aca23a303954d035245a97f78319df3694d63611b717a2ff2131bc2e0a4453fd"
EXPECTED_GOLDEN_CONTENT_HASH: Final = (
    "662cdaf43ee2d5fde03faf5e69fde196114d4f312406d94c00bf53714d03da8e"
)

ALLOWED_OUTCOME_STATUSES: Final = (
    "confirmed",
    "edited_by_staff",
    "abandoned",
    "transferred",
    "unknown",
)

DEFAULT_DEGRADED_OUTCOMES: Final = ("edited_by_staff", "abandoned", "transferred")


class DriftSettings(BaseSettings):
    """Environment-backed settings; every field reads ``VOICELOOP_DRIFT_*``.

    Drift thresholds here are engineering placeholders for an alarm only. They
    are not P2-09 promotion thresholds and do not modify the gate or the sealed
    golden set (context.md sections 2.5 and 14). Defaults stay offline/fake-safe.
    """

    model_config = settings_config("VOICELOOP_DRIFT_")

    fixtures_path: str = "tests/fixtures/turns.jsonl"
    fixture_sha256: str = EXPECTED_FIXTURE_SHA256
    golden_artifact_dir: str = "services/datasets/golden/artifacts"
    golden_version: str = "golden.v1"
    golden_content_hash: str = EXPECTED_GOLDEN_CONTENT_HASH

    min_window_samples: int = Field(
        default=10,
        ge=1,
        description="Minimum baseline AND current samples before a detector may alarm.",
    )
    score_buckets: int = Field(default=10, ge=2, le=32)
    score_mean_delta_threshold: float = Field(default=0.05, ge=0.0, le=1.0)
    score_tv_distance_threshold: float = Field(default=0.20, ge=0.0, le=1.0)
    outcome_rate_delta_threshold: float = Field(default=0.10, ge=0.0, le=1.0)
    degraded_outcomes: tuple[str, ...] = Field(default=DEFAULT_DEGRADED_OUTCOMES)

    min_new_verified: int = Field(
        default=200,
        ge=1,
        description=(
            "Data-volume floor from the context.md section 7 rule of thumb; not a "
            "promotion threshold."
        ),
    )

    manual_override: bool = Field(
        default=False,
        description="Explicit operator flag; an override additionally needs actor and reason.",
    )
    manual_actor_id: str = Field(
        default="",
        description="Pseudonymous audit actor, e.g. operator-1a2b3c4d; never a human name.",
    )
    manual_reason_code: str = Field(default="", description="Allowlisted audit reason code.")

    @field_validator("degraded_outcomes")
    @classmethod
    def _validate_degraded_outcomes(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        """Reject unknown or duplicate outcome statuses."""
        if not value:
            raise ValueError("degraded_outcomes must not be empty")
        for status in value:
            if status not in ALLOWED_OUTCOME_STATUSES:
                raise ValueError(f"unknown outcome status: {status!r}")
        if len(set(value)) != len(value):
            raise ValueError("degraded_outcomes must not contain duplicates")
        return value
