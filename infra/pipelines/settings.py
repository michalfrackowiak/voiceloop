"""Typed settings for the pipeline CLI and its trigger policy (P2-10)."""

from __future__ import annotations

from pydantic import Field, field_validator

from voiceloop_common import BaseSettings, settings_config

DEFAULT_MIN_NEW_VERIFIED = 200


class PipelineSettings(BaseSettings):
    """Environment-backed pipeline settings; all fields read ``VOICELOOP_PIPELINES_*``.

    Defaults are local/fake-safe: no GPU, network, service, or secret is needed.
    ``min_new_verified`` is a starting point from context.md section 7 (a few
    hundred verified examples for a parser-tier retrain), not a promotion
    threshold; promotion thresholds are owned by the gate (P2-09) and the lead.
    """

    model_config = settings_config("VOICELOOP_PIPELINES_")

    fixtures_path: str = "tests/fixtures/turns.jsonl"
    dataset_id_prefix: str = "voiceloop-pipeline"
    min_new_verified: int = Field(
        default=DEFAULT_MIN_NEW_VERIFIED,
        ge=1,
        description="Minimum newly verified examples required for a data-volume trigger.",
    )
    drift_alarm: bool = Field(
        default=False,
        description="Externally raised drift signal (e.g. fed by a future P4-02 detector).",
    )
    manual: bool = Field(
        default=False,
        description="Operator override; a manual trigger is always allowed.",
    )
    eval_sets: tuple[str, ...] = Field(
        default=("general_capability",),
        description="Evaluation set names passed to the evaluator; fixtures are not golden.",
    )
    eval_report_path: str = Field(
        default="fake://evalgate/report.json",
        description=(
            "Report URI/path returned by the fake evaluator; no file is written by default."
        ),
    )
    run_record_path: str = Field(
        default="",
        description="Optional path for the last run record; empty means in-memory only.",
    )
    trainer_git_sha: str = Field(
        default="0" * 40,
        pattern=r"^[0-9a-f]{7,40}$",
        description=(
            "Fake training-code SHA used by the fixture trainer; inject the real SHA in production."
        ),
    )
    trainer_image_digest: str = Field(
        default="fake@sha256:" + "0" * 64,
        pattern=r"^[^\s@]+@sha256:[0-9a-f]{64}$",
        description=(
            "Fake pinned image used by the fixture trainer; inject the real digest in production."
        ),
    )

    @field_validator("eval_sets")
    @classmethod
    def _eval_sets_not_empty(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        """Reject an empty evaluation-set selection."""
        if not value or any(not isinstance(name, str) or not name for name in value):
            raise ValueError("eval_sets must contain at least one non-empty set name")
        return value
