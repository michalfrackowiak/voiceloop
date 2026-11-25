"""Typed settings for the P2-09 promotion gate.

Every field reads an environment variable prefixed with ``VOICELOOP_GATE_``.
Threshold defaults are conservative starting points, not approved production
values: ``context.md`` section 14 requires lead sign-off for any promotion
threshold change.
"""

from __future__ import annotations

from pydantic import Field

from voiceloop_common import BaseSettings, settings_config

from ..settings import EvalSetName


def _default_eval_sets() -> list[EvalSetName]:
    """Return the accepted gate evidence sets; a bare golden label is not evidence."""
    return ["general_capability", "regression"]


class GateSettings(BaseSettings):
    """Configuration for gate decisions, conversion and alias promotion."""

    model_config = settings_config("VOICELOOP_GATE_")

    # The lower bound of the paired challenger-minus-champion e2e delta must be
    # strictly greater than this margin (context.md section 9).
    min_e2e_margin: float = Field(default=0.01, ge=0)
    min_ci_level: float = Field(default=0.95, gt=0, lt=1)
    slice_regression_tolerance: float = Field(default=0.0, ge=0)

    eval_sets: list[EvalSetName] = Field(default_factory=_default_eval_sets)
    audit_log_path: str = "reports/evalgate/promotion_audit.jsonl"
