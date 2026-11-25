"""Typed settings for the sealed golden-set workflow.

Every field reads an environment variable prefixed with ``VOICELOOP_GOLDEN_``.
No secrets, real reviewer identities or production paths live in this module.
"""

from __future__ import annotations

from typing import Literal

from pydantic import Field, model_validator

from voiceloop_common import BaseSettings, settings_config

LabelToolName = Literal["in_memory"]


class GoldenSettings(BaseSettings):
    """Configuration for sampling, labelling and sealing golden-set versions."""

    model_config = settings_config("VOICELOOP_GOLDEN_")

    fixtures_path: str = "tests/fixtures/turns.jsonl"
    label_tool: LabelToolName = "in_memory"
    seed: int = 0

    # Production band from P1-06.1: 300-500 utterances. Tests pass smaller
    # values because the read-only synthetic fixture pool contains only 36.
    target_min_utterances: int = Field(default=300, ge=1)
    target_max_utterances: int = Field(default=500, ge=1)

    require_full_second_review: bool = True
    artifact_dir: str = "services/datasets/golden/artifacts"

    @model_validator(mode="after")
    def _targets_are_ordered(self) -> GoldenSettings:
        """Reject a target band whose lower bound exceeds its upper bound."""
        if self.target_min_utterances > self.target_max_utterances:
            raise ValueError("target_min_utterances must not exceed target_max_utterances")
        return self
