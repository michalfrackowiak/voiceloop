"""Typed settings for the dataset builder (P2-05)."""

from __future__ import annotations

from pydantic import Field, model_validator

from voiceloop_common import BaseSettings, settings_config


class DatasetSettings(BaseSettings):
    """Environment-backed defaults for building and exporting datasets."""

    model_config = settings_config("VOICELOOP_DATASETS_")

    fixtures_path: str = "tests/fixtures/turns.jsonl"
    golden_manifest_path: str = "services/datasets/golden/artifacts/golden.v1.json"
    regression_path: str = ""
    dataset_id_prefix: str = "voiceloop-dataset"
    val_fraction: float = Field(default=0.2, ge=0.0, le=1.0)
    test_fraction: float = Field(default=0.1, ge=0.0, le=1.0)
    split_seed: int = 0
    replay_size: int = Field(default=0, ge=0)
    replay_seed: int = 0
    prior_turns_limit: int = Field(default=5, ge=0)

    @model_validator(mode="after")
    def _split_fractions_are_valid(self) -> DatasetSettings:
        """Reject split fractions that would assign more than all calls."""
        if self.val_fraction + self.test_fraction > 1.0:
            raise ValueError("val_fraction + test_fraction must not exceed 1.0")
        return self
