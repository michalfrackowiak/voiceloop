"""Tests for P2-10 typed settings and env prefix."""

from __future__ import annotations

import pytest
from pydantic import ValidationError as PydanticValidationError

from infra.pipelines.settings import DEFAULT_MIN_NEW_VERIFIED, PipelineSettings


def test_env_prefix_is_pipeline_scoped(monkeypatch: pytest.MonkeyPatch) -> None:
    assert PipelineSettings.model_config["env_prefix"] == "VOICELOOP_PIPELINES_"
    monkeypatch.setenv("VOICELOOP_PIPELINES_MIN_NEW_VERIFIED", "17")
    monkeypatch.setenv("VOICELOOP_PIPELINES_DRIFT_ALARM", "true")
    config = PipelineSettings()
    assert config.min_new_verified == 17
    assert config.drift_alarm is True


def test_defaults_are_local_and_fake_safe() -> None:
    config = PipelineSettings()
    assert config.min_new_verified == DEFAULT_MIN_NEW_VERIFIED
    assert config.fixtures_path == "tests/fixtures/turns.jsonl"
    assert config.eval_sets == ("general_capability",)
    assert config.manual is False
    assert config.run_record_path == ""


def test_invalid_threshold_is_rejected() -> None:
    with pytest.raises(PydanticValidationError):
        PipelineSettings(min_new_verified=0)


def test_empty_eval_sets_are_rejected() -> None:
    with pytest.raises(PydanticValidationError):
        PipelineSettings(eval_sets=())
