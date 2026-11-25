"""Typed settings and conservative LoRA default tests."""

from __future__ import annotations

import pytest
from pydantic import ValidationError as PydanticValidationError

from services.trainers.whisper import (
    DEFAULT_LORA_ALPHA,
    DEFAULT_LORA_R,
    MAX_LORA_SCALING,
    WhisperSettings,
    validate_conservative_lora,
)
from voiceloop_common import ValidationError


def test_defaults_are_conservative_and_no_model_is_hard_coded() -> None:
    settings = WhisperSettings()
    assert settings.backend == "fake"
    assert settings.lora_r == DEFAULT_LORA_R == 8
    assert settings.lora_alpha == DEFAULT_LORA_ALPHA == 1.0
    assert settings.lora_scaling == 0.125
    assert settings.lora_scaling < MAX_LORA_SCALING
    assert settings.lora_scaling < 0.2
    assert settings.condition_on_previous_text is False
    assert settings.preserve_native_sample_rate is True
    assert settings.as_decoding_values()["hotwords"] == ()
    assert settings.as_decoding_values()["initial_prompt"] == ""


def test_env_prefix_is_voiceloop_whisper(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VOICELOOP_WHISPER_LORA_R", "16")
    monkeypatch.setenv("VOICELOOP_WHISPER_LORA_ALPHA", "1.5")
    monkeypatch.setenv("VOICELOOP_WHISPER_REPLAY_FRACTION", "0.5")
    settings = WhisperSettings()
    assert settings.lora_r == 16
    assert settings.lora_alpha == 1.5
    assert settings.lora_scaling == 1.5 / 16
    assert settings.replay_fraction == 0.5


def test_settings_reject_non_conservative_scaling(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VOICELOOP_WHISPER_LORA_ALPHA", "20")
    with pytest.raises(PydanticValidationError, match="alpha/r"):
        WhisperSettings()


def test_explicit_lora_validator_fails_closed() -> None:
    with pytest.raises(ValidationError, match="exceeds bound"):
        validate_conservative_lora(8, 20.0)
    with pytest.raises(ValidationError, match="positive integer"):
        validate_conservative_lora(0, 1.0)
    config = validate_conservative_lora(8, 1.0)
    assert config.scaling == 0.125
    assert config.to_dict()["max_scaling"] == 2.0
