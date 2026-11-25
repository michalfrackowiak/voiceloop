"""Parser settings tests: stand-in defaults and VOICELOOP_PARSER_* prefix."""

from __future__ import annotations

import pytest

from services.trainers.parser import ParserSettings


def test_defaults_are_local_standin() -> None:
    settings = ParserSettings()
    assert settings.backend == "fake"
    assert settings.epochs >= 1
    assert settings.exclude_call_ids == ()
    assert settings.as_training_params()["lora_rank"] >= 1


def test_env_prefix_reads_backend_and_exclusions(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VOICELOOP_PARSER_BACKEND", "trl")
    monkeypatch.setenv("VOICELOOP_PARSER_EXCLUDE_CALL_IDS", '["call-golden"]')
    settings = ParserSettings()
    assert settings.backend == "trl"
    assert settings.exclude_call_ids == ("call-golden",)


def test_training_params_are_stable_and_json_serialisable() -> None:
    settings = ParserSettings(epochs=2, learning_rate=0.001, lora_rank=4, lora_alpha=8.0)
    params = settings.as_training_params()
    assert params == {
        "epochs": 2,
        "learning_rate": 0.001,
        "lora_rank": 4,
        "lora_alpha": 8.0,
    }
    assert (
        ParserSettings(
            epochs=2, learning_rate=0.001, lora_rank=4, lora_alpha=8.0
        ).as_training_params()
        == params
    )
