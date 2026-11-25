"""Typed settings and environment-prefix tests."""

from __future__ import annotations

import pytest

from services.trainers.vocab.settings import EXPECTED_GOLDEN_HASH, VocabSettings


def test_defaults_use_vocaboloop_vocab_values() -> None:
    settings = VocabSettings()
    assert settings.fixtures_path == "tests/fixtures/turns.jsonl"
    assert settings.runtime == "faster-whisper-ctranslate2"
    assert settings.language == "en"
    assert settings.enable_aliases is True
    assert settings.enable_normalisation is True
    assert settings.enable_boosting is True
    assert settings.rewrite_transcript is False
    assert settings.expected_golden_hash == EXPECTED_GOLDEN_HASH
    assert settings.model_config.get("env_prefix") == "VOICELOOP_VOCAB_"


def test_environment_overrides_are_scoped(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VOICELOOP_VOCAB_ENABLE_BOOSTING", "false")
    monkeypatch.setenv("VOICELOOP_VOCAB_MAX_HOTWORDS", "7")
    monkeypatch.setenv("VOICELOOP_VOCAB_NAME_ALIAS_OVERRIDES", '{"foo":"bar"}')
    monkeypatch.setenv("VOICELOOP_EVALGATE_BOOTSTRAP_SEED", "123")
    settings = VocabSettings()
    assert settings.enable_boosting is False
    assert settings.max_hotwords == 7
    assert settings.name_alias_overrides == {"foo": "bar"}
    assert not hasattr(settings, "bootstrap_seed")
