"""Tests for the typed VOICELOOP_NEMO_ settings surface."""

from __future__ import annotations

import pytest

from services.trainers.nemo import NemoSettings


def test_env_prefix_is_scoped(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VOICELOOP_NEMO_BACKEND", "nemo")
    monkeypatch.setenv("VOICELOOP_NEMO_MANIFEST_PATH", "synthetic://dataset.nemo.jsonl")
    monkeypatch.setenv("VOICELOOP_NEMO_EPOCHS", "3")
    monkeypatch.setenv("VOICELOOP_NEMO_BATCH_SIZE", "8")
    monkeypatch.setenv("VOICELOOP_NEMO_CHECKPOINT_URI", "fake://checkpoints/parakeet")
    settings = NemoSettings()
    assert settings.backend == "nemo"
    assert settings.manifest_path == "synthetic://dataset.nemo.jsonl"
    assert settings.epochs == 3
    assert settings.batch_size == 8
    assert settings.checkpoint_uri == "fake://checkpoints/parakeet"


def test_training_params_are_json_serialisable() -> None:
    settings = NemoSettings(epochs=2, learning_rate=0.0002, batch_size=6)
    assert settings.as_training_params() == {
        "epochs": 2,
        "learning_rate": 0.0002,
        "batch_size": 6,
    }


@pytest.mark.parametrize(
    ("field", "value"),
    [("epochs", 0), ("batch_size", 0), ("learning_rate", 0.0), ("backend", "unsloth")],
)
def test_invalid_settings_fail_closed(field: str, value: object) -> None:
    with pytest.raises(ValueError):
        NemoSettings.model_validate({field: value})
