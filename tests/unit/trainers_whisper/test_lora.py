"""Conservative LoRA backend and deterministic adapter tests."""

from __future__ import annotations

import json

import pytest

from services.trainers.whisper import (
    STANDIN_WARNING,
    DataManifest,
    FakeLoraAdapter,
    FakeTransformersPeftBackend,
    LoraConfig,
    build_data_manifest,
    sha256_text,
    validate_conservative_lora,
)
from services.trainers.whisper import dataset as dataset_module
from voiceloop_common import ValidationError

from .conftest import DATASET_HASH, make_example


def _data_manifest(text: str = "reserve a standard room") -> DataManifest:
    example = make_example("ex-1", text=text)
    return build_data_manifest(
        DATASET_HASH,
        (example,),
        replay_fraction=0.0,
        preserve_native_sample_rate=True,
    )


def _train(text: str = "reserve a standard room", scaling: float = 0.125) -> FakeLoraAdapter:
    data = _data_manifest(text)
    lora = validate_conservative_lora(8, scaling * 8)
    return FakeTransformersPeftBackend().train(
        data.examples,
        model="fake-whisper-base",
        seed=7,
        dataset_hash=DATASET_HASH,
        data_manifest_hash=data.content_hash(),
        params={"epochs": 1},
        lora=lora,
    )


def test_backend_is_deterministic_and_labelled() -> None:
    first = _train()
    second = _train()
    assert first.content_hash() == second.content_hash()
    payload = json.dumps(first.to_dict(), sort_keys=True)
    assert first.synthetic_training_standin is True
    assert "NOT A REAL FINE-TUNE" in first.warning
    assert first.lora["scaling"] == 0.125
    assert first.lora["r"] == 8
    assert first.lora["alpha"] == 1.0
    assert "reserve a standard room" not in payload
    assert len(first.deltas) == 1
    assert first.deltas[0].target_text_sha256 == sha256_text("reserve a standard room")


def test_scaling_and_data_change_the_content_hash() -> None:
    base = _train()
    higher = _train(scaling=0.25)
    different_text = _train(text="two nights please")
    assert base.content_hash() != higher.content_hash()
    assert base.content_hash() != different_text.content_hash()


def test_adapter_roundtrip_rejects_tampering() -> None:
    adapter = _train()
    decoded = FakeLoraAdapter.from_dict(adapter.to_dict())
    assert decoded.content_hash() == adapter.content_hash()
    tampered = adapter.to_dict()
    tampered["lora"] = dict(adapter.lora)
    tampered["lora"]["alpha"] = 100.0
    with pytest.raises(ValidationError, match="exceeds bound"):
        FakeLoraAdapter.from_dict(tampered)
    with pytest.raises(ValidationError, match="unexpected fields"):
        FakeLoraAdapter.from_dict({**adapter.to_dict(), "extra": 1})


def test_lora_config_hard_bound() -> None:
    config = LoraConfig(r=8, alpha=17.0, max_scaling=2.0)
    with pytest.raises(ValidationError, match="exceeds bound"):
        config.validate()
    with pytest.raises(ValidationError, match="at most 2.0"):
        LoraConfig(r=8, alpha=1.0, max_scaling=3.0).validate()


def test_standin_warning_matches_dataset_label() -> None:
    assert _train().warning == STANDIN_WARNING
    assert dataset_module.STANDIN_WARNING == STANDIN_WARNING
