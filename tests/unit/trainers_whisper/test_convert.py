"""Merge-to-convert seam and fake CTranslate2 artifact tests."""

from __future__ import annotations

import json

import pytest

from services.trainers.whisper import (
    ConvertedModel,
    Ctranslate2Runtime,
    DecodingSettings,
    FakeConvertedModelStore,
    FakeCtranslate2Backend,
    FakeCtranslate2Runtime,
    FakeLoraAdapter,
    FakeMergeBackend,
    FakeTransformersPeftBackend,
    build_data_manifest,
    sha256_text,
    validate_conservative_lora,
)
from voiceloop_common import ValidationError

from .conftest import AUDIO_ROOT, DATASET_HASH, make_example


def _train_adapter(text: str = "reserve a standard room") -> FakeLoraAdapter:
    """Build a deterministic stand-in adapter for one digest-only example."""
    example = make_example("ex-1", text=text)
    data = build_data_manifest(
        DATASET_HASH, (example,), replay_fraction=0.0, preserve_native_sample_rate=True
    )
    return FakeTransformersPeftBackend().train(
        data.examples,
        model="fake-whisper-base",
        seed=7,
        dataset_hash=DATASET_HASH,
        data_manifest_hash=data.content_hash(),
        params={"epochs": 1},
        lora=validate_conservative_lora(8, 1.0),
    )


def _converted(text: str = "reserve a standard room") -> ConvertedModel:
    example = make_example("ex-1", text=text)
    data = build_data_manifest(
        DATASET_HASH, (example,), replay_fraction=0.0, preserve_native_sample_rate=True
    )
    merged = FakeMergeBackend().merge(_train_adapter(text), model="fake-whisper-base")
    return FakeCtranslate2Backend().convert(
        merged,
        model="fake-whisper-base",
        data_manifest=data,
        decoding=DecodingSettings(),
    )


def test_merge_convert_is_content_addressed_and_pii_free() -> None:
    first = _converted()
    second = _converted()
    assert first.content_hash() == second.content_hash()
    payload = json.dumps(first.to_dict(), sort_keys=True)
    assert first.synthetic_training_standin is True
    assert "NOT A REAL FINE-TUNE" in first.warning
    assert "reserve a standard room" not in payload
    assert all(len(value) == 64 for value in first.transcript_index.values())
    assert all(len(key) == 64 for key in first.transcript_index)
    assert first.decoding["condition_on_previous_text"] is False
    assert first.decoding["hotwords"] == []
    assert first.decoding["initial_prompt"] == ""
    assert first.data_manifest["mix"]["telephony_8k_examples"] == 0
    assert converted_keys_are_promotion_free(first)


def converted_keys_are_promotion_free(model: ConvertedModel) -> bool:
    """Return whether the artifact carries no alias/threshold/promotion field."""
    banned = ("alias", "threshold", "promotion", "champion", "canary")
    return not any(word in key.lower() for key in model.to_dict() for word in banned)


def test_store_roundtrip_and_tamper_detection() -> None:
    store = FakeConvertedModelStore()
    uri = store.save(_converted(), prefix="fake://whisper-artifacts")
    assert "/whisper-ctranslate2/" in uri
    assert store.load(uri).content_hash() == uri.rsplit("/", 1)[-1]
    with pytest.raises(ValidationError, match="not found"):
        store.load(uri.replace(uri.rsplit("/", 1)[-1], "0" * 64))
    store.put_raw(uri, "{}")
    with pytest.raises(ValidationError, match="unexpected fields"):
        store.load(uri)
    store.put_raw(uri, "{bad json")
    with pytest.raises(ValidationError, match="not valid JSON"):
        store.load(uri)


def test_fake_runtime_transcribes_digests_and_fails_closed() -> None:
    store = FakeConvertedModelStore()
    model = _converted()
    uri = store.save(model, prefix="fake://whisper-artifacts")
    runtime: Ctranslate2Runtime = FakeCtranslate2Runtime(store)
    audio_uri = f"{AUDIO_ROOT}/call-train-a/ex-1.wav"
    result = runtime.transcribe(uri, audio_uri)
    assert result.text_sha256 == sha256_text("reserve a standard room")
    assert result.audio_uri_sha256 == sha256_text(audio_uri)
    assert result.sample_rate_hz == 16000
    assert result.synthetic is True
    with pytest.raises(ValidationError, match="not present"):
        runtime.transcribe(uri, "fake://audio/unknown.wav")


def test_decoding_settings_reject_pii_bearing_runtime_config() -> None:
    with pytest.raises(ValidationError, match="hotwords"):
        DecodingSettings(hotwords=("Guest Name",)).validate()
    with pytest.raises(ValidationError, match="initial_prompt"):
        DecodingSettings(initial_prompt="transcribe for Ada").validate()
    raw = _converted().to_dict()
    raw["decoding"] = dict(raw["decoding"])
    raw["decoding"]["hotwords"] = ["Ada Fixture"]
    with pytest.raises(ValidationError, match="hotwords must be empty"):
        ConvertedModel.from_dict(raw)


def test_artifact_from_dict_validates_hashes() -> None:
    raw = _converted().to_dict()
    raw["transcript_index"] = {"short": "0" * 64}
    with pytest.raises(ValidationError, match="audio digest"):
        ConvertedModel.from_dict(raw)
    raw = _converted().to_dict()
    raw["merged_hash"] = "bad"
    with pytest.raises(ValidationError, match="merged_hash"):
        ConvertedModel.from_dict(raw)


def test_merge_descriptor_inherits_adapter_label() -> None:
    merged = FakeMergeBackend().merge(_train_adapter(), model="fake-whisper-base")
    assert merged.synthetic_training_standin is True
    assert merged.warning == "SYNTHETIC/FIXTURE STAND-IN - NOT A REAL FINE-TUNE"
