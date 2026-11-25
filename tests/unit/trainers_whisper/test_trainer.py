"""End-to-end P3-01 trainer contract, replay, lineage and PII tests."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from services.trainers.core import (
    FakeMlflowLogger,
    TrainerRegistry,
    TrainerResult,
    TrainerRunner,
    TrainerSettings,
)
from services.trainers.whisper import (
    EXPECTED_FIXTURE_HASH,
    EXPECTED_GOLDEN_HASH,
    FakeConvertedModelStore,
    HfJsonExampleLoader,
    ReadOnlyHashGuard,
    StaticSampleRateResolver,
    WhisperLoraTrainer,
    WhisperSettings,
    sha256_text,
)
from voiceloop_common import ConfigError, ValidationError, validate

from .conftest import AUDIO_ROOT, DATASET_HASH, SAMPLE_TEXT, make_example


def standin_settings() -> WhisperSettings:
    """Return conservative settings with one deterministic replay example."""
    return WhisperSettings(replay_fraction=0.5, replay_seed=11)


def make_trainer(
    examples: tuple[Any, ...],
    replay_pool: tuple[Any, ...],
    *,
    settings: WhisperSettings | None = None,
    store: FakeConvertedModelStore | None = None,
    excluded_call_ids: tuple[str, ...] | None = None,
) -> WhisperLoraTrainer:
    """Build a trainer with injected stand-in examples and replay pool."""
    return WhisperLoraTrainer(
        settings=settings or standin_settings(),
        examples=examples,
        replay_pool=replay_pool,
        store=store,
        excluded_call_ids=excluded_call_ids,
    )


def test_runner_end_to_end_logs_lineage_and_hyperparameters(
    manifest: dict[str, Any],
    core_settings: TrainerSettings,
    primary_examples: tuple[Any, ...],
    replay_pool: tuple[Any, ...],
) -> None:
    registry = TrainerRegistry()
    registry.register(make_trainer(primary_examples, replay_pool))
    logger = FakeMlflowLogger()
    result: TrainerResult = TrainerRunner(registry, settings=core_settings, logger=logger).run(
        "whisper-lora", manifest
    )
    validate(result, "trainer_contract.schema.json")
    assert result["artifact_uri"].startswith("fake://whisper-artifacts/whisper-ctranslate2/")
    assert result["lineage"] == {
        "dataset_hash": DATASET_HASH,
        "git_sha": core_settings.git_sha,
        "image_digest": core_settings.image_digest,
    }
    assert result["metrics"]["lora_alpha"] == 1.0
    assert result["metrics"]["lora_rank"] == 8
    assert result["metrics"]["lora_scaling"] == 0.125
    assert result["metrics"]["lora_max_scaling"] == 2.0
    assert result["metrics"]["examples_total"] == 4
    assert result["metrics"]["examples_replay"] == 1
    assert result["metrics"]["examples_8k"] == 1
    assert result["metrics"]["replay_fraction"] == 0.25
    assert result["metrics"]["fake_train_loss"] == 0.0
    assert len(logger.runs) == 1
    assert logger.runs[0].artifact_uri == result["artifact_uri"]


def test_converted_artifact_transcribes_synthetic_fixture_digests(
    manifest: dict[str, Any],
    core_settings: TrainerSettings,
    primary_examples: tuple[Any, ...],
    replay_pool: tuple[Any, ...],
) -> None:
    trainer = make_trainer(primary_examples, replay_pool)
    trainer.train(manifest, core_settings)
    transcription = trainer.transcribe(f"{AUDIO_ROOT}/call-train-a/ex-1.wav")
    assert transcription.text_sha256 == sha256_text(SAMPLE_TEXT)
    assert transcription.sample_rate_hz == 16000
    assert transcription.matched is True
    with pytest.raises(ValidationError, match="not present"):
        trainer.transcribe("fake://audio/unknown.wav")


def test_export_is_deterministic_without_promotion_side_effects(
    manifest: dict[str, Any],
    core_settings: TrainerSettings,
    primary_examples: tuple[Any, ...],
    replay_pool: tuple[Any, ...],
) -> None:
    store = FakeConvertedModelStore()
    first = make_trainer(primary_examples, replay_pool, store=store).train(manifest, core_settings)
    second = make_trainer(
        tuple(reversed(primary_examples)), tuple(reversed(replay_pool)), store=store
    ).train(manifest, core_settings)
    assert first["artifact_uri"] == second["artifact_uri"]
    assert first["run_id"] != second["run_id"]
    assert set(first) == {
        "schema_version",
        "contract_part",
        "artifact_uri",
        "metrics",
        "run_id",
        "lineage",
    }
    assert not any(
        word in key.lower()
        for key in first["metrics"]
        for word in ("alias", "threshold", "promotion", "champion")
    )
    artifact = store.load(first["artifact_uri"])
    assert artifact.data_manifest["mix"]["replay_examples"] == 1
    assert artifact.data_manifest["mix"]["sample_rates_hz"] == {"16000": 3, "8000": 1}


def test_golden_and_regression_exclusions_fail_closed(
    manifest: dict[str, Any],
    core_settings: TrainerSettings,
    primary_examples: tuple[Any, ...],
    replay_pool: tuple[Any, ...],
) -> None:
    with pytest.raises(ValidationError, match="excluded call"):
        make_trainer(primary_examples, replay_pool, excluded_call_ids=("call-train-b",)).train(
            manifest, core_settings
        )
    replay_as_primary = (make_example("rp-x", call_id="call-replay-a"),)
    with pytest.raises(ValidationError, match="excluded call"):
        make_trainer(replay_as_primary, replay_pool, excluded_call_ids=("call-replay-a",)).train(
            manifest, core_settings
        )


def test_real_backend_label_requires_injection(
    manifest: dict[str, Any],
    core_settings: TrainerSettings,
    primary_examples: tuple[Any, ...],
) -> None:
    with pytest.raises(ConfigError, match="requires an injected LoraTrainingBackend"):
        WhisperLoraTrainer(
            settings=WhisperSettings(backend="transformers_peft"),
            examples=primary_examples,
        )
    with pytest.raises(ConfigError, match="must produce a converted artifact"):
        make_trainer(primary_examples, ()).transcribe("fake://audio/x.wav")


def test_hf_loader_never_copies_pii_into_artifacts(
    tmp_path: Path,
    manifest: dict[str, Any],
    core_settings: TrainerSettings,
) -> None:
    secrets = ["Ada Fixture", "ada.fixture@example.com", "+1 555 010 0001"]
    audio_uri = "fake://audio/call-synthetic-train/ex-pii.wav"
    row = {
        "example_id": "call-synthetic-train:0:stt",
        "call_id": "call-synthetic-train",
        "kind": "stt",
        "audio_uri": audio_uri,
        "target_text": f"Book for {secrets[0]} {secrets[1]} {secrets[2]}",
        "duration_ms": 300,
        "replay": False,
        "synthetic": True,
    }
    path = tmp_path / "dataset.hf.json"
    path.write_text(json.dumps([row]), encoding="utf-8")
    loader = HfJsonExampleLoader(StaticSampleRateResolver({audio_uri: 8000}, default_hz=8000))
    store = FakeConvertedModelStore()
    trainer = WhisperLoraTrainer(
        settings=WhisperSettings(replay_fraction=0.0),
        example_loader=loader,
        store=store,
    )
    result = trainer.train({**manifest, "hf_path": str(path)}, core_settings)
    payload = json.dumps(store.load(result["artifact_uri"]).to_dict(), sort_keys=True)
    for secret in secrets:
        assert secret not in payload
    assert result["metrics"]["examples_8k"] == 1


def test_protected_hashes_are_checked_before_and_after_training(
    manifest: dict[str, Any],
    core_settings: TrainerSettings,
    primary_examples: tuple[Any, ...],
) -> None:
    guard = ReadOnlyHashGuard()
    before = guard.verify()
    make_trainer(primary_examples, ()).train(manifest, core_settings)
    after = guard.verify()
    guard.assert_unchanged(before, after)
    assert before.golden_content_hash == EXPECTED_GOLDEN_HASH
    assert before.fixture_sha256 == EXPECTED_FIXTURE_HASH
