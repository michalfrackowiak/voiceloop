"""Dataset seam tests: PII-free rows, replay mixing and 8 kHz tags."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from services.trainers.whisper import (
    EXPECTED_FIXTURE_HASH,
    EXPECTED_GOLDEN_HASH,
    EventSampleRateResolver,
    ReadOnlyHashGuard,
    StaticSampleRateResolver,
    WhisperTrainingExample,
    build_data_manifest,
    example_from_hf_row,
    mix_replay,
    sha256_text,
)
from voiceloop_common import ValidationError

from .conftest import DATASET_HASH


def test_hf_row_becomes_digest_only_and_tags_8k() -> None:
    row: dict[str, Any] = {
        "example_id": "call-0004:3:stt",
        "call_id": "call-0004",
        "kind": "stt",
        "audio_uri": "tests/fixtures/audio/call-0004/turn-003.wav",
        "target_text": "Make it four guests.",
        "duration_ms": 240,
        "replay": False,
        "synthetic": True,
    }
    resolver = StaticSampleRateResolver({"tests/fixtures/audio/call-0004/turn-003.wav": 8000})
    example = example_from_hf_row(row, resolver)
    payload = json.dumps(example.to_dict(), sort_keys=True)
    assert example.sample_rate_hz == 8000
    assert example.telephony_band is True
    assert example.effective_bandwidth_hz == 4000
    assert example.target_text_sha256 == sha256_text("Make it four guests.")
    assert "Make it four guests." not in payload
    assert "turn-003.wav" not in payload
    assert example.to_dict()["sample_rate_policy"] == "preserve_native"
    assert example.to_dict()["resampled"] is False


def test_hf_row_rejects_parser_kind_and_bad_rate() -> None:
    resolver = StaticSampleRateResolver({}, default_hz=16000)
    with pytest.raises(ValidationError, match="kind='stt'"):
        example_from_hf_row({"kind": "parser_clean"}, resolver)
    with pytest.raises(ValidationError, match="target text"):
        example_from_hf_row(
            {
                "kind": "stt",
                "example_id": "e1",
                "call_id": "c1",
                "audio_uri": "fake://audio/e1.wav",
                "target_text": "",
                "duration_ms": 100,
            },
            resolver,
        )


def test_event_resolver_reads_read_only_fixture_metadata() -> None:
    resolver = EventSampleRateResolver()
    assert resolver.resolve("tests/fixtures/audio/call-0004/turn-003.wav") == 8000
    assert resolver.resolve("tests/fixtures/audio/call-0001/turn-001.wav") == 16000


def test_replay_is_deterministic_tagged_and_call_disjoint(
    primary_examples: tuple[WhisperTrainingExample, ...],
    replay_pool: tuple[WhisperTrainingExample, ...],
) -> None:
    first = mix_replay(primary_examples, replay_pool, 0.5, 11)
    second = mix_replay(tuple(reversed(primary_examples)), tuple(reversed(replay_pool)), 0.5, 11)
    assert first == second
    assert len(first) == 4
    assert sum(1 for example in first if example.replay) == 1
    assert sum(1 for example in first if example.source == "replay") == 1
    primary_calls = {example.call_id for example in primary_examples}
    replay_calls = {example.call_id for example in first if example.replay}
    assert not primary_calls & replay_calls
    different_seed = mix_replay(primary_examples, replay_pool, 0.5, 1)
    assert {e.example_id for e in different_seed if e.replay} != {
        e.example_id for e in first if e.replay
    }


def test_replay_exclusions_and_duplicates_fail_closed(
    primary_examples: tuple[WhisperTrainingExample, ...],
    replay_pool: tuple[WhisperTrainingExample, ...],
) -> None:
    with pytest.raises(ValidationError, match="excluded call"):
        mix_replay(
            primary_examples,
            replay_pool,
            0.5,
            1,
            excluded_call_ids=frozenset({"call-train-a"}),
        )
    with pytest.raises(ValidationError, match="duplicate Whisper example_id"):
        mix_replay(primary_examples, (primary_examples[0],), 0.5, 1)
    with pytest.raises(ValidationError, match="replay_fraction"):
        mix_replay(primary_examples, replay_pool, 1.5, 1)


def test_mix_manifest_records_replay_and_sample_rates(
    primary_examples: tuple[WhisperTrainingExample, ...],
    replay_pool: tuple[WhisperTrainingExample, ...],
) -> None:
    mixed = mix_replay(primary_examples, replay_pool, 1.0, 3)
    manifest = build_data_manifest(
        DATASET_HASH, mixed, replay_fraction=1.0, preserve_native_sample_rate=True
    )
    assert manifest.source_dataset_hash == DATASET_HASH
    assert manifest.mix["primary_examples"] == 3
    assert manifest.mix["replay_examples"] == 3
    assert manifest.mix["replay_fraction_realised"] == 0.5
    assert manifest.mix["sample_rates_hz"] == {"16000": 4, "8000": 2}
    assert manifest.mix["telephony_8k_examples"] == 2
    assert manifest.mix["preserve_native_sample_rate"] is True
    assert manifest.synthetic_training_standin is True
    assert "NOT A REAL FINE-TUNE" in manifest.warning
    reversed_manifest = build_data_manifest(
        DATASET_HASH,
        tuple(reversed(mixed)),
        replay_fraction=1.0,
        preserve_native_sample_rate=True,
    )
    assert reversed_manifest.content_hash() == manifest.content_hash()
    with pytest.raises(ValidationError, match="preserve-native"):
        build_data_manifest(
            DATASET_HASH, mixed, replay_fraction=0.0, preserve_native_sample_rate=False
        )


def test_guard_verifies_expected_hashes_read_only() -> None:
    guard = ReadOnlyHashGuard()
    report = guard.verify()
    assert report.golden_content_hash == EXPECTED_GOLDEN_HASH
    assert report.fixture_sha256 == EXPECTED_FIXTURE_HASH
    assert report.ok is True
    guard.assert_unchanged(report, guard.check())


def test_guard_detects_tampered_copies(tmp_path: Path) -> None:
    seal = json.loads(
        Path("services/datasets/golden/artifacts/golden.v1.seal.json").read_text(encoding="utf-8")
    )
    manifest = json.loads(
        Path("services/datasets/golden/artifacts/golden.v1.json").read_text(encoding="utf-8")
    )
    (tmp_path / "seal.json").write_text(json.dumps(seal), encoding="utf-8")
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    fixture = tmp_path / "turns.jsonl"
    fixture.write_text('{"synthetic": true}\n', encoding="utf-8")
    guard = ReadOnlyHashGuard(
        root=tmp_path,
        golden_seal_path=Path("seal.json"),
        golden_manifest_path=Path("manifest.json"),
        fixture_path=Path("turns.jsonl"),
    )
    with pytest.raises(ValidationError, match="protected artifact hashes changed"):
        guard.verify()
    before = guard.check()
    assert before.fixture_ok is False
    fixture.write_text('{"synthetic": false}\n', encoding="utf-8")
    with pytest.raises(ValidationError, match="during training"):
        guard.assert_unchanged(before, guard.check())
