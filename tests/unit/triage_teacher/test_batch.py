"""Batch teacher transcription, hashing and read-only fixture tests."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from services.triage.teacher.batch import (
    DeterministicFixtureTeacher,
    FixtureTurnSource,
    InMemoryTurnSource,
    TurnRecord,
    canonical_hash,
    fixture_inputs_digest,
    run_batch,
)
from services.triage.teacher.settings import TeacherSettings
from voiceloop_common import ValidationError


def _event(
    event_id: str,
    *,
    text: str = "Hello world",
    confidences: tuple[float, float] = (0.95, 0.92),
    no_speech_prob: float = 0.002,
) -> dict[str, Any]:
    """Build a minimal schema-shaped turn event for fake-seam tests."""
    words = [
        {"w": "Hello", "conf": confidences[0]},
        {"w": "world", "conf": confidences[1]},
    ]
    return {
        "event_id": event_id,
        "call_id": "call-test",
        "turn_index": 0,
        "audio": {"uri": "fake://audio/tone.wav"},
        "stt": {
            "text": text,
            "words": words,
            "decoder_stats": {"no_speech_prob": no_speech_prob},
        },
    }


def test_canonical_hash_is_stable_and_json_only() -> None:
    """Canonical hashing ignores key order but rejects non-JSON payloads."""
    left = canonical_hash({"b": 2, "a": 1})
    right = canonical_hash({"a": 1, "b": 2})
    assert left == right
    assert len(left) == 64
    assert canonical_hash({"a": 1}) != canonical_hash({"a": 2})
    with pytest.raises(ValidationError, match="non-JSON"):
        canonical_hash({"bad": object()})


def test_fixture_inputs_digest_is_stable(repo_root: Path) -> None:
    """The read-only fixture text digest is deterministic across calls."""
    first = fixture_inputs_digest(root=repo_root)
    assert first == fixture_inputs_digest(root=repo_root)
    assert len(first) == 64


def test_fixture_turn_source_is_read_only_and_label_free(repo_root: Path) -> None:
    """The fixture source loads 36 turns without reading labels or audio bytes."""
    records = tuple(FixtureTurnSource(root=repo_root).iter_turns())
    assert len(records) == 36
    assert len({record.event_id for record in records}) == 36
    assert all("failure_type" not in record.event for record in records)
    assert all("expected_transcript" not in record.event for record in records)
    assert all(record.audio_uri.startswith("tests/fixtures/audio/") for record in records)


def test_fake_teacher_masks_low_confidence_word() -> None:
    """The stand-in masks the lowest-confidence token without label access."""
    record = TurnRecord.from_event(_event("ev-1", confidences=(0.95, 0.41)))
    transcript = DeterministicFixtureTeacher().transcribe(record)
    assert transcript.text == "Hello [unclear]"
    assert "fixture stand-in" in transcript.reason
    assert transcript.content_hash == canonical_hash(
        {
            "kind": "voiceloop.teacher.transcript",
            "schema_version": "internal-1.0.0",
            "event_id": "ev-1",
            "call_id": "call-test",
            "turn_index": 0,
            "audio_uri": "fake://audio/tone.wav",
            "model_id": "fake://teacher/fixture-standin-v1",
            "text": "Hello [unclear]",
        }
    )


def test_fake_teacher_returns_empty_transcript_for_no_speech() -> None:
    """High no-speech probability yields a deterministic empty transcript."""
    record = TurnRecord.from_event(_event("ev-2", no_speech_prob=0.97))
    transcript = DeterministicFixtureTeacher().transcribe(record)
    assert transcript.text == ""
    assert "empty teacher transcript" in transcript.reason


def test_batch_result_has_hashes_but_no_audio_bytes(repo_root: Path) -> None:
    """The batch artifact contains text references and hashes only."""
    settings = TeacherSettings(fixtures_path="tests/fixtures/turns.jsonl", hash_algorithm="sha256")
    batch = run_batch(
        FixtureTurnSource(root=repo_root),
        DeterministicFixtureTeacher(),
        settings,
        fixtures_sha256=fixture_inputs_digest(root=repo_root),
    )
    assert (batch.turns, batch.calls) == (36, 6)
    serialized = json.dumps([record.to_dict() for record in batch.records], ensure_ascii=False)
    assert "audio_bytes" not in serialized
    assert ".wav" in serialized
    assert all(len(record.teacher_text_hash) == 64 for record in batch.records)
    assert all(len(record.production_text_hash) == 64 for record in batch.records)
    again = run_batch(
        FixtureTurnSource(root=repo_root),
        DeterministicFixtureTeacher(),
        settings,
        fixtures_sha256=fixture_inputs_digest(root=repo_root),
    )
    assert again == batch


def test_batch_rejects_duplicate_event_ids() -> None:
    """Duplicate lineage keys fail closed instead of silently overwriting."""
    duplicate = _event("ev-dup")
    records = [TurnRecord.from_event(duplicate), TurnRecord.from_event(duplicate)]
    source = InMemoryTurnSource(records)
    with pytest.raises(ValidationError, match="duplicate event_id"):
        run_batch(source, DeterministicFixtureTeacher())
