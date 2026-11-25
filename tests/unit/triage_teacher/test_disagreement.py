"""Deterministic teacher-vs-production disagreement tests."""

from __future__ import annotations

import pytest

from services.triage.teacher.batch import TurnRecord
from services.triage.teacher.disagreement import (
    EventProductionTranscriptSource,
    InMemoryProductionTranscriptSource,
    score_disagreement,
    tokenize,
)
from services.triage.teacher.settings import SIGNAL_NAME
from voiceloop_common import ValidationError


def _score(production: str, teacher: str) -> float:
    """Score two texts with explicit non-empty content hashes."""
    return score_disagreement(
        "ev-1",
        production,
        teacher,
        production_hash="a" * 64,
        teacher_hash="b" * 64,
    ).score


def test_tokenize_normalizes_case_and_punctuation() -> None:
    """Tokenization is lowercase, punctuation-insensitive and stable."""
    assert tokenize("Hello, WORLD! hello") == ("hello", "world", "hello")


def test_disagreement_is_deterministic_and_bounded() -> None:
    """Identical texts score zero; token changes scale by the longer text."""
    assert _score("Hello world", "hello, WORLD!") == 0.0
    assert _score("a b", "a") == 0.5
    assert _score("a b c", "a b") == pytest.approx(1 / 3, abs=1e-6)
    assert 0.0 <= _score("a b c", "x y z q") <= 1.0


def test_disagreement_handles_empty_transcripts() -> None:
    """Both empty agrees; exactly one empty is full disagreement."""
    assert _score("", "") == 0.0
    assert _score("hello world", "") == 1.0
    assert _score("", "hello world") == 1.0


def test_disagreement_requires_hashes() -> None:
    """Hashes are mandatory so artifact lineage cannot be silently dropped."""
    with pytest.raises(ValidationError, match="content hashes"):
        score_disagreement("ev-1", "a", "b", production_hash="", teacher_hash="c")


def test_production_seams_are_fake_and_hash_consistent() -> None:
    """Event and in-memory production seams return labelled, hashed text."""
    record = TurnRecord.from_event(
        {
            "event_id": "ev-1",
            "call_id": "call-1",
            "turn_index": 0,
            "audio": {"uri": "fake://audio/tone.wav"},
            "stt": {"text": "Hello world"},
        }
    )
    event_source = EventProductionTranscriptSource()
    memory_source = InMemoryProductionTranscriptSource({"ev-1": "Hello world"})
    event_transcript = event_source.transcript_for(record)
    memory_transcript = memory_source.transcript_for(record)
    assert event_transcript.text == memory_transcript.text == "Hello world"
    assert event_transcript.content_hash == memory_transcript.content_hash
    assert event_transcript.source.startswith("fake://")
    assert memory_transcript.source.startswith("fake://")


def test_signal_name_is_untuned_constant() -> None:
    """The signal has one stable name that cannot drift per run."""
    result = score_disagreement("ev-1", "a", "b", production_hash="a" * 64, teacher_hash="b" * 64)
    assert result.name == SIGNAL_NAME
    assert result.to_dict()["name"] == SIGNAL_NAME
