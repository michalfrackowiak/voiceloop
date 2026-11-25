"""Tests for schema-shaped fake word confidences and the triage seam."""

from __future__ import annotations

import pytest

from services.trainers.core import TrainerSettings
from services.trainers.nemo import (
    PROVENANCE_LABEL,
    ConfidenceEmitter,
    InMemorySignalSink,
    NemoManifestRow,
    NemoSettings,
    P2TriageSignalSink,
    ParakeetNeMoTrainer,
    ScoreRecord,
    WordConfidenceOutput,
    build_turn_event,
    fake_word_confidences,
    score_event_with_triage,
)
from services.triage.scoring import InMemoryExplanationStore, ScoringEngine
from voiceloop_common import ValidationError, validate

from .conftest import CALL_IDS, EVENT_IDS


def test_words_are_deterministic_and_schema_shaped(rows: tuple[NemoManifestRow, ...]) -> None:
    row = rows[0]
    first = fake_word_confidences(
        row.text,
        duration_ms=240,
        seed=7,
        model="fixture-parakeet-base",
        transcript_hash=row.fingerprint(),
    )
    second = fake_word_confidences(
        row.text,
        duration_ms=240,
        seed=7,
        model="fixture-parakeet-base",
        transcript_hash=row.fingerprint(),
    )
    assert first == second
    assert first.provenance_label == PROVENANCE_LABEL
    assert len(first.words) == len(row.text.split())
    for word in first.words:
        assert set(word.to_dict()) == {"w", "start_ms", "end_ms", "conf"}
        assert 0 <= word.start_ms <= word.end_ms
        assert 0.0 <= word.conf <= 1.0
    starts = [word.start_ms for word in first.words]
    assert starts == sorted(starts)


def test_different_seed_or_model_changes_fake_confidences(
    rows: tuple[NemoManifestRow, ...],
) -> None:
    row = rows[0]
    base = fake_word_confidences(row.text, duration_ms=240, seed=7, model="model-a")
    other_seed = fake_word_confidences(row.text, duration_ms=240, seed=8, model="model-a")
    other_model = fake_word_confidences(row.text, duration_ms=240, seed=7, model="model-b")
    assert base != other_seed
    assert base != other_model


def test_built_event_validates_against_turn_event_contract(
    rows: tuple[NemoManifestRow, ...], nemo_settings: NemoSettings
) -> None:
    words = fake_word_confidences(
        rows[0].text,
        duration_ms=240,
        seed=7,
        model=nemo_settings.confidence_model_id,
        transcript_hash=rows[0].fingerprint(),
    )
    event = build_turn_event(
        rows[0],
        words=words,
        event_id=EVENT_IDS[0],
        call_id=CALL_IDS[0],
        turn_index=0,
        model_id=nemo_settings.confidence_model_id,
    )
    validate(event, "turn_event")
    assert event["stt"]["words"] == words.as_schema_words()
    assert event["stt"]["text"] == rows[0].text
    # turn_event 1.0.0 forbids extra fields, so the label lives on the wrapper.
    assert "provenance_label" not in event


@pytest.mark.parametrize("invalid_text", ["", "   "])
def test_empty_transcript_fails_closed(invalid_text: str) -> None:
    with pytest.raises(ValidationError):
        fake_word_confidences(invalid_text, duration_ms=240, seed=7, model="model-a")


def test_bad_event_identifiers_fail_closed(
    rows: tuple[NemoManifestRow, ...], nemo_settings: NemoSettings
) -> None:
    words: WordConfidenceOutput = fake_word_confidences(
        rows[0].text, duration_ms=240, seed=7, model=nemo_settings.confidence_model_id
    )
    with pytest.raises(ValidationError):
        build_turn_event(
            rows[0],
            words=words,
            event_id="not-a-uuid",
            call_id=CALL_IDS[0],
            turn_index=0,
            model_id=nemo_settings.confidence_model_id,
        )
    with pytest.raises(ValidationError):
        build_turn_event(
            rows[0],
            words=words,
            event_id=EVENT_IDS[0],
            call_id="",
            turn_index=0,
            model_id=nemo_settings.confidence_model_id,
        )


def test_confidences_flow_into_p2_scoring_without_editing_p2_01(
    rows: tuple[NemoManifestRow, ...],
    nemo_settings: NemoSettings,
    core_settings: TrainerSettings,
) -> None:
    store = InMemoryExplanationStore()
    sink = P2TriageSignalSink(engine=ScoringEngine(store=store))
    trainer = ParakeetNeMoTrainer(settings=nemo_settings)
    emissions = trainer.emit_confidences(
        rows,
        core_settings,
        identifiers=list(zip(EVENT_IDS, CALL_IDS, range(len(rows)), strict=True)),
        sink=sink,
    )

    assert len(emissions) == len(rows)
    for emission in emissions:
        validate(emission.event, "turn_event")
        assert emission.provenance_label == PROVENANCE_LABEL
        assert emission.event["stt"]["words"] == emission.words.as_schema_words()
        assert emission.score.event_id == emission.event["event_id"]
        assert len(store.all()) == len(emissions)
    low_confidence_events = [
        emission for emission in emissions if "low_word_confidence" in emission.score.signal_names
    ]
    assert low_confidence_events
    assert all(emission.score.score > 0.0 for emission in low_confidence_events)


def test_default_emitter_uses_p2_sink(
    rows: tuple[NemoManifestRow, ...],
    nemo_settings: NemoSettings,
    core_settings: TrainerSettings,
) -> None:
    emitter = ConfidenceEmitter(settings=nemo_settings, core_settings=core_settings)
    emission = emitter.emit(rows[0], event_id=EVENT_IDS[0], call_id=CALL_IDS[0], turn_index=0)
    assert isinstance(emitter.sink, P2TriageSignalSink)
    assert emission.score.signal_names


def test_injected_sink_receives_schema_shaped_event(
    rows: tuple[NemoManifestRow, ...],
    nemo_settings: NemoSettings,
    core_settings: TrainerSettings,
) -> None:
    sink = InMemorySignalSink()
    emissions = ParakeetNeMoTrainer(settings=nemo_settings).emit_confidences(
        rows,
        core_settings,
        identifiers=list(zip(EVENT_IDS, CALL_IDS, range(len(rows)), strict=True)),
        sink=sink,
    )
    assert sink.events == tuple(emission.event for emission in emissions)
    for event in sink.events:
        validate(event, "turn_event")
        assert event["stt"]["words"]


def test_score_helper_accepts_any_sink(
    rows: tuple[NemoManifestRow, ...],
    nemo_settings: NemoSettings,
    core_settings: TrainerSettings,
) -> None:
    event = build_turn_event(
        rows[0],
        words=fake_word_confidences(
            rows[0].text,
            duration_ms=240,
            seed=core_settings.seed,
            model=nemo_settings.confidence_model_id,
            transcript_hash=rows[0].fingerprint(),
        ),
        event_id=EVENT_IDS[0],
        call_id=CALL_IDS[0],
        turn_index=0,
        model_id=nemo_settings.confidence_model_id,
    )
    record: ScoreRecord = score_event_with_triage(event, sink=InMemorySignalSink())
    assert record.event_id == EVENT_IDS[0]
    assert record.provenance_label == PROVENANCE_LABEL
