"""One-command CLI and fully injected rehearsal orchestration tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from services.triage.teacher.batch import (
    DeterministicFixtureTeacher,
    InMemoryTurnSource,
    TurnRecord,
)
from services.triage.teacher.cli import main, run_teacher_rehearsal
from services.triage.teacher.disagreement import EventProductionTranscriptSource
from services.triage.teacher.report import (
    InMemoryReportSink,
    InMemoryReviewedLabelSource,
    ReviewedLabel,
)
from services.triage.teacher.scorer import (
    BaselineQueueSelector,
    InMemoryBaselineScorer,
    TeacherQueueSelector,
)
from services.triage.teacher.settings import SIGNAL_NAME, TeacherSettings


def _record(
    event_id: str, index: int, *, low_confidence: bool = False, no_speech: bool = False
) -> TurnRecord:
    """Build a minimal label-free turn record with fake teacher cues."""
    words = [
        {"w": "a", "conf": 0.41 if low_confidence else 0.95},
        {"w": "booking", "conf": 0.95},
        {"w": "request", "conf": 0.95},
    ]
    return TurnRecord.from_event(
        {
            "event_id": event_id,
            "call_id": f"call-{index}",
            "turn_index": 0,
            "audio": {"uri": f"fake://audio/{event_id}.wav"},
            "stt": {
                "text": "a booking request",
                "words": words,
                "decoder_stats": {"no_speech_prob": 0.97 if no_speech else 0.002},
            },
        }
    )


def test_rehearsal_accepts_all_injected_fakes() -> None:
    """Every collaborator is injectable and only the report sink is written."""
    turns = tuple(
        _record(
            f"ev-{index}",
            index,
            low_confidence=index == 1,
            no_speech=index == 4,
        )
        for index in range(1, 5)
    )
    config = TeacherSettings(queue_size=2, signal_weight=1.0)
    baseline = InMemoryBaselineScorer({"ev-1": 0.4, "ev-2": 0.3, "ev-3": 0.9, "ev-4": 0.2})
    labels = InMemoryReviewedLabelSource(
        {
            f"ev-{index}": ReviewedLabel(
                event_id=f"ev-{index}",
                call_id=f"call-{index}",
                turn_index=0,
                routing_label="stt_wrong" if index in (1, 4) else "clean",
                needs_review=index in (1, 2, 4),
                needs_transcription_review=index in (1, 4),
            )
            for index in range(1, 5)
        }
    )
    sink = InMemoryReportSink()
    artifact = run_teacher_rehearsal(
        config,
        turns=InMemoryTurnSource(turns),
        teacher=DeterministicFixtureTeacher(),
        production=EventProductionTranscriptSource(),
        baseline=baseline,
        baseline_selector=BaselineQueueSelector(),
        enhanced_selector=TeacherQueueSelector(),
        labels=labels,
        sink=sink,
    )
    assert sink.last == artifact
    assert artifact["precision"]["baseline"]["selected"] == ["ev-3", "ev-1"]
    assert artifact["precision"]["teacher_enhanced"]["selected"] == ["ev-4", "ev-1"]
    assert artifact["precision"]["delta_overall"] == 0.5
    assert all(signal["name"] == SIGNAL_NAME for signal in artifact["signals"])
    serialized = json.dumps(artifact)
    assert '"audio_bytes"' not in serialized
    assert "base64" not in serialized


def test_cli_writes_artifact_and_labels_standin(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The documented one-command run writes a report and prints an honest claim."""
    output = tmp_path / "report.json"
    assert (
        main(
            [
                "--output",
                str(output),
                "--queue-size",
                "5",
                "--signal-weight",
                "1.0",
            ]
        )
        == 0
    )
    artifact = json.loads(output.read_text(encoding="utf-8"))
    assert artifact["artifact_type"] == "voiceloop.teacher.triage.report"
    assert artifact["batch"]["turns"] == 36
    assert artifact["precision"]["reviewed_labels_synthetic"] is True
    printed = capsys.readouterr().out
    assert "SYNTHETIC/FIXTURE STAND-IN" in printed
    assert "not a production precision claim" in printed
    assert "improves" not in printed.lower()
    assert "EXPLORATORY ONLY" in printed
    assert "P4-04 is NOT DONE" in printed
    assert "improved_at_smaller_depth" not in artifact["precision"]
    assert "NOT DONE" in artifact["precision"]["exploratory_note"]


def test_cli_returns_structured_config_error(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Invalid settings produce a structured error instead of a traceback."""
    assert main(["--queue-size", "0"]) == 2
    error = json.loads(capsys.readouterr().err)
    assert error["code"] == "config_error"
    assert "queue_size" in error["message"]
