"""Synthetic precision evaluation, provenance checks and internal report."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from services.triage.teacher.batch import (
    BatchResult,
    FixtureTurnSource,
    TeacherBatchRecord,
    TurnRecord,
    fixture_inputs_digest,
)
from services.triage.teacher.report import (
    FixtureReviewedLabelSource,
    InMemoryReportSink,
    InMemoryReviewedLabelSource,
    JsonFileReportSink,
    ReviewedLabel,
    _intervals_overlap,
    _wilson_interval,
    build_report,
    evaluate_precision,
    validate_report,
    verify_read_only_inputs,
)
from services.triage.teacher.scorer import (
    BaselineQueueSelector,
    BaselineScore,
    QueueCandidate,
    TeacherQueueSelector,
    TeacherSignal,
)
from services.triage.teacher.settings import (
    ARTIFACT_TYPE,
    RUN_LABEL,
    SIGNAL_NAME,
    TeacherSettings,
)
from voiceloop_common import ValidationError


def _turn(event_id: str, call_id: str, turn_index: int) -> TurnRecord:
    """Build a label-free turn record for fake-seam precision tests."""
    return TurnRecord.from_event(
        {
            "event_id": event_id,
            "call_id": call_id,
            "turn_index": turn_index,
            "audio": {"uri": f"fake://audio/{event_id}.wav"},
            "stt": {"text": "stand-in text"},
        }
    )


def _candidate(event_id: str, baseline: float, disagreement: float) -> QueueCandidate:
    """Build a candidate with one explicit teacher signal."""
    return QueueCandidate(
        event_id=event_id,
        call_id=f"call-{event_id}",
        turn_index=0,
        baseline=BaselineScore(event_id=event_id, score=baseline, signals=()),
        teacher_signal=TeacherSignal(
            event_id=event_id,
            name=SIGNAL_NAME,
            score=disagreement,
            weight=1.0,
            weighted=disagreement,
            reason="fixture stand-in",
            production_text_hash="a" * 64,
            teacher_text_hash="b" * 64,
        ),
    )


def _label(event_id: str, review: bool, transcription: bool) -> ReviewedLabel:
    """Build an explicitly synthetic reviewed label."""
    return ReviewedLabel(
        event_id=event_id,
        call_id="call-fake",
        turn_index=0,
        routing_label="stt_wrong" if transcription else "clean",
        needs_review=review,
        needs_transcription_review=transcription,
    )


@pytest.fixture
def fake_precision_inputs() -> tuple[
    tuple[TurnRecord, ...], tuple[QueueCandidate, ...], InMemoryReviewedLabelSource
]:
    """Return a small hand-built setup with a known precision improvement."""
    turns = tuple(
        [
            _turn("ev-1", "call-1", 0),
            _turn("ev-2", "call-2", 0),
            _turn("ev-3", "call-3", 0),
            _turn("ev-4", "call-4", 0),
        ]
    )
    candidates = tuple(
        [
            _candidate("ev-1", 0.4, 1.0),
            _candidate("ev-2", 0.3, 0.0),
            _candidate("ev-3", 0.9, 0.0),
            _candidate("ev-4", 0.2, 0.8),
        ]
    )
    labels = InMemoryReviewedLabelSource(
        {
            "ev-1": _label("ev-1", True, True),
            "ev-2": _label("ev-2", True, False),
            "ev-3": _label("ev-3", False, False),
            "ev-4": _label("ev-4", True, True),
        }
    )
    return turns, candidates, labels


def test_precision_comparison_counts_and_honest_delta(
    fake_precision_inputs: tuple[
        tuple[TurnRecord, ...], tuple[QueueCandidate, ...], InMemoryReviewedLabelSource
    ],
    settings: TeacherSettings,
) -> None:
    """Both selectors see identical synthetic labels; counts and deltas are exact."""
    turns, candidates, labels = fake_precision_inputs
    config = TeacherSettings(queue_size=2)
    comparison = evaluate_precision(
        candidates, turns, labels, BaselineQueueSelector(), TeacherQueueSelector(), config
    )
    assert (comparison.turns, comparison.calls, comparison.queue_size) == (4, 4, 2)
    assert comparison.baseline.selected == ("ev-3", "ev-1")
    assert comparison.enhanced.selected == ("ev-1", "ev-4")
    assert comparison.baseline.precision_overall == 0.5
    assert comparison.enhanced.precision_overall == 1.0
    assert comparison.delta_overall == 0.5
    assert comparison.baseline.positives_overall == 3
    assert comparison.baseline.positives_transcription == 2
    assert comparison.enhanced.positives_overall == 3
    assert comparison.baseline.recall_overall == pytest.approx(1 / 3, abs=1e-6)
    assert comparison.enhanced.recall_overall == pytest.approx(2 / 3, abs=1e-6)
    assert "SYNTHETIC/FIXTURE STAND-IN" in comparison.claim
    assert "production" in comparison.claim.lower()


def test_depth_curve_is_exploratory_with_counts_and_ci(
    fake_precision_inputs: tuple[
        tuple[TurnRecord, ...], tuple[QueueCandidate, ...], InMemoryReviewedLabelSource
    ],
) -> None:
    """The full curve is disclosed with counts/CIs and no improvement claim."""
    turns, candidates, labels = fake_precision_inputs
    config = TeacherSettings(queue_size=4)
    comparison = evaluate_precision(
        candidates, turns, labels, BaselineQueueSelector(), TeacherQueueSelector(), config
    )
    assert [point.queue_size for point in comparison.depth_curve] == [1, 2, 3, 4]
    assert comparison.depth_curve[-1].baseline == comparison.baseline
    assert comparison.depth_curve[0].delta_overall == 1.0
    payload = comparison.to_dict()
    assert len(payload["depth_curve"]) == 4
    for banned in (
        "improved_at_smaller_depth",
        "improved_overall",
        "improved_transcription_review",
    ):
        assert banned not in payload
    assert "NOT DONE" in payload["exploratory_note"]
    for point in payload["depth_curve"]:
        assert point["n"] == point["queue_size"]
        assert 0 <= point["baseline_tp_overall"] <= point["n"]
        assert 0 <= point["enhanced_tp_overall"] <= point["n"]
        assert 0 <= point["baseline_tp_transcription"] <= point["n"]
        assert 0 <= point["enhanced_tp_transcription"] <= point["n"]
        assert len(point["baseline_precision_overall_ci95"]) == 2
        assert len(point["enhanced_precision_transcription_ci95"]) == 2
        assert isinstance(point["ci95_overlap"]["overall"], bool)
        assert "EXPLORATORY ONLY" in point["confidence_note"]
        assert "insufficient n" in point["confidence_note"]


def test_wilson_interval_and_overlap_helper() -> None:
    """Small-sample intervals are bounded and overlap detection is exact."""
    low, high = _wilson_interval(0, 1)
    assert low == 0.0
    assert 0.79 < high < 0.80
    low, high = _wilson_interval(1, 1)
    assert 0.20 < low < 0.21
    assert high == 1.0
    assert _wilson_interval(0, 0) == (0.0, 1.0)
    assert _intervals_overlap(_wilson_interval(0, 1), _wilson_interval(1, 1)) is True
    assert _intervals_overlap((0.1, 0.2), (0.3, 0.4)) is False


def test_fixture_reviewed_labels_are_synthetic_only(repo_root: Path) -> None:
    """Fixture labels are clearly synthetic and used only for evaluation."""
    source = FixtureReviewedLabelSource(root=repo_root)
    turns = tuple(FixtureTurnSource(root=repo_root).iter_turns())
    labels = [source.label_for(turn) for turn in turns]
    assert len(labels) == 36
    assert all(label.synthetic for label in labels)
    assert sum(label.needs_review for label in labels) == 10
    assert sum(label.needs_transcription_review for label in labels) == 6


def test_provenance_verifies_seal_and_fixture_hash(settings: TeacherSettings) -> None:
    """Read-only inputs are hash-verified without loading golden labels."""
    provenance = verify_read_only_inputs(settings)
    assert provenance["golden_seal_ok"] is True
    assert provenance["golden_content_hash"] == settings.expected_golden_content_hash
    assert provenance["golden_labels_loaded"] is False
    assert provenance["golden_used_for_training_or_tuning"] is False
    assert provenance["real_teacher_model_used"] is False
    assert provenance["audio_bytes_in_artifact"] is False
    assert provenance["fixtures_sha256"] == fixture_inputs_digest(settings.fixtures_path)
    assert len(provenance["fixtures_sha256"]) == 64


def _fake_batch(fixtures_sha256: str) -> BatchResult:
    """Build a four-record fake batch with content hashes only."""
    records = tuple(
        TeacherBatchRecord(
            event_id=f"ev-{index}",
            call_id=f"call-{index}",
            turn_index=0,
            audio_uri=f"fake://audio/ev-{index}.wav",
            production_text="stand-in text",
            teacher_text="stand-in text",
            production_text_hash="a" * 64,
            teacher_text_hash="b" * 64,
            teacher_model_id="fake://teacher/fixture-standin-v1",
            reason="fixture stand-in",
        )
        for index in range(1, 5)
    )
    return BatchResult(
        run_label=RUN_LABEL,
        teacher_model_id="fake://teacher/fixture-standin-v1",
        hash_algorithm="sha256",
        fixtures_sha256=fixtures_sha256,
        records=records,
    )


def test_report_build_validate_and_write(
    fake_precision_inputs: tuple[
        tuple[TurnRecord, ...], tuple[QueueCandidate, ...], InMemoryReviewedLabelSource
    ],
    settings: TeacherSettings,
    tmp_path: Path,
) -> None:
    """The internal artifact validates and writes as canonical JSON only."""
    turns, candidates, labels = fake_precision_inputs
    provenance = verify_read_only_inputs(settings)
    batch = _fake_batch(provenance["fixtures_sha256"])
    comparison = evaluate_precision(
        candidates, turns, labels, BaselineQueueSelector(), TeacherQueueSelector(), settings
    )
    artifact = build_report(settings, batch, candidates, comparison, provenance)
    validate_report(artifact)
    sink = InMemoryReportSink()
    assert sink.write(artifact) == "memory://teacher-report"
    assert sink.last == artifact

    output = tmp_path / "nested" / "teacher.json"
    assert JsonFileReportSink(output, root=tmp_path).write(artifact) == str(output.resolve())
    written = json.loads(output.read_text(encoding="utf-8"))
    assert written["artifact_type"] == ARTIFACT_TYPE
    assert written["status"] == RUN_LABEL
    assert written["precision"]["reviewed_labels_synthetic"] is True
    written_text = output.read_text(encoding="utf-8")
    assert '"audio_bytes"' not in written_text
    assert "base64" not in written_text


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda artifact: artifact["provenance"].update(real_teacher_model_used=True),
            "real teacher",
        ),
        (
            lambda artifact: artifact["precision"].update(reviewed_labels_synthetic=False),
            "synthetic",
        ),
        (
            lambda artifact: artifact["precision"].update(improved_transcription_review=True),
            "precision claim key",
        ),
        (
            lambda artifact: artifact["precision"]["depth_curve"][0].pop("confidence_note"),
            "exploratory depth_curve",
        ),
        (lambda artifact: artifact.pop("signals"), "teacher report missing"),
    ],
)
def test_report_validation_rejects_false_claims(
    fake_precision_inputs: tuple[
        tuple[TurnRecord, ...], tuple[QueueCandidate, ...], InMemoryReviewedLabelSource
    ],
    settings: TeacherSettings,
    mutate: Any,
    message: str,
) -> None:
    """Tampered artifacts claiming production use or missing fields fail closed."""
    turns, candidates, labels = fake_precision_inputs
    provenance = verify_read_only_inputs(settings)
    batch = _fake_batch(provenance["fixtures_sha256"])
    comparison = evaluate_precision(
        candidates, turns, labels, BaselineQueueSelector(), TeacherQueueSelector(), settings
    )
    artifact = build_report(settings, batch, candidates, comparison, provenance)
    mutate(artifact)
    with pytest.raises(ValidationError, match=message):
        validate_report(artifact)
