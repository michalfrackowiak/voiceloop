"""Baseline adapter, one-signal join and injectable queue selectors."""

from __future__ import annotations

from pathlib import Path

import pytest

from services.triage.scoring import TriageScoringSettings, load_prior_turns, score_turn
from services.triage.teacher.batch import (
    DeterministicFixtureTeacher,
    FixtureTurnSource,
    TurnRecord,
    canonical_hash,
    fixture_inputs_digest,
    run_batch,
)
from services.triage.teacher.disagreement import (
    EventProductionTranscriptSource,
    InMemoryProductionTranscriptSource,
)
from services.triage.teacher.scorer import (
    BaselineQueueSelector,
    BaselineScore,
    InMemoryBaselineScorer,
    P2_01BaselineScorer,
    QueueCandidate,
    TeacherOnlySelector,
    TeacherQueueSelector,
    TeacherSignal,
    build_candidates,
)
from services.triage.teacher.settings import SIGNAL_NAME, TeacherSettings
from voiceloop_common import ValidationError


def _make_candidate(event_id: str, baseline: float, disagreement: float) -> QueueCandidate:
    """Build an explicit candidate for queue-policy unit tests."""
    return QueueCandidate(
        event_id=event_id,
        call_id="call-test",
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


def test_selectors_consume_signals_through_injected_policy() -> None:
    """Baseline ignores the signal; enhanced and teacher-only consume it."""
    candidates = [
        _make_candidate("ev-high-baseline", 0.9, 0.0),
        _make_candidate("ev-high-teacher", 0.2, 0.8),
        _make_candidate("ev-middle", 0.5, 0.2),
    ]
    assert BaselineQueueSelector().select(candidates, 2) == (
        "ev-high-baseline",
        "ev-middle",
    )
    assert TeacherQueueSelector().select(candidates, 2) == (
        "ev-high-teacher",
        "ev-high-baseline",
    )
    assert TeacherOnlySelector().select(candidates, 2) == (
        "ev-high-teacher",
        "ev-middle",
    )
    assert BaselineQueueSelector().name == "baseline_no_teacher"
    assert TeacherQueueSelector().name == "teacher_enhanced"


def test_build_candidates_joins_one_signal_without_labels(repo_root: Path) -> None:
    """The join yields one candidate per turn with a single weighted signal."""
    settings = TeacherSettings(signal_weight=2.0)
    turns = tuple(FixtureTurnSource(root=repo_root).iter_turns())
    batch = run_batch(
        FixtureTurnSource(root=repo_root),
        DeterministicFixtureTeacher(),
        settings,
        fixtures_sha256=fixture_inputs_digest(root=repo_root),
    )
    scores = {record.event_id: 0.25 for record in turns}
    candidates = build_candidates(
        turns,
        batch,
        EventProductionTranscriptSource(),
        InMemoryBaselineScorer(scores),
        settings,
    )
    assert len(candidates) == 36
    assert all(candidate.baseline.score == 0.25 for candidate in candidates)
    assert all(candidate.teacher_signal.name == SIGNAL_NAME for candidate in candidates)
    assert all(candidate.teacher_signal.weight == 2.0 for candidate in candidates)
    assert candidates[0].teacher_signal.production_text_hash == canonical_hash(
        {
            "kind": "voiceloop.production.transcript",
            "schema_version": "internal-1.0.0",
            "event_id": candidates[0].event_id,
            "text": turns[0].event["stt"]["text"],
        }
    )


def test_build_candidates_rejects_production_hash_mismatch(repo_root: Path) -> None:
    """A divergent production transcript fails closed before queue selection."""
    settings = TeacherSettings()
    turns = tuple(FixtureTurnSource(root=repo_root).iter_turns())
    batch = run_batch(FixtureTurnSource(root=repo_root), DeterministicFixtureTeacher(), settings)
    changed = {turn.event_id: "different text" for turn in turns}
    with pytest.raises(ValidationError, match="hash mismatch"):
        build_candidates(
            turns,
            batch,
            InMemoryProductionTranscriptSource(changed),
            InMemoryBaselineScorer({turn.event_id: 0.1 for turn in turns}),
            settings,
        )


def test_p2_01_baseline_adapter_keeps_teacher_disabled(repo_root: Path) -> None:
    """The baseline adapter calls P2-01 with its teacher seam explicitly off."""
    record = next(FixtureTurnSource(root=repo_root).iter_turns())
    expected = score_turn(
        record.event,
        settings=TriageScoringSettings(teacher_enabled=False),
        prior_turns=load_prior_turns(record.event, root=repo_root),
        persist=False,
    )
    result = P2_01BaselineScorer(root=repo_root).score(record)
    assert result.score == expected.score
    assert result.signals == tuple(signal.name for signal in expected.signals)
    assert not any("teacher" in name for name in result.signals)
    assert result.source == "fake://baseline/p2_01_disabled_teacher"


def test_in_memory_baseline_fake_requires_a_score() -> None:
    """Missing fake scores raise instead of defaulting to a silent value."""
    scorer = InMemoryBaselineScorer({})
    record = TurnRecord.from_event(
        {
            "event_id": "ev-missing",
            "call_id": "call-1",
            "turn_index": 0,
            "audio": {"uri": "fake://audio/tone.wav"},
            "stt": {"text": "hi"},
        }
    )
    with pytest.raises(ValidationError, match="no baseline score"):
        scorer.score(record)
