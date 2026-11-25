"""End-to-end scoring, ranking, store and CLI tests for P2-01."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import pytest

from services.triage.scoring.explanation import ScoreExplanation
from services.triage.scoring.mismatch import detect_final_record_mismatch
from services.triage.scoring.repair import load_prior_turns
from services.triage.scoring.scoring import (
    ScoringEngine,
    ScoringError,
    load_fixture_scores,
    main,
    score_turn,
)
from services.triage.scoring.settings import TriageScoringSettings
from voiceloop_common import validate

KNOWN_FAILURE_TYPES = {"stt_wrong", "parser_wrong", "unusable_audio"}
CLEAN_TYPES = {"clean", "both_fine"}


def _fixture_loader(
    repo_root: Path,
) -> Callable[[Mapping[str, Any]], Sequence[Mapping[str, Any]]]:
    """Build a loader that resolves fixture context URIs against the repo root."""

    def _load(event: Mapping[str, Any]) -> Sequence[Mapping[str, Any]]:
        return load_prior_turns(event, root=repo_root)

    return _load


def test_fixture_events_are_schema_valid(
    load_labels: Callable[[], list[dict[str, Any]]],
    load_event: Callable[[str | Path], dict[str, Any]],
) -> None:
    """Scoring consumes the authoritative turn-event contract."""
    label = next(row for row in load_labels() if row["case_id"] == "misheard_surname")
    validate(load_event(label["event_path"]), "turn_event")


def test_known_failures_rank_above_clean_turns_by_mean_score(
    repo_root: Path, turns_path: Path
) -> None:
    """Fixture failure labels order above clean turns using event data only."""
    rows = load_fixture_scores(turns_path, prior_turn_loader=_fixture_loader(repo_root))
    assert len(rows) == 36
    failures = [row for row in rows if row.label["failure_type"] in KNOWN_FAILURE_TYPES]
    clean = [row for row in rows if row.label["failure_type"] in CLEAN_TYPES]
    assert len(failures) >= 9
    assert len(clean) >= 20
    failure_mean = sum(row.explanation.score for row in failures) / len(failures)
    clean_mean = sum(row.explanation.score for row in clean) / len(clean)
    assert failure_mean > clean_mean
    assert min(row.explanation.score for row in failures) > max(
        row.explanation.score for row in clean
    )

    by_case = {row.label["case_id"]: row for row in failures if row.label["case_id"]}
    for case_id in (
        "misheard_surname",
        "date_confusion",
        "schema_invalid_parse",
        "silence_hallucination",
        "misheard_guests",
        "nights_misparsed",
    ):
        assert by_case[case_id].explanation.score > 0.0

    silence = [row for row in failures if row.label["failure_type"] == "unusable_audio"]
    assert len(silence) == 4
    assert all(row.explanation.score > 0.0 for row in silence)


def test_each_failure_explanation_lists_its_fired_signals(
    repo_root: Path, turns_path: Path
) -> None:
    """Every non-zero score identifies the signals, weights and reasons."""
    rows = load_fixture_scores(turns_path, prior_turn_loader=_fixture_loader(repo_root))
    for row in rows:
        explanation = row.explanation
        assert 0.0 <= explanation.score <= 1.0
        if row.label["failure_type"] in KNOWN_FAILURE_TYPES:
            assert explanation.signals
        for signal in explanation.signals:
            assert signal.name
            assert signal.reason
            assert signal.weight >= 0.0
            assert 0.0 < signal.severity <= 1.0
            assert signal.contribution == round(signal.weight * signal.severity, 6)
        assert explanation.signal_names == tuple(signal.name for signal in explanation.signals)


def test_no_fixture_final_record_mismatch_is_faked(
    load_labels: Callable[[], list[dict[str, Any]]],
    load_event: Callable[[str | Path], dict[str, Any]],
    turns_path: Path,
) -> None:
    """Document that fixture final records agree with their parser outputs.

    The final-record signal is exercised by a derived mutation in
    ``test_mismatch.py``; no fixture ground truth is altered here.
    """
    settings = TriageScoringSettings()
    for label in load_labels():
        event = load_event(label["event_path"])
        assert detect_final_record_mismatch(event, settings) == []


def test_explanations_are_stored_per_turn_and_idempotent(
    load_labels: Callable[[], list[dict[str, Any]]],
    load_event: Callable[[str | Path], dict[str, Any]],
) -> None:
    """Upserting the same event keeps one explanation keyed by event_id."""
    from services.triage.scoring import InMemoryExplanationStore

    label = next(row for row in load_labels() if row["case_id"] == "misheard_surname")
    event = load_event(label["event_path"])
    store = InMemoryExplanationStore()
    engine = ScoringEngine(store=store)
    first = engine.score(event)
    second = engine.score(event)
    assert first == second
    assert len(store.all()) == 1
    assert store.get(str(event["event_id"])) == first
    assert store.get("missing") is None


def test_weights_are_configurable(turns_path: Path) -> None:
    """Changing a weight changes both the score and the recorded explanation."""
    rows = load_fixture_scores(turns_path)
    misheard = next(row for row in rows if row.label["case_id"] == "misheard_surname")
    baseline = misheard.explanation.score
    quiet_confidence = TriageScoringSettings(weight_low_word_confidence=0.0)
    rows = load_fixture_scores(turns_path, settings=quiet_confidence)
    adjusted = next(row for row in rows if row.label["case_id"] == "misheard_surname")
    assert adjusted.explanation.score < baseline
    low_conf = next(
        signal for signal in adjusted.explanation.signals if signal.name == "low_word_confidence"
    )
    assert low_conf.weight == 0.0
    strong_mismatch = TriageScoringSettings(weight_final_record_mismatch=2.0)
    assert strong_mismatch.weight_final_record_mismatch > max(
        strong_mismatch.weight_hypothesis_parser_mismatch,
        strong_mismatch.weight_decoder_stats,
    )


def test_settings_use_triage_env_prefix(monkeypatch: pytest.MonkeyPatch) -> None:
    """Weights read from VOICELOOP_TRIAGE_* environment variables."""
    monkeypatch.setenv("VOICELOOP_TRIAGE_WEIGHT_DECODER_STATS", "0.77")
    settings = TriageScoringSettings()
    assert settings.weight_decoder_stats == 0.77


def test_explanation_json_round_trip(
    load_labels: Callable[[], list[dict[str, Any]]],
    load_event: Callable[[str | Path], dict[str, Any]],
) -> None:
    """Explanations survive JSON serialisation for storage or review handoff."""
    label = next(row for row in load_labels() if row["case_id"] == "date_confusion")
    explanation = score_turn(load_event(label["event_path"]))
    assert ScoreExplanation.from_dict(explanation.to_dict()) == explanation


@pytest.mark.parametrize(
    "event",
    [
        {"call_id": "c", "turn_index": 0, "stt": {"text": ""}},
        {"event_id": "e", "turn_index": 0, "stt": {"text": ""}},
        {"event_id": "e", "call_id": "c", "turn_index": -1, "stt": {"text": ""}},
    ],
)
def test_invalid_identifiers_raise_scoring_error(event: dict[str, Any]) -> None:
    """Missing required identifiers are rejected explicitly."""
    with pytest.raises(ScoringError):
        score_turn(event)


def test_cli_writes_explanations(tmp_path: Path, turns_path: Path) -> None:
    """The CLI produces one explanation per fixture turn without mutating fixtures."""
    output = tmp_path / "scores.json"
    assert main(["--fixtures", str(turns_path), "--output", str(output)]) == 0
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["turns"] == 36
    assert len(report["explanations"]) == 36
    assert report["by_failure_type"]["unusable_audio"]["turns"] == 4
