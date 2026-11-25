"""Tests for the deterministic P4-02 runner, replay seams and CLI."""

from __future__ import annotations

import io
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from infra.pipelines.drift import (
    EXPECTED_FIXTURE_SHA256,
    SYNTHETIC_LABEL,
    DriftSettings,
    InMemoryAuditSink,
    InMemoryOutcomeWindowSource,
    InMemoryScoreWindowSource,
    ManualOverride,
    RecordingPipeline,
    build_fixture_replay_sources,
    fixture_sha256,
    main,
    run_once,
    verify_fixture_hashes,
    write_report,
)
from voiceloop_common import Clock, ValidationError

REPO_ROOT = Path(__file__).resolve().parents[3]


def _sources(
    score_window: Any, outcome_window: Any
) -> tuple[InMemoryScoreWindowSource, InMemoryOutcomeWindowSource]:
    """Wrap injected windows in deterministic fake sources."""
    return (
        InMemoryScoreWindowSource(score_window),
        InMemoryOutcomeWindowSource(outcome_window),
    )


def test_drift_and_volume_invoke_fake_pipeline(
    drifted_score_window: Any,
    drifted_outcome_window: Any,
    drift_settings: DriftSettings,
    fixed_clock: Clock,
) -> None:
    scores, outcomes = _sources(drifted_score_window, drifted_outcome_window)
    pipeline = RecordingPipeline()
    report = run_once(
        score_source=scores,
        outcome_source=outcomes,
        pipeline=pipeline,
        new_verified_count=5,
        settings=drift_settings,
        clock=fixed_clock,
    )
    assert report.invoked is True
    assert report.decision.triggered is True
    assert [reason.value for reason in report.decision.reasons] == ["drift", "data_volume"]
    assert len(pipeline.invocations) == 1
    decision, evidence = pipeline.invocations[0]
    assert decision["triggered"] is True
    assert evidence["drifted"] is True
    payload = report.to_dict()
    assert payload["label"] == SYNTHETIC_LABEL
    assert payload["no_training_performed"] is True
    assert payload["pipeline_result"]["training_performed"] is False


def test_volume_without_drift_never_invokes_pipeline(
    stable_score_window: Any,
    stable_outcome_window: Any,
    drift_settings: DriftSettings,
    fixed_clock: Clock,
) -> None:
    scores, outcomes = _sources(stable_score_window, stable_outcome_window)
    pipeline = RecordingPipeline()
    report = run_once(
        score_source=scores,
        outcome_source=outcomes,
        pipeline=pipeline,
        new_verified_count=100,
        settings=drift_settings,
        clock=fixed_clock,
        scheduled_tick=True,
    )
    assert report.invoked is False
    assert pipeline.invocations == []
    assert report.decision.reasons == ()
    assert report.decision.to_dict()["calendar_tick_is_trigger"] is False


def test_manual_override_invokes_and_audits(
    stable_score_window: Any,
    stable_outcome_window: Any,
    drift_settings: DriftSettings,
    fixed_clock: Clock,
) -> None:
    scores, outcomes = _sources(stable_score_window, stable_outcome_window)
    sink = InMemoryAuditSink()
    report = run_once(
        score_source=scores,
        outcome_source=outcomes,
        pipeline=RecordingPipeline(),
        new_verified_count=7,
        settings=drift_settings,
        manual_override=ManualOverride("reviewer-1a2b3c4d", "review_backlog"),
        audit_sink=sink,
        clock=fixed_clock,
    )
    assert report.invoked is True
    assert report.decision.reason is not None
    assert report.decision.reason.value == "manual_override"
    assert len(sink.entries) == 1
    assert sink.entries[0]["actor_id"] == "reviewer-1a2b3c4d"
    assert sink.entries[0]["reason_code"] == "review_backlog"


def test_reports_are_deterministic_with_a_frozen_clock(
    drifted_score_window: Any,
    drifted_outcome_window: Any,
    drift_settings: DriftSettings,
    fixed_clock: Clock,
) -> None:
    scores, outcomes = _sources(drifted_score_window, drifted_outcome_window)
    first = run_once(
        score_source=scores,
        outcome_source=outcomes,
        pipeline=RecordingPipeline(),
        new_verified_count=5,
        settings=drift_settings,
        clock=fixed_clock,
    )
    second = run_once(
        score_source=scores,
        outcome_source=outcomes,
        pipeline=RecordingPipeline(),
        new_verified_count=5,
        settings=drift_settings,
        clock=fixed_clock,
    )
    assert json.dumps(first.to_dict(), sort_keys=True) == json.dumps(
        second.to_dict(), sort_keys=True
    )


def test_fixture_replay_is_deterministic_and_split_by_call(drift_settings: DriftSettings) -> None:
    first_scores, first_outcomes = build_fixture_replay_sources(settings=drift_settings)
    second_scores, second_outcomes = build_fixture_replay_sources(settings=drift_settings)
    assert first_scores.load() == second_scores.load()
    assert first_outcomes.load() == second_outcomes.load()
    score_window = first_scores.load()
    outcome_window = first_outcomes.load()
    assert score_window.baseline != score_window.current
    assert len(score_window.baseline) == len(score_window.current) == 18
    assert len(outcome_window.baseline) == len(outcome_window.current) == 3

    labels_path = REPO_ROOT / "tests/fixtures/turns.jsonl"
    rows = [json.loads(line) for line in labels_path.read_text(encoding="utf-8").splitlines()]
    calls = sorted({str(row["call_id"]) for row in rows})
    baseline_calls = set(calls[: len(calls) // 2])
    expected_baseline_outcomes = sum(
        1 for row in rows if row["call_id"] in baseline_calls and row.get("expected_outcome")
    )
    assert expected_baseline_outcomes == len(outcome_window.baseline)


def test_replayed_fixture_drift_triggers_pipeline(
    drift_settings: DriftSettings, fixed_clock: Clock
) -> None:
    def replay_score(event: Any) -> float:
        """Map fixture call halves to a synthetic low/high score distribution."""
        return 0.05 if str(event.get("call_id", "")) <= "call-0003" else 0.85

    scores, outcomes = build_fixture_replay_sources(settings=drift_settings, scorer=replay_score)
    pipeline = RecordingPipeline()
    report = run_once(
        score_source=scores,
        outcome_source=outcomes,
        pipeline=pipeline,
        new_verified_count=5,
        settings=drift_settings,
        clock=fixed_clock,
    )
    assert report.decision.drifted is True
    assert report.invoked is True
    assert len(pipeline.invocations) == 1
    assert "drift" in report.decision.to_dict()["reasons"]


def test_hash_helpers_are_read_only_and_match_expected() -> None:
    assert fixture_sha256("tests/fixtures/turns.jsonl") == EXPECTED_FIXTURE_SHA256
    assert verify_fixture_hashes() == {
        "fixture_sha256_ok": True,
        "golden_content_hash_ok": True,
    }


def test_cli_emits_labelled_decision_and_report(
    drifted_score_window: Any,
    drifted_outcome_window: Any,
    drift_settings: DriftSettings,
    fixed_clock: Clock,
) -> None:
    scores, outcomes = _sources(drifted_score_window, drifted_outcome_window)
    pipeline = RecordingPipeline()
    out = io.StringIO()
    code = main(
        ["--new-verified", "5", "--min-new-verified", "5", "--scheduled-tick"],
        settings=drift_settings,
        score_source=scores,
        outcome_source=outcomes,
        pipeline=pipeline,
        clock=fixed_clock,
        stdout=out,
    )
    assert code == 0
    assert len(pipeline.invocations) == 1
    lines = out.getvalue().splitlines()
    assert len(lines) == 2
    assert lines[0].startswith("DECISION: ")
    assert lines[1].startswith("REPORT: ")
    decision = json.loads(lines[0].removeprefix("DECISION: "))
    report = json.loads(lines[1].removeprefix("REPORT: "))
    assert decision["label"] == SYNTHETIC_LABEL
    assert decision["scheduled_tick"] is True
    assert decision["triggered"] is True
    assert report["invoked"] is True


def test_cli_calendar_only_and_invalid_overrides(
    stable_score_window: Any,
    stable_outcome_window: Any,
    drift_settings: DriftSettings,
    fixed_clock: Clock,
) -> None:
    scores, outcomes = _sources(stable_score_window, stable_outcome_window)
    out = io.StringIO()
    code = main(
        ["--new-verified", "100", "--min-new-verified", "5", "--scheduled-tick"],
        settings=drift_settings,
        score_source=scores,
        outcome_source=outcomes,
        pipeline=RecordingPipeline(),
        clock=fixed_clock,
        stdout=out,
    )
    assert code == 0
    assert (
        json.loads(out.getvalue().splitlines()[0].removeprefix("DECISION: "))["triggered"] is False
    )

    for argv in (
        ["--min-new-verified", "0"],
        ["--manual-actor", "operator-1a2b3c4d"],
        ["--manual-reason", "lead_authorized"],
    ):
        error_out = io.StringIO()
        assert (
            main(
                argv,
                settings=drift_settings,
                score_source=scores,
                outcome_source=outcomes,
                pipeline=RecordingPipeline(),
                clock=fixed_clock,
                stdout=error_out,
            )
            == 2
        )
        assert error_out.getvalue().startswith("ERROR: ")


def test_cli_manual_override_writes_audit_line(
    stable_score_window: Any,
    stable_outcome_window: Any,
    drift_settings: DriftSettings,
    fixed_clock: Clock,
) -> None:
    scores, outcomes = _sources(stable_score_window, stable_outcome_window)
    out = io.StringIO()
    code = main(
        [
            "--new-verified",
            "6",
            "--min-new-verified",
            "5",
            "--manual-actor",
            "operator-1a2b3c4d",
            "--manual-reason",
            "incident_replay",
        ],
        settings=drift_settings,
        score_source=scores,
        outcome_source=outcomes,
        pipeline=RecordingPipeline(),
        clock=fixed_clock,
        stdout=out,
    )
    assert code == 0
    lines = out.getvalue().splitlines()
    assert len(lines) == 3
    audit = json.loads(lines[1].removeprefix("AUDIT: "))
    assert audit["actor_id"] == "operator-1a2b3c4d"
    assert audit["reason_code"] == "incident_replay"


def test_write_report_uses_sorted_json_and_rejects_pii(
    drifted_score_window: Any,
    drifted_outcome_window: Any,
    drift_settings: DriftSettings,
    fixed_clock: Clock,
    tmp_path: Path,
) -> None:
    scores, outcomes = _sources(drifted_score_window, drifted_outcome_window)
    report = run_once(
        score_source=scores,
        outcome_source=outcomes,
        pipeline=RecordingPipeline(),
        new_verified_count=5,
        settings=drift_settings,
        clock=fixed_clock,
    )
    destination = write_report(report.to_dict(), tmp_path / "nested" / "report.json")
    payload = json.loads(destination.read_text(encoding="utf-8"))
    assert payload == report.to_dict()
    assert destination.read_text(encoding="utf-8").endswith("\n")
    with pytest.raises(ValidationError):
        write_report({"email": "ada.fixture@example.com"}, tmp_path / "bad.json")


def test_run_once_checks_last_run_at_from_record(
    drifted_score_window: Any,
    drifted_outcome_window: Any,
    drift_settings: DriftSettings,
    fixed_clock: Clock,
) -> None:
    scores, outcomes = _sources(drifted_score_window, drifted_outcome_window)
    report = run_once(
        score_source=scores,
        outcome_source=outcomes,
        pipeline=RecordingPipeline(),
        new_verified_count=5,
        settings=drift_settings,
        clock=fixed_clock,
        last_run_at=datetime(2026, 11, 24, 12, tzinfo=UTC),
    )
    assert report.decision.to_dict()["last_run_at"] == "2026-11-24T12:00:00+00:00"
