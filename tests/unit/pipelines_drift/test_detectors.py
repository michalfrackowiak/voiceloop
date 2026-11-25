"""Tests for descriptive drift detectors and report safety."""

from __future__ import annotations

import json
import math

import pytest

from infra.pipelines.drift import (
    SYNTHETIC_LABEL,
    DetectorReport,
    DriftSettings,
    OutcomeStatus,
    OutcomeWindow,
    TriageScoreWindow,
    assert_no_sensitive_text,
    detect_all,
    detect_outcome_rate_drift,
    detect_score_drift,
)
from voiceloop_common import ValidationError


def test_score_drift_reports_counts_and_descriptive_deltas(
    drifted_score_window: TriageScoreWindow, drift_settings: DriftSettings
) -> None:
    signal = detect_score_drift(drifted_score_window, drift_settings)
    assert signal.name == "triage_score_distribution"
    assert signal.baseline_count == 6
    assert signal.current_count == 6
    assert signal.enough_data is True
    assert signal.drifted is True
    assert "score_mean_delta" in signal.reasons
    assert "score_distribution_shift" in signal.reasons
    assert signal.deltas["baseline_mean"] == pytest.approx(0.056667, abs=1e-6)
    assert signal.deltas["current_mean"] == pytest.approx(0.855, abs=1e-6)
    assert signal.deltas["mean_delta"] == pytest.approx(0.798333, abs=1e-6)
    assert signal.deltas["total_variation_distance"] == pytest.approx(1.0)
    # All baseline scores fall in bucket 0 and all current scores in bucket 8.
    assert signal.deltas["bucket_00_delta"] == pytest.approx(-1.0, abs=1e-6)
    assert signal.deltas["bucket_08_delta"] == pytest.approx(1.0, abs=1e-6)


def test_stable_scores_do_not_alarm(
    stable_score_window: TriageScoreWindow, drift_settings: DriftSettings
) -> None:
    signal = detect_score_drift(stable_score_window, drift_settings)
    assert signal.drifted is False
    assert signal.reasons == ()
    assert signal.deltas["mean_delta"] == 0.0
    assert signal.deltas["total_variation_distance"] == 0.0


def test_small_windows_do_not_alarm_but_still_report_deltas(
    drift_settings: DriftSettings,
) -> None:
    signal = detect_score_drift(TriageScoreWindow((0.01,), (0.99,)), drift_settings)
    assert signal.drifted is False
    assert signal.enough_data is False
    assert signal.reasons == ("insufficient_window_samples",)
    assert signal.baseline_count == 1 and signal.current_count == 1
    assert signal.deltas["mean_delta"] == pytest.approx(0.98, abs=1e-6)


def test_empty_score_windows_report_undefined_deltas(drift_settings: DriftSettings) -> None:
    signal = detect_score_drift(TriageScoreWindow((), ()), drift_settings)
    assert signal.deltas["mean_delta"] is None
    assert signal.deltas["baseline_mean"] is None
    assert signal.drifted is False


def test_outcome_rate_drift_reports_counts_and_rates(
    drifted_outcome_window: OutcomeWindow, drift_settings: DriftSettings
) -> None:
    signal = detect_outcome_rate_drift(drifted_outcome_window, drift_settings)
    assert signal.name == "outcome_rates"
    assert signal.baseline_count == 6 and signal.current_count == 6
    assert signal.drifted is True
    assert "outcome_status_rate_shift" in signal.reasons
    assert "degraded_outcome_rate_shift" in signal.reasons
    assert signal.deltas["rate_confirmed_baseline"] == pytest.approx(5 / 6, abs=1e-6)
    assert signal.deltas["rate_confirmed_current"] == 0.0
    assert signal.deltas["rate_abandoned_delta"] == pytest.approx(4 / 6, abs=1e-6)
    assert signal.deltas["degraded_rate_baseline"] == pytest.approx(1 / 6, abs=1e-6)
    assert signal.deltas["degraded_rate_current"] == 1.0


def test_stable_outcomes_do_not_alarm(
    stable_outcome_window: OutcomeWindow, drift_settings: DriftSettings
) -> None:
    signal = detect_outcome_rate_drift(stable_outcome_window, drift_settings)
    assert signal.drifted is False
    assert signal.reasons == ()
    assert signal.deltas["rate_confirmed_delta"] == 0.0


def test_outcome_window_with_few_observations_does_not_alarm(
    drift_settings: DriftSettings,
) -> None:
    signal = detect_outcome_rate_drift(
        OutcomeWindow((OutcomeStatus.CONFIRMED,), (OutcomeStatus.ABANDONED,)), drift_settings
    )
    assert signal.drifted is False
    assert signal.enough_data is False
    assert signal.reasons == ("insufficient_window_samples",)
    assert signal.deltas["degraded_rate_delta"] == 1.0


@pytest.mark.parametrize("bad_score", [math.nan, math.inf, -0.01, 1.1, True, "0.5"])
def test_invalid_scores_are_rejected(bad_score: object, drift_settings: DriftSettings) -> None:
    with pytest.raises(ValidationError):
        detect_score_drift(TriageScoreWindow((bad_score,), ()), drift_settings)  # type: ignore[arg-type]


def test_invalid_outcomes_are_rejected(drift_settings: DriftSettings) -> None:
    with pytest.raises(ValidationError):
        detect_outcome_rate_drift(OutcomeWindow(("confirmed",), ("maybe",)), drift_settings)


def test_combined_report_is_labelled_and_has_no_raw_observations(
    drifted_score_window: TriageScoreWindow,
    drifted_outcome_window: OutcomeWindow,
    drift_settings: DriftSettings,
) -> None:
    report = detect_all(drifted_score_window, drifted_outcome_window, drift_settings)
    assert isinstance(report, DetectorReport)
    assert report.drifted is True
    payload = report.to_dict()
    assert payload["label"] == SYNTHETIC_LABEL
    assert payload["causal_claim"] == "none"
    assert payload["promotion_decision"] == "not_evaluated"
    assert [signal["name"] for signal in payload["signals"]] == [
        "triage_score_distribution",
        "outcome_rates",
    ]
    for signal in payload["signals"]:
        assert signal["semantics"] == "descriptive_only_no_causal_claim"
        assert set(signal) >= {
            "name",
            "drifted",
            "reasons",
            "baseline_count",
            "current_count",
            "deltas",
            "enough_data",
        }


@pytest.mark.parametrize(
    "bad_payload",
    [
        {"email": "ada.fixture@example.com"},
        {"phone": "+1 555 010 0001"},
        {"card": "4111111111111111"},
        {"audio": "tests/fixtures/audio/call-0001/turn-000.wav"},
        {"bytes": b"audio bytes"},
    ],
)
def test_report_safety_screen_rejects_sensitive_values(bad_payload: object) -> None:
    with pytest.raises(ValidationError):
        assert_no_sensitive_text(bad_payload)


def test_report_safety_screen_allows_labels_and_timestamps() -> None:
    payload = {"label": SYNTHETIC_LABEL, "checked_at": "2026-11-25T12:00:00+00:00"}
    assert_no_sensitive_text(payload)
    assert json.loads(json.dumps(payload)) == payload
