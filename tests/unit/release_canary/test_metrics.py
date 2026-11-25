"""Tests for paired, outcome-based metrics and fail-closed regression detection."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from services.release.canary import (
    Arm,
    CanaryPair,
    CanarySettings,
    PredictionRecord,
    detect_regression,
    score_canary,
)
from services.release.shadow.scoring import OutcomeRecord

NOW = datetime(2026, 1, 1, tzinfo=UTC)
IGNORED = frozenset({"call_id", "status", "updated_at"})


def prediction(call_id: str, arm: Arm, fields: dict[str, Any]) -> PredictionRecord:
    """Project booking fields into a PII-safe prediction record."""
    return PredictionRecord.from_fields(
        call_id, arm, fields, IGNORED, observed_at=NOW, synthetic=True
    )


def outcome(
    call_id: str,
    fields: dict[str, Any],
    status: str = "confirmed",
) -> OutcomeRecord:
    """Project a final booking record into a PII-safe outcome."""
    return OutcomeRecord.from_payload(
        call_id,
        {"status": status, "final_record": fields, "updated_at": ""},
        IGNORED,
        True,
    )


def pair(
    call_id: str,
    champion_fields: dict[str, Any],
    canary_fields: dict[str, Any],
    *,
    arm: Arm = Arm.CANARY,
    outcome_record: OutcomeRecord | None = None,
    baseline: PredictionRecord | None = None,
    canary_prediction: PredictionRecord | None = None,
) -> CanaryPair:
    """Build one paired observation with injectable missing sides."""
    return CanaryPair(
        call_id=call_id,
        arm=arm,
        canary=(
            canary_prediction
            if canary_prediction is not None
            else prediction(call_id, Arm.CANARY, canary_fields)
        ),
        champion=(
            baseline if baseline is not None else prediction(call_id, Arm.CHAMPION, champion_fields)
        ),
        outcome=outcome_record if outcome_record is not None else outcome(call_id, champion_fields),
    )


def test_paired_regression_is_detected(settings: CanarySettings) -> None:
    """A strictly negative paired outcome delta requires rollback."""
    good = {"guests": 2, "time": "19:00"}
    bad = {"guests": -1, "time": "19:00"}
    pairs = [pair(f"call-{index:04d}", good, bad) for index in range(1, 4)]
    metrics = score_canary(pairs, settings)
    verdict = detect_regression(metrics, settings)

    assert metrics.sample_counts["calls_scored"] == 3
    assert metrics.canary_call_accuracy == 0.0
    assert metrics.champion_call_accuracy == 1.0
    assert metrics.paired_delta_call_accuracy == -1.0
    assert metrics.paired_call_losses == 3
    assert verdict.rollback is True
    assert verdict.reason == "outcome_metric_regression"
    assert verdict.evidence_status == "complete"


def test_no_regression_when_canary_matches(settings: CanarySettings) -> None:
    """Equal predictions yield zero paired delta and no rollback."""
    fields = {"guests": 4, "time": "12:30"}
    pairs = [pair(f"call-{index:04d}", fields, fields) for index in range(1, 4)]
    metrics = score_canary(pairs, settings)
    verdict = detect_regression(metrics, settings)

    assert metrics.paired_delta_call_accuracy == 0.0
    assert metrics.paired_delta_field_accuracy == 0.0
    assert metrics.paired_call_ties == 3
    assert verdict.rollback is False
    assert verdict.reason == "no_regression_observed"


def test_canary_better_is_not_a_regression(settings: CanarySettings) -> None:
    """A positive paired delta is descriptive only; it never promotes."""
    champion_fields = {"guests": 0, "time": "10:00"}
    canary_fields = {"guests": 2, "time": "10:00"}
    pairs = [
        pair(
            f"call-{index:04d}",
            champion_fields,
            canary_fields,
            outcome_record=outcome(f"call-{index:04d}", canary_fields),
        )
        for index in range(1, 4)
    ]
    verdict = detect_regression(score_canary(pairs, settings), settings)
    assert verdict.rollback is False


def test_missing_outcome_fails_closed(settings: CanarySettings) -> None:
    """A missing final record requires rollback, not a pass."""
    fields = {"guests": 2}
    missing = CanaryPair(
        call_id="call-0001",
        arm=Arm.CANARY,
        canary=prediction("call-0001", Arm.CANARY, fields),
        champion=prediction("call-0001", Arm.CHAMPION, fields),
        outcome=None,
    )
    metrics = score_canary([missing], settings)
    verdict = detect_regression(metrics, settings)

    assert metrics.sample_counts["calls_missing_outcome"] == 1
    assert verdict.rollback is True
    assert verdict.reason == "incomplete_outcome_evidence"
    assert verdict.evidence_status == "incomplete"


def test_missing_baseline_and_canary_fail_closed(settings: CanarySettings) -> None:
    """Missing same-call baseline or canary prediction is incomplete evidence."""
    fields = {"guests": 2}
    missing_baseline = CanaryPair(
        call_id="call-0001",
        arm=Arm.CANARY,
        canary=prediction("call-0001", Arm.CANARY, fields),
        champion=None,
        outcome=outcome("call-0001", fields),
    )
    missing_canary = CanaryPair(
        call_id="call-0002",
        arm=Arm.CANARY,
        canary=None,
        champion=prediction("call-0002", Arm.CHAMPION, fields),
        outcome=outcome("call-0002", fields),
    )
    metrics = score_canary([missing_baseline, missing_canary], settings)
    verdict = detect_regression(metrics, settings)

    assert metrics.sample_counts["calls_missing_baseline_prediction"] == 1
    assert metrics.sample_counts["calls_missing_canary_prediction"] == 1
    assert verdict.evidence_status == "incomplete"
    assert verdict.rollback is True


def test_empty_final_record_and_identity_mismatch_fail_closed(
    settings: CanarySettings,
) -> None:
    """Empty final records and call-ID mismatches are incomplete evidence."""
    fields = {"guests": 2}
    empty = CanaryPair(
        call_id="call-0001",
        arm=Arm.CANARY,
        canary=prediction("call-0001", Arm.CANARY, fields),
        champion=prediction("call-0001", Arm.CHAMPION, fields),
        outcome=outcome("call-0001", {}),
    )
    mismatch = CanaryPair(
        call_id="call-0002",
        arm=Arm.CANARY,
        canary=prediction("call-0002", Arm.CANARY, fields),
        champion=prediction("call-0002", Arm.CHAMPION, fields),
        outcome=outcome("call-9999", fields),
    )
    metrics = score_canary([empty, mismatch], settings)
    verdict = detect_regression(metrics, settings)

    assert metrics.sample_counts["calls_empty_final_record"] == 1
    assert metrics.sample_counts["calls_identity_mismatch"] == 1
    assert verdict.reason == "incomplete_outcome_evidence"
    assert verdict.rollback is True


def test_nonscorable_status_is_insufficient(settings: CanarySettings) -> None:
    """Non-scorable final statuses are excluded, not treated as complete evidence."""
    fields = {"guests": 2}
    pairs = [
        CanaryPair(
            call_id=f"call-{index:04d}",
            arm=Arm.CANARY,
            canary=prediction(f"call-{index:04d}", Arm.CANARY, fields),
            champion=prediction(f"call-{index:04d}", Arm.CHAMPION, fields),
            outcome=outcome(f"call-{index:04d}", fields, status="cancelled"),
        )
        for index in range(1, 4)
    ]
    metrics = score_canary(pairs, settings)
    verdict = detect_regression(metrics, settings)

    assert metrics.sample_counts["calls_excluded_by_status"] == 3
    assert metrics.sample_counts["calls_scored"] == 0
    assert verdict.evidence_status == "insufficient"
    assert verdict.rollback is False


def test_insufficient_pairs_do_not_pass(settings: CanarySettings) -> None:
    """Too few paired calls are insufficient; the runner handles the window bound."""
    fields = {"guests": 2}
    verdict = detect_regression(
        score_canary([pair("call-0001", fields, fields)], settings), settings
    )
    assert verdict.evidence_status == "insufficient"
    assert verdict.rollback is False


def test_control_traffic_is_counted_but_not_judged(settings: CanarySettings) -> None:
    """Control calls contribute to observed counts but not canary scoring."""
    fields = {"guests": 2}
    control = pair("call-0001", fields, {"guests": -1}, arm=Arm.CHAMPION)
    metrics = score_canary([control], settings)
    assert metrics.sample_counts["calls_control"] == 1
    assert metrics.sample_counts["calls_scored"] == 0


def test_reports_are_labelled_and_wer_free(settings: CanarySettings) -> None:
    """All report fragments carry the synthetic label and never mention WER."""
    fields = {"guests": 2}
    metrics = score_canary(
        [pair(f"call-{index:04d}", fields, fields) for index in range(1, 4)], settings
    )
    verdict = detect_regression(metrics, settings)
    for fragment in (metrics.to_dict(), verdict.to_dict()):
        assert fragment["evidence_label"] == settings.evidence_label
        assert "SYNTHETIC" in fragment["evidence_label"]
        assert "wer" not in str(fragment).lower()
    assert verdict.to_dict()["promotion_decision"] == "not_evaluated"
