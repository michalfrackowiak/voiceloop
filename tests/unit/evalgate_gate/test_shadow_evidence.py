"""P3-08.1: fail-closed validation of P3-07 shadow comparison reports."""

from __future__ import annotations

from collections.abc import Callable
from datetime import timedelta
from typing import Any

import pytest

from services.evalgate.gate import (
    GateRefusal,
    ShadowEvidenceError,
    ShadowEvidenceRequest,
    ShadowEvidenceSettings,
    verify_shadow_evidence,
)
from services.release.shadow import validate_shadow_report
from tests.unit.evalgate_gate.shadow_fakes import (
    CANDIDATE_URI,
    CHAMPION_URI,
    NOW,
    make_shadow_report,
)

TIMESTAMP = NOW.isoformat(timespec="seconds").replace("+00:00", "Z")


def _request(*, evaluated_at: str = TIMESTAMP) -> ShadowEvidenceRequest:
    """Return a request matching the fake STT artifact pair."""
    return ShadowEvidenceRequest(
        candidate_version="v4",
        champion_version="v3",
        candidate_artifact_uri=CANDIDATE_URI,
        champion_artifact_uri=CHAMPION_URI,
        evaluated_at=evaluated_at,
    )


def _expect_refusal(
    report: dict[str, Any] | None,
    expected_code: str,
    *,
    request: ShadowEvidenceRequest | None = None,
    settings: ShadowEvidenceSettings | None = None,
) -> GateRefusal:
    """Assert that verify_shadow_evidence fails closed with a stable code."""
    with pytest.raises(ShadowEvidenceError) as exc:
        verify_shadow_evidence(
            report, request or _request(), settings or ShadowEvidenceSettings(), now=NOW
        )
    assert exc.value.code == expected_code
    assert exc.value.reasons
    assert all("Fixture" not in reason and "@" not in reason for reason in exc.value.reasons)
    return GateRefusal(code=exc.value.code, reasons=exc.value.reasons)


def test_valid_report_is_accepted_and_matches_p3_07_shape() -> None:
    report = make_shadow_report()

    validate_shadow_report(report)
    summary = verify_shadow_evidence(report, _request(), ShadowEvidenceSettings(), now=NOW)

    assert summary.turns_observed == 36
    assert summary.calls_scored == 6
    assert summary.successful_results == 6
    assert summary.generated_at == TIMESTAMP
    assert summary.synthetic is True


def test_missing_report_refuses() -> None:
    refusal = _expect_refusal(None, "shadow_evidence_missing")

    assert refusal.code == "shadow_evidence_missing"


@pytest.mark.parametrize(
    ("mutate", "code"),
    [
        (lambda r: r.update({"promotion_decision": "promote"}), "shadow_evidence_promotion_claim"),
        (lambda r: r["promotion"].update({"alias_writes": 1}), "shadow_evidence_promotion_claim"),
        (
            lambda r: r["promotion"].update({"thresholds_changed": True}),
            "shadow_evidence_promotion_claim",
        ),
        (lambda r: r.update({"status": "LIVE TRAFFIC"}), "shadow_evidence_not_synthetic"),
        (
            lambda r: r["provenance"].update({"synthetic": False}),
            "shadow_evidence_not_synthetic",
        ),
        (
            lambda r: r["provenance"].update({"fixture_standin": False}),
            "shadow_evidence_not_synthetic",
        ),
        (
            lambda r: r["provenance"].update({"real_traffic": True}),
            "shadow_evidence_not_synthetic",
        ),
        (
            lambda r: r["provenance"].update({"audio_bytes_logged": True}),
            "shadow_evidence_not_synthetic",
        ),
        (
            lambda r: r["shadow_evidence"].update({"synthetic_standin": False}),
            "shadow_evidence_not_synthetic",
        ),
        (lambda r: r["sample_counts"].pop("calls_scored"), "shadow_evidence_incomplete"),
        (
            lambda r: r["sample_counts"].update({"turns_observed": 0}),
            "shadow_evidence_incomplete",
        ),
        (
            lambda r: r["provenance"].update({"successful_results": 0}),
            "shadow_evidence_incomplete",
        ),
        (
            lambda r: r["metrics"]["candidate"].pop("outcome_call_accuracy"),
            "shadow_evidence_incomplete",
        ),
        (lambda r: r["deltas"].pop("outcome_error_rate"), "shadow_evidence_incomplete"),
        (lambda r: r.pop("generated_at"), "shadow_evidence_incomplete"),
        (
            lambda r: r["provenance"].update({"join_key": "event_id"}),
            "shadow_evidence_incomplete",
        ),
        (
            lambda r: r.update({"artifact_type": "other.shadow.comparison"}),
            "shadow_evidence_invalid",
        ),
        (
            lambda r: r.update({"comparison_schema_version": "internal-2.0.0"}),
            "shadow_evidence_invalid",
        ),
        (
            lambda r: r["isolation"].update({"within_budget": False}),
            "shadow_evidence_invalid",
        ),
        (
            lambda r: r["isolation"].update({"models_invoked_by_call_path": True}),
            "shadow_evidence_invalid",
        ),
        (
            lambda r: r["metric_semantics"].update({"primary_metric": "wer"}),
            "shadow_evidence_invalid",
        ),
        (
            lambda r: r["provenance"]["models"][0].update(
                {"artifact_uri": "converted://another-model"}
            ),
            "shadow_evidence_mismatch",
        ),
        (
            lambda r: r["provenance"]["models"][1].update({"synthetic": False}),
            "shadow_evidence_not_synthetic",
        ),
        (
            lambda r: r["shadow_evidence"].update({"turns_observed": 35}),
            "shadow_evidence_mismatch",
        ),
        (
            lambda r: r["shadow_evidence"].update({"outcome_error_rate": 0.9}),
            "shadow_evidence_mismatch",
        ),
        (lambda r: r.update({"report_uri": "fake://other.json"}), "shadow_evidence_mismatch"),
    ],
)
def test_bad_reports_fail_closed(mutate: Callable[[dict[str, Any]], None], code: str) -> None:
    """Every malformed or overclaiming report must yield its stable refusal."""
    report = make_shadow_report()
    mutate(report)

    refusal = _expect_refusal(report, code)

    assert refusal.task_id == "P3-08"
    assert refusal.to_dict()["code"] == code


def test_report_older_than_max_age_is_stale() -> None:
    report = make_shadow_report(
        generated_at=(NOW - timedelta(hours=25))
        .isoformat(timespec="seconds")
        .replace("+00:00", "Z")
    )

    _expect_refusal(report, "shadow_evidence_stale")


def test_report_predating_evaluated_artifact_is_stale() -> None:
    report = make_shadow_report(
        generated_at=(NOW - timedelta(minutes=1))
        .isoformat(timespec="seconds")
        .replace("+00:00", "Z")
    )

    _expect_refusal(report, "shadow_evidence_stale")


def test_configured_minimums_are_enforced() -> None:
    report = make_shadow_report(turns_observed=3, calls_scored=3, successful_results=2)
    settings = ShadowEvidenceSettings(min_turns_observed=4, min_successful_results=3)

    _expect_refusal(report, "shadow_evidence_incomplete", settings=settings)


def test_invalid_timestamp_is_refused_without_echoing_value() -> None:
    report = make_shadow_report(generated_at="not-a-timestamp")

    refusal = _expect_refusal(report, "shadow_evidence_incomplete")

    assert "not-a-timestamp" not in " ".join(refusal.reasons)
