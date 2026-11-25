"""P3-09.2: drift comparison is explicit, synthetic and descriptive only."""

from __future__ import annotations

from pathlib import Path

import pytest

from services.datasets.golden.maintenance import (
    ARTIFACT_LABEL,
    FIXTURE_SHA256,
    HUMAN_REVIEW_GATE,
    SEALED_CONTENT_HASH,
    GoldenMaintenanceSettings,
    build_drift_report,
    canonical_json,
    contains_pii_pattern,
    validate_drift_report,
    write_drift_report,
)
from voiceloop_common import ValidationError


def test_report_has_explicit_sample_counts_and_descriptive_interpretation(
    settings: GoldenMaintenanceSettings,
) -> None:
    """Counts and shares are shown without a promotion or causal decision."""
    report = build_drift_report(settings)
    assert report["label"] == ARTIFACT_LABEL
    assert report["sealed_content_hash"] == SEALED_CONTENT_HASH
    assert report["fixture_sha256"] == FIXTURE_SHA256
    assert report["sample_counts"] == {
        "sealed_golden_turns": 36,
        "sealed_golden_calls": 6,
        "synthetic_incoming_turns": 12,
        "synthetic_incoming_calls": 2,
    }
    assert report["interpretation"]["mode"] == "descriptive_only"
    assert report["interpretation"]["causal_claim"] == "none"
    assert report["interpretation"]["alarm"] == "not_evaluated"
    assert report["promotion"] == "not_evaluated"
    assert report["threshold_change"] == "none"
    assert set(report["golden"]["features"]) == {
        "audio_kind",
        "sample_rate_hz",
        "slot_present",
    }
    assert sum(report["golden"]["features"]["audio_kind"].values()) == 36
    assert sum(report["incoming"]["features"]["audio_kind"].values()) == 12
    assert validate_drift_report(report) == ()
    assert not contains_pii_pattern(canonical_json(report))


def test_report_rejects_promotion_or_causal_claims(
    settings: GoldenMaintenanceSettings,
) -> None:
    """Interpretation fields cannot be upgraded into an action or cause."""
    promoted = build_drift_report(settings)
    promoted["promotion"] = "promote"
    assert any("promotion" in error for error in validate_drift_report(promoted))

    causal = build_drift_report(settings)
    causal["interpretation"]["causal_claim"] = "caused by fixture noise"
    assert any("interpretation" in error for error in validate_drift_report(causal))


def test_human_review_gate_is_present_and_validated(
    settings: GoldenMaintenanceSettings,
) -> None:
    """No drift report is valid without an explicit human-review gate."""
    report = build_drift_report(settings)
    assert report["human_review_gate"] == HUMAN_REVIEW_GATE
    assert report["interpretation"]["human_review_gate"] == HUMAN_REVIEW_GATE
    assert validate_drift_report(report) == ()

    missing_top_level = dict(report)
    missing_top_level.pop("human_review_gate")
    assert any("human-review gate" in error for error in validate_drift_report(missing_top_level))

    missing_interpretation = build_drift_report(settings)
    missing_interpretation["interpretation"].pop("human_review_gate")
    assert any(
        "descriptive-only text" in error for error in validate_drift_report(missing_interpretation)
    )


def test_report_write_stays_outside_sealed_artifacts(
    settings: GoldenMaintenanceSettings, tmp_path: Path
) -> None:
    """A report can be written to maintenance storage, never to sealed files."""
    report = build_drift_report(settings)
    with pytest.raises(ValidationError, match="sealed golden artifacts"):
        write_drift_report(report, "services/datasets/golden/artifacts/drift.json", settings)
    destination = write_drift_report(report, tmp_path / "drift.json", settings)
    assert destination.exists()
    payload = destination.read_text(encoding="utf-8")
    assert ARTIFACT_LABEL in payload
    assert not contains_pii_pattern(payload)
