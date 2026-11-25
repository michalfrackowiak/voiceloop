"""P3-09.0: typed settings and PII screening fail closed."""

from __future__ import annotations

from pathlib import Path

import pytest

from services.datasets.golden.maintenance import (
    ARTIFACT_LABEL,
    FIXTURE_SHA256,
    SEALED_CONTENT_HASH,
    GoldenMaintenanceSettings,
    assert_allowed_actor,
    assert_safe_text,
    contains_pii_pattern,
)
from voiceloop_common import ValidationError


def test_defaults_use_typed_prefix_and_committed_anchors() -> None:
    """Defaults pin the synthetic anchors and use the maintenance-owned output."""
    settings = GoldenMaintenanceSettings()
    assert settings.model_config.get("env_prefix") == "VOICELOOP_GOLDEN_MAINT_"
    assert settings.expected_sealed_content_hash == SEALED_CONTENT_HASH
    assert settings.expected_fixture_sha256 == FIXTURE_SHA256
    assert settings.artifact_label == ARTIFACT_LABEL
    assert settings.sealed_version == "golden.v1"
    assert settings.output_dir == "services/datasets/golden/maintenance/artifacts"
    assert settings.first_reviewer_id != settings.second_reviewer_id


def test_environment_overrides_are_scoped(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Only VOICELOOP_GOLDEN_MAINT_ variables configure this package."""
    monkeypatch.setenv("VOICELOOP_GOLDEN_MAINT_OUTPUT_DIR", str(tmp_path / "out"))
    monkeypatch.setenv("VOICELOOP_GOLDEN_MAINT_CANDIDATE_CALLS", "3")
    monkeypatch.setenv("VOICELOOP_GOLDEN_MAINT_FIRST_REVIEWER_ID", "reviewer-x")
    monkeypatch.setenv("VOICELOOP_GOLDEN_MAINT_SECOND_REVIEWER_ID", "reviewer-y")
    monkeypatch.setenv("VOICELOOP_GOLDEN_MAINT_TASK_ID", "P3-09-env")
    monkeypatch.setenv("VOICELOOP_GOLDEN_CANDIDATE_CALLS", "99")

    settings = GoldenMaintenanceSettings()
    assert settings.output_dir == str(tmp_path / "out")
    assert settings.candidate_calls == 3
    assert (settings.first_reviewer_id, settings.second_reviewer_id) == (
        "reviewer-x",
        "reviewer-y",
    )
    assert settings.task_id == "P3-09-env"


def test_single_reviewer_and_sealed_output_fail_closed() -> None:
    """A shared reviewer ID or sealed destination cannot be configured."""
    with pytest.raises(ValidationError, match="distinct"):
        GoldenMaintenanceSettings(
            first_reviewer_id="reviewer-same", second_reviewer_id="reviewer-same"
        )
    with pytest.raises(ValidationError, match="sealed golden artifacts"):
        GoldenMaintenanceSettings(output_dir="services/datasets/golden/artifacts/nested")


def test_wrong_label_or_unsafe_reason_fails_closed() -> None:
    """The mandatory stand-in label and PII screening cannot be bypassed."""
    with pytest.raises(ValidationError, match="artifact_label"):
        GoldenMaintenanceSettings(artifact_label="human-reviewed golden set")
    with pytest.raises(ValidationError):
        GoldenMaintenanceSettings(review_reason="contact alice@example.com")


def test_unlisted_and_reviewer_actors_fail_closed(
    settings: GoldenMaintenanceSettings,
) -> None:
    """Only allowlisted pseudonyms that are not reviewers may propose changes."""
    with pytest.raises(ValidationError, match="allowlist"):
        assert_allowed_actor("unknown-actor-999", settings)
    for reviewer_id in (settings.first_reviewer_id, settings.second_reviewer_id):
        with pytest.raises(ValidationError, match="reviewer"):
            assert_allowed_actor(reviewer_id, settings)
    with pytest.raises(ValidationError, match="reviewer"):
        GoldenMaintenanceSettings(allowed_actor_ids=(settings.first_reviewer_id,))


def test_pii_screening_detects_email_phone_and_card_patterns() -> None:
    """Free text carrying contact or card-like patterns is rejected."""
    assert contains_pii_pattern("alice@example.com")
    assert contains_pii_pattern("555-123-4567")
    assert contains_pii_pattern("4111111111111111")
    assert not contains_pii_pattern("fa6ec0e180daefa41257478329800e897")
    for value in ("alice@example.com", "555-123-4567", "4111111111111111"):
        with pytest.raises(ValidationError):
            assert_safe_text(value, "purpose")
