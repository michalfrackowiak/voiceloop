"""Tests for VOICELOOP_DRIFT_ typed settings."""

from __future__ import annotations

import pytest

from infra.pipelines.drift import (
    EXPECTED_FIXTURE_SHA256,
    EXPECTED_GOLDEN_CONTENT_HASH,
    SYNTHETIC_LABEL,
    DriftSettings,
    ManualOverride,
    manual_override_from_settings,
)
from voiceloop_common import ValidationError


def test_env_prefix_is_drift_only(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VOICELOOP_DRIFT_MIN_NEW_VERIFIED", "7")
    monkeypatch.setenv("VOICELOOP_PIPELINES_MIN_NEW_VERIFIED", "999")
    settings = DriftSettings()
    assert settings.min_new_verified == 7
    assert settings.model_config.get("env_prefix") == "VOICELOOP_DRIFT_"


def test_defaults_are_labelled_and_offline() -> None:
    settings = DriftSettings()
    assert "SYNTHETIC/FIXTURE STAND-IN" in SYNTHETIC_LABEL
    assert settings.fixture_sha256 == EXPECTED_FIXTURE_SHA256
    assert settings.golden_content_hash == EXPECTED_GOLDEN_CONTENT_HASH
    assert settings.manual_override is False
    assert settings.min_window_samples >= 1
    assert settings.min_new_verified == 200


def test_degraded_outcomes_are_validated() -> None:
    with pytest.raises(ValueError):
        DriftSettings(degraded_outcomes=("confirmed", "not_a_status"))
    with pytest.raises(ValueError):
        DriftSettings(degraded_outcomes=())
    settings = DriftSettings(degraded_outcomes=("unknown",))
    assert settings.degraded_outcomes == ("unknown",)


def test_manual_override_requires_explicit_flag_actor_and_reason() -> None:
    with pytest.raises(ValidationError):
        manual_override_from_settings(DriftSettings(manual_actor_id="operator-1a2b3c4d"))
    assert manual_override_from_settings(DriftSettings()) is None

    override = manual_override_from_settings(
        DriftSettings(
            manual_override=True,
            manual_actor_id="operator-1a2b3c4d",
            manual_reason_code="incident_replay",
        )
    )
    assert isinstance(override, ManualOverride)
    assert override.actor_id == "operator-1a2b3c4d"
    assert override.reason_code == "incident_replay"

    with pytest.raises(ValidationError):
        manual_override_from_settings(
            DriftSettings(manual_override=True, manual_reason_code="lead_authorized")
        )


def test_disallowed_override_tokens_are_rejected() -> None:
    with pytest.raises(ValidationError):
        ManualOverride("ada.fixture@example.com", "lead_authorized")
    with pytest.raises(ValidationError):
        ManualOverride("operator-1a2b3c4d", "free text with a name")
