"""P3-08: typed shadow settings use only the VOICELOOP_GATE_SHADOW_ prefix."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from services.evalgate.gate import ShadowEvidenceSettings


def test_defaults() -> None:
    settings = ShadowEvidenceSettings()

    assert settings.max_report_age_hours == pytest.approx(24.0)
    assert settings.min_turns_observed == 1
    assert settings.min_successful_results == 1
    assert settings.refusal_log_path == "reports/evalgate/shadow_refusals.jsonl"


def test_env_prefix(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VOICELOOP_GATE_SHADOW_MAX_REPORT_AGE_HOURS", "6.5")
    monkeypatch.setenv("VOICELOOP_GATE_SHADOW_MIN_TURNS_OBSERVED", "4")
    monkeypatch.setenv("VOICELOOP_GATE_SHADOW_MIN_SUCCESSFUL_RESULTS", "2")
    monkeypatch.setenv("VOICELOOP_GATE_SHADOW_REFUSAL_LOG_PATH", "tmp/shadow-refusals.jsonl")

    settings = ShadowEvidenceSettings()

    assert settings.max_report_age_hours == pytest.approx(6.5)
    assert settings.min_turns_observed == 4
    assert settings.min_successful_results == 2
    assert settings.refusal_log_path == "tmp/shadow-refusals.jsonl"


def test_unrelated_prefix_is_ignored(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VOICELOOP_GATE_MAX_REPORT_AGE_HOURS", "1.0")
    monkeypatch.setenv("VOICELOOP_RELEASE_SHADOW_MAX_PENDING", "7")

    settings = ShadowEvidenceSettings()

    assert settings.max_report_age_hours == pytest.approx(24.0)


def test_invalid_ranges_are_rejected() -> None:
    with pytest.raises(ValidationError):
        ShadowEvidenceSettings(max_report_age_hours=0)
    with pytest.raises(ValidationError):
        ShadowEvidenceSettings(min_turns_observed=0)
    with pytest.raises(ValidationError):
        ShadowEvidenceSettings(min_successful_results=-1)
