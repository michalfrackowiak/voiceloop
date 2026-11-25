"""Tests for bounded, typed canary settings."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from services.release.canary import (
    MAX_CANARY_FRACTION,
    MAX_WINDOW_SECONDS,
    SYNTHETIC_LABEL,
    CanarySettings,
)


def test_fraction_is_bounded() -> None:
    """Fraction above the cap and below zero must be rejected."""
    assert MAX_CANARY_FRACTION == 0.20
    with pytest.raises(ValidationError):
        CanarySettings(canary_fraction=0.21)
    with pytest.raises(ValidationError):
        CanarySettings(canary_fraction=-0.01)
    assert CanarySettings(canary_fraction=0.0).canary_fraction == 0.0
    assert CanarySettings(canary_fraction=MAX_CANARY_FRACTION).canary_fraction == 0.20


def test_window_is_bounded() -> None:
    """Window must be positive and within the explicit upper bound."""
    assert MAX_WINDOW_SECONDS == 3600.0
    with pytest.raises(ValidationError):
        CanarySettings(window_seconds=0.0)
    with pytest.raises(ValidationError):
        CanarySettings(window_seconds=3601.0)
    assert CanarySettings(window_seconds=3600.0).window().total_seconds() == 3600.0


def test_batch_bounds_are_enforced() -> None:
    """Calls-per-tick and max-pairs cannot exceed the explicit caps."""
    with pytest.raises(ValidationError):
        CanarySettings(max_calls_per_tick=129)
    with pytest.raises(ValidationError):
        CanarySettings(max_pairs=10_001)
    with pytest.raises(ValidationError):
        CanarySettings(min_paired_calls=0)


def test_status_and_ignored_field_parsing() -> None:
    """CSV settings parse into trimmed, non-empty frozensets."""
    settings = CanarySettings(
        scorable_statuses_csv="confirmed, edited_by_staff ,,",
        ignored_outcome_fields_csv="call_id, status ,",
    )
    assert settings.scorable_statuses() == frozenset({"confirmed", "edited_by_staff"})
    assert settings.ignored_outcome_fields() == frozenset({"call_id", "status"})


def test_blank_text_is_rejected() -> None:
    """Blank identifiers, paths and evidence labels are not permitted."""
    with pytest.raises(ValidationError):
        CanarySettings(evidence_label="   ")
    with pytest.raises(ValidationError):
        CanarySettings(task_id="")
    with pytest.raises(ValidationError):
        CanarySettings(rollback_actor_id="  ")


def test_env_prefix_and_synthetic_label(monkeypatch: pytest.MonkeyPatch) -> None:
    """Environment overrides use VOICELOOP_CANARY_ and keep the synthetic label."""
    monkeypatch.setenv("VOICELOOP_CANARY_CANARY_FRACTION", "0.1")
    monkeypatch.setenv("VOICELOOP_CANARY_WINDOW_SECONDS", "120")
    settings = CanarySettings()
    assert settings.canary_fraction == 0.1
    assert settings.window_seconds == 120.0
    assert settings.evidence_label == SYNTHETIC_LABEL
    assert SYNTHETIC_LABEL == "SYNTHETIC/FIXTURE STAND-IN - NOT LIVE CANARY EVIDENCE"
