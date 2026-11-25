"""P3-07: typed settings and env overrides."""

from __future__ import annotations

from datetime import timedelta

import pytest
from pydantic import ValidationError

from services.release.shadow import ShadowSettings


def test_defaults_are_bounded_and_evidence_only() -> None:
    settings = ShadowSettings()

    assert settings.task_id == "P3-07"
    assert settings.max_pending >= 1
    assert settings.mirror_budget_ms > 0
    assert settings.max_turns_per_drain >= 1
    assert settings.output_path == "reports/release/shadow/comparison.json"
    assert settings.scorable_statuses() == frozenset({"confirmed", "edited_by_staff"})
    assert settings.ignored_outcome_fields() == frozenset({"call_id", "status", "updated_at"})
    assert settings.mirror_budget() == timedelta(milliseconds=settings.mirror_budget_ms)


def test_env_prefix_is_scoped(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VOICELOOP_RELEASE_SHADOW_MAX_PENDING", "3")
    monkeypatch.setenv("VOICELOOP_RELEASE_SHADOW_MIRROR_BUDGET_MS", "1.5")
    monkeypatch.setenv("VOICELOOP_RELEASE_SHADOW_SCORABLE_STATUSES_CSV", "confirmed")
    monkeypatch.setenv("VOICELOOP_RELEASE_SHADOW_OUTPUT_PATH", "C:/tmp/shadow.json")

    settings = ShadowSettings()

    assert settings.max_pending == 3
    assert settings.mirror_budget_ms == 1.5
    assert settings.scorable_statuses() == frozenset({"confirmed"})
    assert settings.output_path == "C:/tmp/shadow.json"
    assert settings.mirror_budget() == timedelta(milliseconds=1.5)


def test_invalid_capacity_and_budget_are_rejected() -> None:
    with pytest.raises(ValidationError):
        ShadowSettings(max_pending=0)
    with pytest.raises(ValidationError):
        ShadowSettings(mirror_budget_ms=0)
    with pytest.raises(ValidationError):
        ShadowSettings(output_path="   ")
