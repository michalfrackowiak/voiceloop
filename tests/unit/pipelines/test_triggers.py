"""Tests for the P2-10 trigger policy."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from infra.pipelines.settings import PipelineSettings
from infra.pipelines.triggers import (
    TriggerReason,
    count_new_verified,
    decide_trigger,
    should_trigger,
)
from voiceloop_common import ValidationError

NOW = datetime(2026, 6, 1, 9, 0, tzinfo=UTC)


def test_new_data_threshold_is_inclusive(pipeline_settings: PipelineSettings) -> None:
    assert pipeline_settings.min_new_verified == 2
    assert not should_trigger(
        new_verified_count=1,
        settings=pipeline_settings,
        checked_at=NOW,
    )
    decision = decide_trigger(
        new_verified_count=2,
        settings=pipeline_settings,
        checked_at=NOW,
    )
    assert decision.triggered
    assert decision.reasons == (TriggerReason.NEW_DATA,)
    assert decision.reason == TriggerReason.NEW_DATA


def test_drift_and_manual_are_independent_triggers(pipeline_settings: PipelineSettings) -> None:
    drift = decide_trigger(new_verified_count=0, settings=pipeline_settings, drift_alarm=True)
    manual = decide_trigger(new_verified_count=0, settings=pipeline_settings, manual=True)
    assert drift.triggered and drift.reason == TriggerReason.DRIFT
    assert manual.triggered and manual.reason == TriggerReason.MANUAL


def test_manual_takes_priority_and_all_reasons_are_kept(
    pipeline_settings: PipelineSettings,
) -> None:
    decision = decide_trigger(
        new_verified_count=50,
        settings=pipeline_settings,
        drift_alarm=True,
        manual=True,
        checked_at=NOW,
    )
    assert decision.reasons == (
        TriggerReason.MANUAL,
        TriggerReason.DRIFT,
        TriggerReason.NEW_DATA,
    )
    assert decision.reason == TriggerReason.MANUAL


def test_time_passing_alone_never_triggers(pipeline_settings: PipelineSettings) -> None:
    long_ago = NOW - timedelta(days=365)
    decision = decide_trigger(
        new_verified_count=0,
        settings=pipeline_settings,
        last_run_at=long_ago,
        checked_at=NOW,
    )
    assert not decision.triggered
    assert decision.reasons == ()
    assert not should_trigger(
        new_verified_count=0,
        settings=pipeline_settings,
        last_run_at=long_ago,
        checked_at=NOW + timedelta(days=400),
    )


def test_settings_flags_can_enable_triggers(pipeline_settings: PipelineSettings) -> None:
    enabled = pipeline_settings.model_copy(update={"drift_alarm": True, "manual": True})
    decision = decide_trigger(new_verified_count=0, settings=enabled)
    assert decision.drift_alarm and decision.manual
    assert decision.triggered


def test_count_new_verified_uses_set_difference() -> None:
    assert count_new_verified(["a", "b", "c"], {"b"}) == 2
    assert count_new_verified([], {"b"}) == 0
    assert count_new_verified(["a", "a"], []) == 1


@pytest.mark.parametrize("bad_count", [-1, 1.5, "3", True])
def test_invalid_new_verified_count_is_rejected(bad_count: object) -> None:
    with pytest.raises(ValidationError):
        decide_trigger(new_verified_count=bad_count)  # type: ignore[arg-type]


def test_decision_serialises_with_utc_timestamps(pipeline_settings: PipelineSettings) -> None:
    decision = decide_trigger(
        new_verified_count=3,
        settings=pipeline_settings,
        last_run_at=NOW - timedelta(hours=1),
        checked_at=NOW,
    )
    payload = decision.to_dict()
    assert payload["triggered"] is True
    assert payload["reason"] == "new_verified_data"
    assert payload["reasons"] == ["new_verified_data"]
    assert payload["threshold"] == 2
    assert payload["checked_at"] == "2026-06-01T09:00:00+00:00"
    assert payload["last_run_at"] == "2026-06-01T08:00:00+00:00"
