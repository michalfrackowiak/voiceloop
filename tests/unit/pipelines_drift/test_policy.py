"""Tests for the data-volume AND (drift OR manual) trigger policy."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from typing import cast

import pytest

from infra.pipelines.drift import (
    SYNTHETIC_LABEL,
    DriftSettings,
    InMemoryAuditSink,
    ManualOverride,
    TriggerDecision,
    TriggerReason,
    decide_trigger,
)
from voiceloop_common import Clock, ValidationError


@pytest.mark.parametrize(
    ("count", "drifted", "expected"),
    [(5, True, True), (6, True, True), (5, False, False), (4, True, False)],
)
def test_volume_and_drift_are_both_required(
    count: int,
    drifted: bool,
    expected: bool,
    drift_settings: DriftSettings,
    fixed_clock: Clock,
) -> None:
    decision = decide_trigger(
        new_verified_count=count,
        drifted=drifted,
        settings=drift_settings,
        clock=fixed_clock,
    )
    assert decision.triggered is expected
    assert decision.reasons == (
        (TriggerReason.DRIFT, TriggerReason.DATA_VOLUME) if expected else ()
    )


def test_manual_override_with_volume_triggers_and_is_audited(
    drift_settings: DriftSettings, fixed_clock: Clock
) -> None:
    sink = InMemoryAuditSink()
    override = ManualOverride("operator-1a2b3c4d", "incident_replay")
    decision = decide_trigger(
        new_verified_count=10,
        drifted=False,
        settings=drift_settings,
        manual_override=override,
        audit_sink=sink,
        clock=fixed_clock,
    )
    assert decision.triggered is True
    assert decision.reason == TriggerReason.MANUAL_OVERRIDE
    assert decision.reasons == (TriggerReason.MANUAL_OVERRIDE, TriggerReason.DATA_VOLUME)
    assert decision.audit_recorded is True
    assert len(sink.entries) == 1
    entry = sink.entries[0]
    assert entry["actor_id"] == "operator-1a2b3c4d"
    assert entry["reason_code"] == "incident_replay"
    assert entry["reason"] == "Operator requested an incident-replay retraining rehearsal."
    assert entry["checked_at"] == "2026-11-25T12:00:00+00:00"
    assert entry["label"] == SYNTHETIC_LABEL


def test_manual_override_without_volume_is_audited_but_does_not_trigger(
    drift_settings: DriftSettings, fixed_clock: Clock
) -> None:
    sink = InMemoryAuditSink()
    decision = decide_trigger(
        new_verified_count=1,
        drifted=True,
        settings=drift_settings,
        manual_override=ManualOverride("lead-a1b2c3d4", "lead_authorized"),
        audit_sink=sink,
        clock=fixed_clock,
    )
    assert decision.triggered is False
    assert decision.reasons == ()
    assert "insufficient_new_verified_data" in decision.blocked_reasons
    assert len(sink.entries) == 1
    assert sink.entries[0]["triggered"] is False


def test_calendar_tick_alone_never_triggers(
    drift_settings: DriftSettings, fixed_clock: Clock
) -> None:
    decision = decide_trigger(
        new_verified_count=0,
        drifted=False,
        settings=drift_settings,
        clock=fixed_clock,
        scheduled_tick=True,
        last_run_at=datetime(2020, 1, 1, tzinfo=UTC),
    )
    assert decision.triggered is False
    assert decision.reasons == ()
    assert decision.scheduled_tick is True
    payload = decision.to_dict()
    assert payload["calendar_tick_is_trigger"] is False
    assert "calendar" not in payload["reasons"]


def test_all_reasons_follow_manual_drift_data_priority(
    drift_settings: DriftSettings, fixed_clock: Clock
) -> None:
    decision = decide_trigger(
        new_verified_count=50,
        drifted=True,
        settings=drift_settings,
        manual_override=ManualOverride("admin-1a2b3c4d", "review_backlog"),
        clock=fixed_clock,
    )
    assert decision.reasons == (
        TriggerReason.MANUAL_OVERRIDE,
        TriggerReason.DRIFT,
        TriggerReason.DATA_VOLUME,
    )
    assert decision.reason == TriggerReason.MANUAL_OVERRIDE
    assert decision.blocked_reasons == ()


@pytest.mark.parametrize("bad_count", [-1, 1.5, "3", True])
def test_invalid_counts_are_rejected(bad_count: object, fixed_clock: Clock) -> None:
    with pytest.raises(ValidationError):
        decide_trigger(new_verified_count=cast(int, bad_count), drifted=True, clock=fixed_clock)


def test_invalid_drifted_value_is_rejected(fixed_clock: Clock) -> None:
    with pytest.raises(ValidationError):
        decide_trigger(new_verified_count=5, drifted=cast(bool, 1), clock=fixed_clock)


def test_serialised_decision_is_labelled_and_pii_free(
    drift_settings: DriftSettings, fixed_clock: Clock
) -> None:
    decision = decide_trigger(
        new_verified_count=7,
        drifted=True,
        settings=drift_settings,
        clock=fixed_clock,
        last_run_at=datetime(2026, 11, 24, 12, 0, tzinfo=UTC) + timedelta(minutes=30, seconds=15),
    )
    payload = decision.to_dict()
    assert payload["label"] == SYNTHETIC_LABEL
    assert payload["last_run_at"] == "2026-11-24T12:30:15+00:00"
    assert json.loads(json.dumps(payload)) == payload
    assert isinstance(decision, TriggerDecision)
    assert TriggerReason.allows_start("drift") is True
    assert TriggerReason.allows_start("calendar_tick") is False
