"""Tests for deterministic, bounded, PII-safe call-level traffic splitting."""

from __future__ import annotations

import pytest

from services.release.canary import Arm, CanarySettings, TrafficSplit
from voiceloop_common import ValidationError


def test_assignment_is_deterministic(settings: CanarySettings) -> None:
    """The same pseudonymous call ID always maps to the same arm and bucket."""
    split = TrafficSplit(settings)
    first = split.assign("call-0002")
    second = split.assign("call-0002")
    assert first == second
    assert first.call_id == "call-0002"
    assert 0.0 <= first.bucket < 1.0
    assert first.arm in {Arm.CANARY, Arm.CHAMPION}


def test_bounded_fraction_produces_both_arms(settings: CanarySettings) -> None:
    """At 20% with a fixed salt, the calls stay inside the explicit bound."""
    split = TrafficSplit(settings)
    ids = [f"call-{index:04d}" for index in range(1, 7)]
    counts = split.counts(ids)
    assert counts[Arm.CANARY.value] == 3
    assert counts[Arm.CHAMPION.value] == 3
    assert sum(counts.values()) == len(ids)


def test_test_and_zero_fractions(settings: CanarySettings) -> None:
    """A zero fraction sends everything to control; full-cap stays below 100%."""
    assert TrafficSplit(CanarySettings(canary_fraction=0.0)).counts(["a", "b"]) == {
        Arm.CANARY.value: 0,
        Arm.CHAMPION.value: 2,
    }
    capped = TrafficSplit(CanarySettings(canary_fraction=0.20, split_salt=settings.split_salt))
    ids = [f"call-{index:04d}" for index in range(1, 101)]
    counts = capped.counts(ids)
    assert 0 < counts[Arm.CANARY.value] < len(ids)


def test_salt_changes_buckets(settings: CanarySettings) -> None:
    """Different salts move buckets; the configured salt is exposed safely."""
    original = TrafficSplit(settings)
    changed = TrafficSplit(CanarySettings(split_salt="other-salt", canary_fraction=0.20))
    assert original.salt == settings.split_salt
    assert original.assign("call-0002").bucket != changed.assign("call-0002").bucket


def test_raw_digit_runs_and_blank_ids_are_rejected(settings: CanarySettings) -> None:
    """Unredacted phone/card-like IDs and blank IDs never enter an audit field."""
    split = TrafficSplit(settings)
    with pytest.raises(ValidationError):
        split.assign("1234567890123456")
    with pytest.raises(ValidationError):
        split.assign("")
    with pytest.raises(ValidationError):
        split.assign("   ")


def test_audit_fields_are_pseudonymous(settings: CanarySettings) -> None:
    """Split audit fields contain only the call ID, arm and numeric bucket."""
    fields = TrafficSplit(settings).assign("call-0002").to_fields()
    assert set(fields) == {"call_id", "arm", "bucket"}
    assert fields["call_id"] == "call-0002"
    assert fields["arm"] in {"canary", "champion"}
