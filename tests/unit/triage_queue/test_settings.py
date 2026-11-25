"""Settings validation for P2-02."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from services.triage.queue import TriageQueueSettings


def test_defaults_respect_random_slice_band() -> None:
    settings = TriageQueueSettings()
    assert 0.03 <= settings.random_slice_fraction <= 0.05
    assert settings.random_slice_target is None
    assert 1 <= settings.daily_cap
    assert 0.0 <= settings.blind_share <= 1.0


def test_environment_prefix_is_component_specific(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VOICELOOP_TRIAGE_QUEUE_DAILY_CAP", "5")
    monkeypatch.setenv("VOICELOOP_TRIAGE_QUEUE_BLIND_SHARE", "0.25")
    monkeypatch.setenv("VOICELOOP_TRIAGE_DAILY_CAP", "999")
    settings = TriageQueueSettings()
    assert settings.daily_cap == 5
    assert settings.blind_share == 0.25


def test_fraction_outside_invariant_band_is_rejected() -> None:
    with pytest.raises(ValidationError):
        TriageQueueSettings(random_slice_fraction=0.02)
    with pytest.raises(ValidationError):
        TriageQueueSettings(random_slice_fraction=0.06)


def test_small_pool_override_and_bounds() -> None:
    settings = TriageQueueSettings(random_slice_target=2, daily_cap=8, blind_share=0.5)
    assert settings.random_slice_target == 2
    assert settings.daily_cap == 8
    assert settings.blind_share == 0.5
    with pytest.raises(ValidationError):
        TriageQueueSettings(random_slice_target=-1)
    with pytest.raises(ValidationError):
        TriageQueueSettings(daily_cap=0)
    with pytest.raises(ValidationError):
        TriageQueueSettings(blind_share=1.1)
