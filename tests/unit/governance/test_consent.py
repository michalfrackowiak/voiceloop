"""P1-09.1: consent flags gate dataset eligibility."""

from __future__ import annotations

from collections.abc import Callable
from copy import deepcopy
from typing import Any

from services.datasets.filters import (
    DropReason,
    consent_reasons,
    filter_by_consent,
)
from services.ingest.consent import consent_block, is_eligible


def test_clean_synthetic_fixture_is_eligible(clean_event: dict[str, Any]) -> None:
    """Both flags true is the only eligible consent state."""
    assert is_eligible(clean_event) is True
    assert consent_reasons(clean_event) == ()
    assert consent_block(clean_event) == {"recording": True, "training": True}


def test_no_training_consent_fixture_is_ineligible(
    no_training_consent_event: dict[str, Any],
) -> None:
    """The named P0-05 consent edge case must fail closed."""
    assert is_eligible(no_training_consent_event) is False
    assert consent_reasons(no_training_consent_event) == (DropReason.TRAINING_CONSENT_DENIED,)


def test_missing_or_malformed_flags_fail_closed(
    clean_event: dict[str, Any],
    with_transcript: Callable[[dict[str, Any], str], dict[str, Any]],
) -> None:
    """Missing consent blocks and non-boolean flags are ineligible."""
    without_consent = deepcopy(clean_event)
    del without_consent["consent"]
    assert is_eligible(without_consent) is False
    assert consent_reasons(without_consent) == (DropReason.CONSENT_MISSING,)

    recording_denied = deepcopy(clean_event)
    recording_denied["consent"]["recording"] = False
    assert is_eligible(recording_denied) is False
    assert consent_reasons(recording_denied) == (DropReason.RECORDING_CONSENT_DENIED,)

    not_a_bool = with_transcript(clean_event, "clean")
    not_a_bool["consent"]["training"] = "true"
    assert is_eligible(not_a_bool) is False
    assert DropReason.TRAINING_CONSENT_DENIED in consent_reasons(not_a_bool)


def test_filter_by_consent_excludes_exactly_the_denied_fixture(
    fixture_events: list[dict[str, Any]],
) -> None:
    """The consent filter produces auditable drops without touching events."""
    original = deepcopy(fixture_events)
    kept, dropped = filter_by_consent(fixture_events)

    assert fixture_events == original
    assert len(fixture_events) == 36
    assert len(kept) == 35
    assert len(dropped) == 1
    record = dropped[0]
    assert record.call_id == "call-0005"
    assert record.turn_index == 2
    assert record.reasons == (DropReason.TRAINING_CONSENT_DENIED,)
    assert record.event_id == no_training_consent_id(fixture_events)


def no_training_consent_id(events: list[dict[str, Any]]) -> str:
    """Return the event id of the named no-training-consent fixture."""
    return next(
        event["event_id"]
        for event in events
        if event.get("call_id") == "call-0005" and event.get("turn_index") == 2
    )
