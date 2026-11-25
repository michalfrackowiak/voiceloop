"""Number and date normalisation rule tests."""

from __future__ import annotations

from typing import Any

from services.trainers.vocab import normalise_slots, normalise_transcript
from services.trainers.vocab.normalise import (
    extract_guests,
    extract_itinerary_days,
    extract_nights,
)


def _slots(**changes: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "check_in": "2026-12-05",
        "check_out": "2026-12-07",
        "contact": {"email": "sam.sample@example.com", "phone": "+1 555 010 0002"},
        "guest_name": "Sam Sample",
        "guests": 1,
        "room_type": "standard",
    }
    base.update(changes)
    return base


def test_spoken_counts_are_extracted() -> None:
    text = "I'd like three nights and two guests, please."
    assert extract_nights(text) == 3
    assert extract_guests(text) == 2
    assert extract_nights("Check in December fifth for two nights, one guest.") == 2
    assert extract_guests("Make it four guests.") == 4
    assert extract_guests("Make it a deluxe room.") is None


def test_date_confusion_fixture_is_fixed() -> None:
    text = "I'd like three nights and two guests, please."
    result, applied = normalise_slots(text, _slots(guests=3))
    assert result["check_out"] == "2026-12-08"
    assert result["guests"] == 2
    assert applied == ("nights_to_checkout", "spoken_guest_count")
    assert result["room_type"] == "standard"


def test_nights_misparsed_fixture_is_fixed() -> None:
    result, applied = normalise_slots(
        "I'll stay four nights.",
        _slots(check_in="2027-02-02", check_out="2027-02-04", guests=2),
    )
    assert result["check_out"] == "2027-02-06"
    assert applied == ("nights_to_checkout",)


def test_swapped_ordinal_dates_are_detected_and_reordered() -> None:
    text = "Arrive on the tenth and leave on the fourteenth."
    assert extract_itinerary_days(text) == (10, 14)
    result, applied = normalise_slots(text, _slots(check_in="2026-12-14", check_out="2026-12-10"))
    assert (result["check_in"], result["check_out"]) == ("2026-12-10", "2026-12-14")
    assert applied == ("arrival_departure_order",)


def test_lone_check_in_date_does_not_trigger_a_swap() -> None:
    assert extract_itinerary_days("Check in on the twelfth of December.") is None
    result, applied = normalise_slots("Check in on the twelfth of December.", _slots())
    assert result == _slots()
    assert applied == ()


def test_agreed_counts_are_left_untouched() -> None:
    slots = _slots()
    result, applied = normalise_slots("Check in December fifth for two nights, one guest.", slots)
    assert result == slots
    assert applied == ()


def test_invalid_iso_date_does_not_raise() -> None:
    result, applied = normalise_slots("three nights", _slots(check_in="not-a-date"))
    assert result["check_in"] == "not-a-date"
    assert "nights_to_checkout" not in applied


def test_overflowing_nights_rule_fails_closed_and_stays_idempotent() -> None:
    slots = _slots(check_in="9999-12-31", check_out="9999-12-30")
    result, applied = normalise_slots("999 nights", slots)
    assert result == slots
    assert applied == ()
    again, again_applied = normalise_slots("999 nights", result)
    assert again == result
    assert again_applied == ()


def test_canonical_transcript_helper_is_deterministic() -> None:
    assert normalise_transcript("I'd like three nights and two guests.") == (
        "I'd like 3 nights and 2 guests."
    )
    assert normalise_transcript("Arrive on the tenth and leave on the fourteenth.") == (
        "Arrive on the 10th and leave on the 14th."
    )
    assert normalise_transcript("Make it a deluxe room.") == "Make it a deluxe room."
