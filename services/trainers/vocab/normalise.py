"""Deterministic stdlib-only number and date normalisation rules.

Rules are derived from read-only synthetic fixture failures (``date_confusion``,
``nights_misparsed`` and ``schema_invalid_parse``). They operate on an
ASR-derived transcript plus the slots produced from it; they do not call a
parser, a model, or a clock. The sealed golden set is only ever used for a
read-only regression check, never to tune these rules.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from datetime import date, timedelta
from typing import Any

_CARDINALS: dict[str, int] = {
    "zero": 0,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
    "thirteen": 13,
    "fourteen": 14,
    "fifteen": 15,
    "sixteen": 16,
    "seventeen": 17,
    "eighteen": 18,
    "nineteen": 19,
    "twenty": 20,
    "thirty": 30,
}
_ORDINALS: dict[str, int] = {
    "first": 1,
    "second": 2,
    "third": 3,
    "fourth": 4,
    "fifth": 5,
    "sixth": 6,
    "seventh": 7,
    "eighth": 8,
    "ninth": 9,
    "tenth": 10,
    "eleventh": 11,
    "twelfth": 12,
    "thirteenth": 13,
    "fourteenth": 14,
    "fifteenth": 15,
    "sixteenth": 16,
    "seventeenth": 17,
    "eighteenth": 18,
    "nineteenth": 19,
    "twentieth": 20,
    "twenty-first": 21,
    "twenty-second": 22,
    "twenty-third": 23,
    "thirtieth": 30,
}
_CARDINAL_ALT = "|".join(sorted((re.escape(word) for word in _CARDINALS), key=len, reverse=True))
_ORDINAL_ALT = "|".join(sorted((re.escape(word) for word in _ORDINALS), key=len, reverse=True))
_NUMBER_SOURCE = rf"(?:\d{{1,3}}|{_CARDINAL_ALT})"
_DAY_SOURCE = rf"(?:\d{{1,2}}(?:st|nd|rd|th)?|{_ORDINAL_ALT})"
_COUNT_RE = re.compile(
    rf"\b(?P<number>{_NUMBER_SOURCE})\s+(?P<unit>nights?|guests?)\b", re.IGNORECASE
)
_ARRIVAL_RE = re.compile(
    rf"\b(?:arrive|arrival|check\s*in|from)\s+(?:on\s+)?(?:the\s+)?(?P<day>{_DAY_SOURCE})\b",
    re.IGNORECASE,
)
_DEPARTURE_RE = re.compile(
    rf"\b(?:leave|check\s*out|to)\s+(?:on\s+)?(?:the\s+)?(?P<day>{_DAY_SOURCE})\b",
    re.IGNORECASE,
)


def _number_value(token: str) -> int | None:
    """Convert a cardinal word or digit token to an integer."""
    folded = token.casefold()
    if folded.isdigit():
        return int(folded)
    return _CARDINALS.get(folded)


def _day_value(token: str) -> int | None:
    """Convert an ordinal word or suffixed digit token to a day number."""
    folded = token.casefold().strip()
    if folded[:-2].isdigit() and folded[-2:] in {"st", "nd", "rd", "th"}:
        return int(folded[:-2])
    if folded.isdigit():
        return int(folded)
    return _ORDINALS.get(folded)


def _ordinal_suffix(day: int) -> str:
    """Return the English ordinal suffix for a day number."""
    if 10 <= day % 100 <= 20:
        return "th"
    return {1: "st", 2: "nd", 3: "rd"}.get(day % 10, "th")


def normalise_transcript(text: str) -> str:
    """Canonicalise explicit spoken counts and ordinals into digit forms.

    Off by default in the rule pipeline because the fixture gold transcripts
    keep word forms; this pure helper is available for downstream consumers
    that want a canonical transcript and can score against canonical labels.
    """

    def replace_count(match: re.Match[str]) -> str:
        value = _number_value(match.group("number"))
        if value is None:
            return match.group(0)
        return f"{value} {match.group('unit')}"

    result = _COUNT_RE.sub(replace_count, text)
    ordinal_pattern = re.compile(rf"\bthe\s+(?P<day>{_DAY_SOURCE})\b", re.IGNORECASE)

    def replace_ordinal(match: re.Match[str]) -> str:
        day = _day_value(match.group("day"))
        if day is None:
            return match.group(0)
        return f"the {day}{_ordinal_suffix(day)}"

    return ordinal_pattern.sub(replace_ordinal, result)


def extract_nights(text: str) -> int | None:
    """Return the explicitly spoken number of nights, if present."""
    for match in _COUNT_RE.finditer(text):
        if match.group("unit").casefold().startswith("night"):
            value = _number_value(match.group("number"))
            if value is not None:
                return value
    return None


def extract_guests(text: str) -> int | None:
    """Return the explicitly spoken guest count, if present."""
    for match in _COUNT_RE.finditer(text):
        if match.group("unit").casefold().startswith("guest"):
            value = _number_value(match.group("number"))
            if value is not None:
                return value
    return None


def extract_itinerary_days(text: str) -> tuple[int, int] | None:
    """Return ``(arrival_day, departure_day)`` when both are explicit.

    Only ordinal/day phrases following an arrival or departure cue are used;
    a lone check-in date is deliberately ignored so clean turns stay unchanged.
    """
    arrivals = [
        (match.start(), value)
        for match in _ARRIVAL_RE.finditer(text)
        if (value := _day_value(match.group("day"))) is not None
    ]
    departures = [
        (match.start(), value)
        for match in _DEPARTURE_RE.finditer(text)
        if (value := _day_value(match.group("day"))) is not None
    ]
    if not arrivals or not departures:
        return None
    return arrivals[0][1], departures[0][1]


def normalise_slots(text: str, slots: Mapping[str, Any]) -> tuple[dict[str, Any], tuple[str, ...]]:
    """Apply number/date rules to an ASR-derived slot dict.

    Args:
        text: Transcript produced by the STT stand-in.
        slots: Slots parsed from that transcript.

    Returns:
        A new slot dict and the names of rules that changed it. Inputs are
        never mutated, and a rule fires only when the transcript contradicts
        the slot value, so verified clean turns remain untouched.

    """
    result = dict(slots)
    applied: list[str] = []
    nights = extract_nights(text)
    check_in = result.get("check_in")
    check_out = result.get("check_out")
    if nights is not None and isinstance(check_in, str) and isinstance(check_out, str):
        try:
            expected_out = (date.fromisoformat(check_in) + timedelta(days=nights)).isoformat()
        except (ValueError, OverflowError):
            # Invalid dates or arithmetic beyond datetime's range are not
            # evidence for a rule; leave the slots untouched (fail closed).
            expected_out = ""
        if expected_out and check_out != expected_out:
            result["check_out"] = expected_out
            applied.append("nights_to_checkout")

    guests = extract_guests(text)
    if guests is not None and result.get("guests") != guests:
        result["guests"] = guests
        applied.append("spoken_guest_count")

    days = extract_itinerary_days(text)
    check_in = result.get("check_in")
    check_out = result.get("check_out")
    if days is not None and isinstance(check_in, str) and isinstance(check_out, str):
        try:
            actual = (date.fromisoformat(check_in).day, date.fromisoformat(check_out).day)
        except ValueError:
            actual = (-1, -1)
        expected = days
        if actual != expected and actual == (expected[1], expected[0]):
            result["check_in"], result["check_out"] = check_out, check_in
            applied.append("arrival_departure_order")
    return result, tuple(applied)
