"""Slot mismatch signals: final booking record and hypothesis-vs-parser."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from datetime import date
from typing import Any

from .settings import TriageScoringSettings
from .signals import Observation, _as_mapping, _number

_NUMBER_WORDS: dict[str, int] = {
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
}
_NUMBER_TOKEN = "|".join((r"\d+", *_NUMBER_WORDS))
_NIGHTS_PATTERN = re.compile(rf"\b(?P<number>{_NUMBER_TOKEN})\s+nights?\b", re.IGNORECASE)
_GUESTS_PATTERN = re.compile(rf"\b(?P<number>{_NUMBER_TOKEN})\s+guests?\b", re.IGNORECASE)
_MISSING = "<missing>"


def _canonical(value: Any) -> str:
    """Return a stable JSON string for comparison of opaque slot values."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def _flatten(record: Mapping[str, Any], prefix: str = "") -> dict[str, Any]:
    """Flatten nested slot objects into dotted field paths."""
    flattened: dict[str, Any] = {}
    for key, value in record.items():
        path = f"{prefix}.{key}" if prefix else str(key)
        child = _as_mapping(value)
        if child is not None:
            flattened.update(_flatten(child, path))
        else:
            flattened[path] = value
    return flattened


def _display(value: Any) -> str:
    """Return a short, deterministic value preview for explanations."""
    rendered = _canonical(value)
    return rendered if len(rendered) <= 60 else rendered[:57] + "..."


def _spoken_number(text: str, pattern: re.Pattern[str]) -> int | None:
    """Return the first number immediately preceding ``nights`` or ``guests``."""
    match = pattern.search(text)
    if match is None:
        return None
    token = match.group("number")
    if token.isdigit():
        return int(token)
    return _NUMBER_WORDS.get(token.lower())


def _parsed_nights(output: Mapping[str, Any]) -> int | None:
    """Return nights implied by check-in/check-out, when both parse correctly."""
    check_in = output.get("check_in")
    check_out = output.get("check_out")
    if not isinstance(check_in, str) or not isinstance(check_out, str):
        return None
    try:
        start = date.fromisoformat(check_in)
        end = date.fromisoformat(check_out)
    except ValueError:
        return None
    nights = (end - start).days
    return nights if nights > 0 else None


def detect_final_record_mismatch(
    event: Mapping[str, Any], settings: TriageScoringSettings
) -> list[Observation]:
    """Compare parser hypotheses against ``outcome.final_record`` when present.

    This signal carries the highest configured weight (context.md section 6).
    A missing, empty or non-object final record fires nothing: outcomes arrive
    late and must not be inferred.

    Args:
        event: A schema-valid turn event.
        settings: Weights for this run.

    Returns:
        Zero or one final-record mismatch observation.

    """
    parser = _as_mapping(event.get("parser"))
    parsed = _as_mapping(parser.get("output")) if parser is not None else None
    outcome = _as_mapping(event.get("outcome"))
    final = _as_mapping(outcome.get("final_record")) if outcome is not None else None
    if not parsed or not final:
        return []

    parsed_fields = _flatten(parsed)
    final_fields = _flatten(final)
    differing = sorted(
        key
        for key in parsed_fields.keys() | final_fields.keys()
        if _canonical(parsed_fields.get(key, _MISSING))
        != _canonical(final_fields.get(key, _MISSING))
    )
    if not differing:
        return []
    preview = ", ".join(
        f"{key}: {_display(parsed_fields.get(key, _MISSING))} -> "
        f"{_display(final_fields.get(key, _MISSING))}"
        for key in differing[:3]
    )
    return [
        Observation(
            name="final_record_mismatch",
            weight=settings.weight_final_record_mismatch,
            severity=1.0,
            reason=(
                f"parser output differs from final booking record on {len(differing)} "
                f"field(s): {preview}"
            ),
        )
    ]


def detect_hypothesis_parser_mismatch(
    event: Mapping[str, Any], settings: TriageScoringSettings
) -> list[Observation]:
    """Compare explicit guest numbers in the hypothesis with parsed slots.

    Final records are not available on most turns, so parser-only errors such
    as "three nights" becoming two nights are otherwise invisible. This signal
    uses only the event's own STT text and parser output: no label or expected
    slot from the fixture manifest is consulted.

    Args:
        event: A schema-valid turn event.
        settings: Weights for this run.

    Returns:
        Zero, one or two slot mismatch observations.

    """
    parser = _as_mapping(event.get("parser"))
    output = _as_mapping(parser.get("output")) if parser is not None else None
    if not output:
        return []
    stt = _as_mapping(event.get("stt"))
    text = stt.get("text", "") if stt is not None else ""
    if not isinstance(text, str):
        return []

    observations: list[Observation] = []
    spoken_guests = _spoken_number(text, _GUESTS_PATTERN)
    parsed_guests = _number(output.get("guests"))
    if spoken_guests is not None and parsed_guests != float(spoken_guests):
        observations.append(
            Observation(
                name="hypothesis_guest_count_mismatch",
                weight=settings.weight_hypothesis_parser_mismatch,
                severity=1.0,
                reason=(
                    f"hypothesis says {spoken_guests} guest(s); parser guests="
                    f"{_display(output['guests']) if 'guests' in output else _MISSING}"
                ),
            )
        )

    spoken_nights = _spoken_number(text, _NIGHTS_PATTERN)
    parsed_nights = _parsed_nights(output)
    if spoken_nights is not None and parsed_nights != spoken_nights:
        observations.append(
            Observation(
                name="hypothesis_nights_mismatch",
                weight=settings.weight_hypothesis_parser_mismatch,
                severity=1.0,
                reason=(
                    f"hypothesis says {spoken_nights} night(s); parser check-out minus "
                    f"check-in is {parsed_nights if parsed_nights is not None else _MISSING}"
                ),
            )
        )
    return observations


def detect_mismatches(
    event: Mapping[str, Any], settings: TriageScoringSettings
) -> list[Observation]:
    """Return final-record and hypothesis-vs-parser mismatch observations."""
    return detect_final_record_mismatch(event, settings) + detect_hypothesis_parser_mismatch(
        event, settings
    )
