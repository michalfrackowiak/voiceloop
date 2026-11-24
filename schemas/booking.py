"""Booking slot validation with a per-project swappable schema path."""

from __future__ import annotations

import json
import os
import re
from datetime import date
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator, FormatChecker
from jsonschema.exceptions import ValidationError

from . import REPO_ROOT
from . import schema_path as resolve_schema_path

BOOKING_SCHEMA_ENV = "VOICELOOP_BOOKING_SCHEMA"
DEFAULT_BOOKING_SCHEMA = "booking.schema.json"
EXAMPLE_REQUIRED_FIELDS = ("check_in", "check_out", "guests", "room_type", "guest_name")
KNOWN_ROOM_TYPES = ("standard", "deluxe", "suite")
_EMAIL_PATTERN = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


class BookingValidationError(ValueError):
    """Raised when a booking record violates the active schema or validators."""


def booking_schema_path(override: str | Path | None = None) -> Path:
    """Return the active booking schema path.

    Precedence is the explicit argument, then :data:`BOOKING_SCHEMA_ENV`, then the
    bundled example schema. Relative paths are resolved against the repository root.

    Args:
        override: Optional path to a project-specific schema file.

    Returns:
        Absolute path to the schema file to validate against.

    """
    raw: str | Path | None = (
        override if override is not None else os.environ.get(BOOKING_SCHEMA_ENV)
    )
    if raw is None or (isinstance(raw, str) and not raw.strip()):
        return resolve_schema_path(DEFAULT_BOOKING_SCHEMA)
    candidate = Path(raw).expanduser()
    if not candidate.is_absolute():
        candidate = REPO_ROOT / candidate
    return candidate.resolve()


def load_booking_schema(schema_path: str | Path | None = None) -> dict[str, Any]:
    """Load the active booking schema.

    Args:
        schema_path: Optional path override, see :func:`booking_schema_path`.

    Returns:
        The parsed schema as a dictionary.

    Raises:
        BookingValidationError: If the file is missing or is not a JSON object.

    """
    return _read_schema(booking_schema_path(schema_path))


def validate_booking(record: dict[str, Any], schema_path: str | Path | None = None) -> None:
    """Validate a booking record and its cross-field rules.

    Args:
        record: Decoded booking slots.
        schema_path: Optional path override, see :func:`booking_schema_path`.

    Raises:
        BookingValidationError: With one or more clear, semicolon-separated messages.

    """
    schema = load_booking_schema(schema_path)
    custom_errors = _custom_errors(record, schema)
    if custom_errors:
        raise BookingValidationError("; ".join(custom_errors))

    validator = Draft202012Validator(schema, format_checker=FormatChecker())
    errors = sorted(
        validator.iter_errors(record),
        key=lambda error: [str(part) for part in error.path],
    )
    if errors:
        raise BookingValidationError("; ".join(_describe(error) for error in errors))


def _read_schema(path: Path) -> dict[str, Any]:
    """Read a JSON object from a schema file, with a clear error on failure."""
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise BookingValidationError(f"Booking schema not found or unreadable: {path}") from exc
    try:
        schema = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise BookingValidationError(f"Invalid JSON in booking schema {path}: {exc}") from exc
    if not isinstance(schema, dict):
        raise BookingValidationError(f"Booking schema {path} must contain a JSON object")
    return schema


def _custom_errors(record: dict[str, Any], schema: dict[str, Any]) -> list[str]:
    """Collect messages for checks that JSON Schema cannot express cleanly."""
    if not isinstance(record, dict):
        return ["booking record must be a JSON object"]

    errors: list[str] = []
    required = schema.get("required", [])
    if isinstance(required, list):
        errors.extend(
            f"missing required field: {field}"
            for field in required
            if isinstance(field, str) and field not in record
        )

    guests = record.get("guests")
    minimum = _guests_minimum(schema)
    if isinstance(guests, int) and not isinstance(guests, bool) and guests < minimum:
        errors.append(f"guests must be at least {minimum} (got {guests})")

    room_type = record.get("room_type")
    known_room_types = _known_room_types(schema)
    if isinstance(room_type, str) and room_type not in known_room_types:
        expected = ", ".join(known_room_types)
        errors.append(f"unknown room_type {room_type!r} (expected one of: {expected})")

    check_in = record.get("check_in")
    check_out = record.get("check_out")
    if isinstance(check_in, str) and isinstance(check_out, str):
        start, end = _parse_date(check_in), _parse_date(check_out)
        if start is not None and end is not None and end <= start:
            errors.append(
                f"check_out must be after check_in (got check_in={check_in}, check_out={check_out})"
            )

    contact = record.get("contact")
    if isinstance(contact, dict) and "email" in contact:
        email = contact["email"]
        if isinstance(email, str) and not _EMAIL_PATTERN.fullmatch(email.strip()):
            errors.append(f"contact.email is not a valid email address (got {email!r})")

    return errors


def _guests_minimum(schema: dict[str, Any]) -> int:
    """Read the guests minimum from the active schema, defaulting to one."""
    properties = schema.get("properties", {})
    guests = properties.get("guests", {}) if isinstance(properties, dict) else {}
    value = guests.get("minimum", 1) if isinstance(guests, dict) else 1
    return value if isinstance(value, int) and not isinstance(value, bool) else 1


def _known_room_types(schema: dict[str, Any]) -> tuple[str, ...]:
    """Read allowed room types from the active schema, defaulting to the example set."""
    properties = schema.get("properties", {})
    room_type = properties.get("room_type", {}) if isinstance(properties, dict) else {}
    values = room_type.get("enum") if isinstance(room_type, dict) else None
    if isinstance(values, list) and values and all(isinstance(value, str) for value in values):
        return tuple(values)
    return KNOWN_ROOM_TYPES


def _parse_date(value: str) -> date | None:
    """Parse an ISO date, returning None when the schema should report the format error."""
    try:
        return date.fromisoformat(value)
    except ValueError:
        return None


def _describe(error: ValidationError) -> str:
    """Render a jsonschema error as a readable message with its JSON path."""
    if error.validator == "format" and error.path and error.path[-1] == "email":
        return f"{error.json_path}: is not a valid email address"
    return f"{error.json_path}: {error.message}"
