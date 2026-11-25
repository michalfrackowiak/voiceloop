"""Schema-generated form descriptor and validator tests."""

from __future__ import annotations

from typing import Any

import pytest

from ui.errors import UiValidationError
from ui.form import FormDescriptor, build_form_descriptor, validate_slots


def test_descriptor_is_generated_from_pinned_schema(booking_descriptor: FormDescriptor) -> None:
    """The real booking schema supplies fields, required flags and enum values."""
    assert booking_descriptor.allow_unknown_fields is False
    assert booking_descriptor.cross_field_rules == ("check_out_after_check_in",)
    assert booking_descriptor.schema_version == "1.0.0"

    by_path = {field.path: field for field in booking_descriptor.fields}
    assert set(by_path) == {
        "check_in",
        "check_out",
        "contact.email",
        "contact.phone",
        "guest_name",
        "guests",
        "room_type",
    }
    assert {path for path, field in by_path.items() if field.required} == {
        "check_in",
        "check_out",
        "guest_name",
        "guests",
        "room_type",
    }
    assert by_path["room_type"].enum == ("standard", "deluxe", "suite")
    assert by_path["guests"].minimum == 1.0
    assert by_path["check_in"].format == "date"


def test_valid_schema_slots_pass(booking_descriptor: FormDescriptor) -> None:
    """A pseudonymous, schema-valid booking produces no issues."""
    values: dict[str, Any] = {
        "check_in": "2026-12-05",
        "check_out": "2026-12-08",
        "guest_name": "guest-token-0001",
        "guests": 2,
        "room_type": "standard",
        "contact": {"phone": "tok-contact"},
    }

    assert validate_slots(booking_descriptor, values) == ()


def test_check_out_after_check_in_is_enforced(booking_descriptor: FormDescriptor) -> None:
    """Check-out must be strictly later than check-in."""
    values: dict[str, Any] = {
        "check_in": "2026-12-05",
        "check_out": "2026-12-05",
        "guest_name": "guest-token-0001",
        "guests": 1,
        "room_type": "standard",
    }

    codes = {(issue.path, issue.code) for issue in validate_slots(booking_descriptor, values)}

    assert ("check_out", "check_out_after_check_in") in codes


def test_room_type_enum_and_required_fields_are_enforced(
    booking_descriptor: FormDescriptor,
) -> None:
    """Unsupported room types and missing required slots fail closed."""
    values: dict[str, Any] = {
        "check_in": "2026-12-05",
        "check_out": "2026-12-08",
        "guests": 1,
        "room_type": "penthouse",
    }

    codes = {(issue.path, issue.code) for issue in validate_slots(booking_descriptor, values)}

    assert ("room_type", "not_in_enum") in codes
    assert ("guest_name", "required") in codes


def test_unknown_fields_and_invalid_guest_count_are_enforced(
    booking_descriptor: FormDescriptor,
) -> None:
    """Extra slots and out-of-range integers are rejected without echoing values."""
    values: dict[str, Any] = {
        "check_in": "2026-12-05",
        "check_out": "2026-12-08",
        "guest_name": "guest-token-0001",
        "guests": 0,
        "room_type": "suite",
        "notes": "rejected",
    }

    issues = validate_slots(booking_descriptor, values)
    codes = {(issue.path, issue.code) for issue in issues}

    assert ("guests", "below_minimum") in codes
    assert ("notes", "unknown_field") in codes
    assert all("rejected" not in issue.to_dict().values() for issue in issues)


def test_date_and_email_formats_are_enforced(booking_descriptor: FormDescriptor) -> None:
    """Format keywords from the schema are active validators."""
    values: dict[str, Any] = {
        "check_in": "05/12/2026",
        "check_out": "2026-12-08",
        "guest_name": "guest-token-0001",
        "guests": 1,
        "room_type": "deluxe",
        "contact": {"email": "not-an-email"},
    }

    codes = {(issue.path, issue.code) for issue in validate_slots(booking_descriptor, values)}

    assert ("check_in", "invalid_date") in codes
    assert ("contact.email", "invalid_email") in codes


def test_descriptor_rejects_unsupported_schema_shape() -> None:
    """A non-object root cannot be represented by this form harness."""
    with pytest.raises(UiValidationError):
        build_form_descriptor({"type": "array", "items": {"type": "string"}}, "digest")


def test_descriptor_rejects_unsupported_scalar_keyword() -> None:
    """Unsupported draft keywords fail closed instead of being silently ignored."""
    schema: dict[str, Any] = {
        "type": "object",
        "additionalProperties": False,
        "required": ["field"],
        "properties": {"field": {"type": "string", "pattern": "^tok-"}},
    }

    with pytest.raises(UiValidationError):
        build_form_descriptor(schema, "digest")
