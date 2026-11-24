import json
from pathlib import Path
from typing import Any

import pytest

from schemas import REPO_ROOT, load_schema, validate
from schemas.booking import (
    BOOKING_SCHEMA_ENV,
    BookingValidationError,
    booking_schema_path,
    load_booking_schema,
    validate_booking,
)

EXAMPLES = REPO_ROOT / "schemas" / "examples"
SCHEMA_NAME = "booking.schema.json"


def _example(name: str) -> dict[str, Any]:
    return json.loads((EXAMPLES / name).read_text(encoding="utf-8"))


def test_schema_metadata_and_slots() -> None:
    schema = load_schema(SCHEMA_NAME)
    assert schema["schema_version"] == "1.0.0"
    assert schema["$id"].startswith("https://voiceloop.example/schemas/")
    assert set(schema["required"]) == {
        "check_in",
        "check_out",
        "guests",
        "room_type",
        "guest_name",
    }
    assert set(schema["properties"]) == {
        "check_in",
        "check_out",
        "guests",
        "room_type",
        "guest_name",
        "contact",
    }
    assert schema["properties"]["room_type"]["enum"] == ["standard", "deluxe", "suite"]


def test_valid_example_validates() -> None:
    record = _example("booking.valid.json")
    validate_booking(record)
    validate(record, SCHEMA_NAME)


@pytest.mark.parametrize(
    ("filename", "expected"),
    [
        ("booking.invalid.missing_required.json", "missing required field: guests"),
        ("booking.invalid.missing_required.json", "missing required field: room_type"),
        ("booking.invalid.dates.json", "check_out must be after check_in"),
        ("booking.invalid.room_type.json", "unknown room_type 'penthouse'"),
        ("booking.invalid.guests.json", "guests must be at least 1"),
        ("booking.invalid.email.json", "contact.email is not a valid email address"),
    ],
)
def test_invalid_examples_fail_with_useful_errors(filename: str, expected: str) -> None:
    with pytest.raises(BookingValidationError) as excinfo:
        validate_booking(_example(filename))
    assert expected in str(excinfo.value)


def test_check_out_equal_to_check_in_is_rejected() -> None:
    record = _example("booking.valid.json")
    record["check_out"] = record["check_in"]
    with pytest.raises(BookingValidationError, match="check_out must be after check_in"):
        validate_booking(record)


def test_contact_is_optional() -> None:
    record = _example("booking.valid.json")
    del record["contact"]
    validate_booking(record)


def test_non_string_email_is_reported_by_schema() -> None:
    record = _example("booking.valid.json")
    record["contact"]["email"] = 123
    with pytest.raises(BookingValidationError, match=r"\$\.contact\.email"):
        validate_booking(record)


def test_extra_field_is_rejected() -> None:
    record = _example("booking.valid.json")
    record["unexpected"] = "value"
    with pytest.raises(BookingValidationError, match="Additional properties are not allowed"):
        validate_booking(record)


def test_swappable_path_via_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    custom = tmp_path / "project_booking.schema.json"
    custom.write_text(
        json.dumps(
            {
                "$schema": "https://json-schema.org/draft/2020-12/schema",
                "type": "object",
                "additionalProperties": False,
                "required": ["guest_name"],
                "properties": {"guest_name": {"type": "string"}},
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv(BOOKING_SCHEMA_ENV, str(custom))
    assert booking_schema_path() == custom
    assert load_booking_schema()["required"] == ["guest_name"]
    validate_booking({"guest_name": "Grace Hopper"})

    monkeypatch.delenv(BOOKING_SCHEMA_ENV)
    with pytest.raises(BookingValidationError, match="missing required field: check_in"):
        validate_booking({"guest_name": "Grace Hopper"})


def test_swappable_path_via_argument(tmp_path: Path) -> None:
    custom = tmp_path / "penthouse.schema.json"
    schema = json.loads((REPO_ROOT / "schemas" / SCHEMA_NAME).read_text(encoding="utf-8"))
    schema["properties"]["room_type"]["enum"] = ["penthouse"]
    custom.write_text(json.dumps(schema), encoding="utf-8")

    record = _example("booking.valid.json")
    record["room_type"] = "penthouse"
    validate_booking(record, schema_path=custom)
    assert booking_schema_path(custom) == custom
    with pytest.raises(BookingValidationError, match="unknown room_type 'penthouse'"):
        validate_booking(record)


def test_missing_schema_file_has_clear_error(tmp_path: Path) -> None:
    missing = tmp_path / "missing.schema.json"
    with pytest.raises(BookingValidationError, match="Booking schema not found"):
        validate_booking(_example("booking.valid.json"), schema_path=missing)
