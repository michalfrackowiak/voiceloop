import json
from typing import Any

import pytest
from jsonschema.exceptions import ValidationError

from schemas import REPO_ROOT, validate
from schemas.turn_event import load_turn_event_schema, validate_turn_event

EXAMPLES = REPO_ROOT / "schemas" / "examples"
SCHEMA_NAME = "turn_event.schema.json"


def _example(name: str) -> dict[str, Any]:
    return json.loads((EXAMPLES / name).read_text(encoding="utf-8"))


@pytest.mark.parametrize(
    "filename",
    ["turn_event.valid.minimal.json", "turn_event.valid.full.json"],
)
def test_valid_examples_validate(filename: str) -> None:
    payload = _example(filename)
    validate_turn_event(payload)
    validate(payload, SCHEMA_NAME)


@pytest.mark.parametrize(
    ("filename", "expected"),
    [
        ("turn_event.invalid.missing_required.json", "'stt' is a required property"),
        ("turn_event.invalid.bad_status.json", "'cancelled' is not one of"),
    ],
)
def test_invalid_examples_fail_with_useful_errors(filename: str, expected: str) -> None:
    with pytest.raises(ValidationError) as excinfo:
        validate_turn_event(_example(filename))
    message = str(excinfo.value)
    assert expected in message
    assert SCHEMA_NAME in message
    assert "at $" in message


def test_required_fields_are_only_the_ingest_minimum() -> None:
    assert set(load_turn_event_schema()["required"]) == {
        "event_id",
        "call_id",
        "turn_index",
        "audio",
        "stt",
    }


def test_unknown_top_level_field_is_rejected() -> None:
    payload = _example("turn_event.valid.minimal.json")
    payload["surprise"] = True
    with pytest.raises(ValidationError, match="Additional properties are not allowed"):
        validate_turn_event(payload)


def test_empty_stt_text_is_valid() -> None:
    payload = _example("turn_event.valid.minimal.json")
    payload["stt"]["text"] = ""
    validate_turn_event(payload)


def test_invalid_timestamp_is_rejected() -> None:
    payload = _example("turn_event.valid.minimal.json")
    payload["ts"] = "yesterday"
    with pytest.raises(ValidationError, match="date-time"):
        validate_turn_event(payload)


def test_word_confidence_out_of_range_is_rejected() -> None:
    payload = _example("turn_event.valid.full.json")
    payload["stt"]["words"][0]["conf"] = 1.5
    with pytest.raises(ValidationError, match="conf"):
        validate_turn_event(payload)
