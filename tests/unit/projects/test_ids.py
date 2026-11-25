"""Tests for field-name sanitization in project ID validation."""

from __future__ import annotations

from typing import cast

import pytest

from services.projects import InvalidProjectIdError, validate_project_id


@pytest.mark.parametrize(
    ("field", "expected"),
    [
        ("project_id", "project_id is required"),
        ("call_id", "call_id is required"),
        ("default_project_id", "default_project_id is required"),
        ("eval_set_name", "eval_set_name is required"),
    ],
)
def test_allowlisted_fields_keep_their_stable_refusal_code(field: str, expected: str) -> None:
    with pytest.raises(InvalidProjectIdError) as exc:
        validate_project_id(None, field=field)
    assert str(exc.value) == expected


@pytest.mark.parametrize(
    "field",
    [
        "ada@example.com",
        "Ada Fixture",
        "+1-555-010-0001",
        "4111111111111111",
        "unknown_field",
        "-field",
        "field;drop",
        "a" * 65,
    ],
)
def test_caller_supplied_field_never_leaks(field: str) -> None:
    with pytest.raises(InvalidProjectIdError) as exc:
        validate_project_id(None, field=field)
    message = str(exc.value)
    assert message == "project_id is required"
    assert field not in message


def test_non_string_field_falls_back_to_fixed_name() -> None:
    with pytest.raises(InvalidProjectIdError) as exc:
        validate_project_id(None, field=cast(str, None))
    assert str(exc.value) == "project_id is required"


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("", "project_id is required"),
        ("x" * 65, "project_id length is outside the allowed range"),
        ("ada@example.com", "project_id must not contain email-like text"),
        ("4111111111111111", "project_id must not contain phone/card-like digit runs"),
        ("Ada", "project_id must be a lowercase pseudonymous slug"),
    ],
)
def test_stable_refusal_codes_are_unchanged_with_hostile_field(
    value: object, expected: str
) -> None:
    with pytest.raises(InvalidProjectIdError) as exc:
        validate_project_id(value, field="ada@example.com")
    assert str(exc.value) == expected
    assert "ada@example.com" not in str(exc.value)
