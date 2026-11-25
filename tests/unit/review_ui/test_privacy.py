"""Privacy-screen regression tests for exact schema descriptions."""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

import pytest

from ui.constants import SCHEMA_PATH
from ui.errors import UiValidationError
from ui.privacy import assert_no_sensitive_text


def _descriptions(node: Any) -> set[str]:
    """Collect pinned description strings from a JSON-like schema payload."""
    found: set[str] = set()
    if isinstance(node, Mapping):
        for key, value in node.items():
            if key == "description" and isinstance(value, str):
                found.add(value)
            found.update(_descriptions(value))
    elif isinstance(node, (list, tuple)):
        for value in node:
            found.update(_descriptions(value))
    return found


def test_exact_pinned_schema_descriptions_are_allowed() -> None:
    """Only descriptions from the hash-pinned schema are exempt from screening."""
    schema: Any = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    descriptions = _descriptions(schema)

    assert len(descriptions) == 5
    for description in descriptions:
        assert_no_sensitive_text({"description": description})


def test_injected_pii_description_is_screened() -> None:
    """A PII-shaped value under ``description`` is ordinary text, not a bypass."""
    payload = {
        "description": "Guest ada.fixture@example.com called +1 555 123 4567",
    }

    with pytest.raises(UiValidationError):
        assert_no_sensitive_text(payload)


def test_modified_pinned_description_is_screened() -> None:
    """A near-miss description does not inherit the schema exemption."""
    pinned = "ISO 8601 check-in date (YYYY-MM-DD)."

    with pytest.raises(UiValidationError):
        assert_no_sensitive_text({"description": pinned + " ada.fixture@example.com"})


def test_non_string_description_with_pii_is_screened() -> None:
    """Nested values under ``description`` are still traversed."""
    payload = {"description": ["guest", "ada.fixture@example.com"]}

    with pytest.raises(UiValidationError):
        assert_no_sensitive_text(payload)
