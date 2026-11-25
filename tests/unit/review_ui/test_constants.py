"""Fixed labels, pinned hashes and privacy-screen tests."""

from __future__ import annotations

import pytest

from ui.constants import (
    EXPECTED_BOOKING_SCHEMA_SHA256,
    EXPECTED_FIXTURE_SHA256,
    EXPECTED_GOLDEN_CONTENT_HASH,
    SYNTHETIC_LABEL,
    USABILITY_STATUS,
)
from ui.errors import UiValidationError
from ui.hashes import verify_read_only_inputs
from ui.privacy import assert_no_sensitive_text


def test_pinned_hashes_match_read_only_inputs() -> None:
    """The fixture, booking schema and golden stand-in are unchanged."""
    result = verify_read_only_inputs().to_dict()

    assert result["fixture_sha256"] == EXPECTED_FIXTURE_SHA256
    assert result["booking_schema_sha256"] == EXPECTED_BOOKING_SCHEMA_SHA256
    assert result["golden_content_hash"] == EXPECTED_GOLDEN_CONTENT_HASH
    assert result["fixture_hash_ok"] is True
    assert result["booking_schema_hash_ok"] is True
    assert result["golden_hash_ok"] is True
    assert result["golden_synthetic"] is True
    assert result["golden_workflow_stand_in"] is True


def test_fixed_labels_declare_no_human_usability_result() -> None:
    """Public labels cannot be mistaken for a human study result."""
    assert SYNTHETIC_LABEL == "SYNTHETIC/FIXTURE STAND-IN - NOT A HUMAN USABILITY RESULT"
    assert USABILITY_STATUS == "NOT DONE"


@pytest.mark.parametrize(
    "value",
    [
        "ada.fixture@example.com",
        "+1 555 010 0001",
        "1234567",
        "s3://bucket/turn.wav",
        "http://example.invalid",
        "Ada Fixture",
    ],
)
def test_privacy_screen_blocks_pii_and_prose(value: str) -> None:
    """Emails, phone-like strings, long digits, URIs and prose fail closed."""
    with pytest.raises(UiValidationError):
        assert_no_sensitive_text({"note": value})


@pytest.mark.parametrize(
    "value",
    [
        "item-token-0001",
        "tok-a",
        "reviewer-token-02",
        "guest-token-0001",
        "2026-12-05",
        "standard",
        "NOT DONE",
        SYNTHETIC_LABEL,
    ],
)
def test_privacy_screen_allows_pseudonyms_and_fixed_labels(value: str) -> None:
    """Only the documented pseudonym grammar and fixed labels are allowed."""
    assert_no_sensitive_text({"value": value})
