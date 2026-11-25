"""P1-09.2: deterministic transcript PII detection with negative controls."""

from __future__ import annotations

import pytest

from services.datasets.filters import PiiCategory, detect_pii, has_card_number


def categories(text: str) -> set[PiiCategory]:
    """Return the detected category set for a text sample."""
    return {finding.category for finding in detect_pii(text)}


def test_card_runs_with_and_without_separators_are_detected() -> None:
    """13–19 digit card-like runs are found with spaces or dashes."""
    samples = (
        "card 4111111111111111 please",
        "card 4111 1111 1111 1111 please",
        "card 5500-0000-0000-0004 please",
        "card 378282246310005 please",
    )
    for text in samples:
        assert PiiCategory.CARD_NUMBER in categories(text), text
        assert has_card_number(text) is True


@pytest.mark.parametrize(
    "separator",
    ["  ", "\t", "\n", "\r\n", "\u00a0", ".", "/", "_", "\u2013"],
)
def test_card_runs_with_repeated_or_unicode_separators_are_detected(
    separator: str,
) -> None:
    """Double spaces, unicode whitespace and non-space separators cannot hide a card."""
    text = f"4111{separator}1111{separator}1111{separator}1111"
    assert has_card_number(text) is True
    assert PiiCategory.CARD_NUMBER in categories(text)


def test_long_contiguous_digit_run_is_hard_dropped() -> None:
    """A phone glued to a card in one 26-digit run still triggers the drop."""
    text = "+15550100001411111111111111"
    assert has_card_number(text) is True
    assert PiiCategory.CARD_NUMBER in categories(text)


def test_card_findings_report_spans_without_raw_values() -> None:
    """Findings carry offsets and categories only."""
    text = "Use card 4111 1111 1111 1111 for this."
    findings = detect_pii(text)
    card = next(f for f in findings if f.category is PiiCategory.CARD_NUMBER)
    assert text[card.start : card.end] == "4111 1111 1111 1111"
    assert card.as_dict() == {
        "category": "card_number",
        "start": card.start,
        "end": card.end,
    }
    assert "4111" not in str(card.as_dict())


def test_email_phone_and_name_patterns_are_detected() -> None:
    """Non-card PII categories return category plus offsets."""
    detected = detect_pii("Mr. Smith can be reached at sam.smith@example.com or +1 555 010 0001.")
    found = categories("Mr. Smith can be reached at sam.smith@example.com or +1 555 010 0001.")
    assert {PiiCategory.PERSON_NAME, PiiCategory.EMAIL, PiiCategory.PHONE} <= found
    assert detected[0].start < detected[-1].end
    assert all(finding.start < finding.end for finding in detected)

    national = detect_pii("Call 555-010-0001 or (555) 010-0001.")
    assert sum(f.category is PiiCategory.PHONE for f in national) == 2

    name_cue = detect_pii("The name is Ada Fixture on the booking.")
    assert any(f.category is PiiCategory.PERSON_NAME for f in name_cue)


def test_negative_controls_do_not_fire_hard_drop() -> None:
    """Dates, short numbers, non-emails and ordinary text are not cards."""
    negatives = (
        "Check in 2026-12-12 and check out 2026-12-14.",
        "Order 12345, table 7, three nights and two guests.",
        "This is not a card: 4111 1111 1111.",
        "not-an-email and foo@bar are not addresses.",
        "Hi, I'd like to book a room for December.",
    )
    for text in negatives:
        assert has_card_number(text) is False, text
    assert PiiCategory.EMAIL not in categories("not-an-email and foo@bar")
    assert PiiCategory.PHONE not in categories("Order 12345, table 7")


def test_clean_synthetic_fixture_transcript_has_no_pii() -> None:
    """A known clean fixture passes the PII screen unchanged."""
    clean_text = "Hi, I'd like to book a room for December."
    assert detect_pii(clean_text) == ()
    assert has_card_number(clean_text) is False
