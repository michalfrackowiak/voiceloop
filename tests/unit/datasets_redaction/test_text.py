"""Transcript pseudonymisation: stable salts, hard-drop cards and no leaks."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pytest

from services.datasets.filters import detect_pii
from services.datasets.redaction.settings import (
    InMemoryAuditSink,
    RedactionSettings,
)
from services.datasets.redaction.spans import CardNumberRejected, RedactionError
from services.datasets.redaction.text import (
    InMemoryTranscriptSource,
    TranscriptRecord,
    make_pseudonym,
    pseudonymise_text,
    pseudonymise_transcript,
)
from voiceloop_common import StorageError

TURNS_PATH = Path(__file__).resolve().parents[3] / "tests" / "fixtures" / "turns.jsonl"


def _fixture_rows() -> list[dict[str, Any]]:
    """Load the read-only synthetic label manifest."""
    return [
        json.loads(line)
        for line in TURNS_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _names(row: dict[str, Any]) -> tuple[str, ...]:
    """Extract the reviewed guest name from a synthetic label row."""
    slots = row.get("expected_slots", {})
    name = slots.get("guest_name") if isinstance(slots, dict) else None
    return (str(name),) if isinstance(name, str) and name else ()


def test_pseudonym_is_stable_per_call_and_differs_across_calls(
    settings: RedactionSettings,
) -> None:
    text = "The name is Ada Fixture."
    first = pseudonymise_text(
        text, call_id="call-1", settings=settings, known_names=["Ada Fixture"]
    )
    second = pseudonymise_text(
        text, call_id="call-1", settings=settings, known_names=["Ada Fixture"]
    )
    other = pseudonymise_text(
        text, call_id="call-2", settings=settings, known_names=["Ada Fixture"]
    )
    assert first.text == second.text
    assert first.text != other.text
    assert first.replacements[0].token.startswith("[PERSON_")
    assert "Ada" not in first.text


def test_default_api_pseudonymises_full_cued_name_without_known_names(
    settings: RedactionSettings,
) -> None:
    """Regression: a bare-title finding must not leave a cued surname behind."""
    text = "The name is Mr. Smith."
    result = pseudonymise_text(text, call_id="call-cued", settings=settings)
    assert "Smith" not in result.text
    assert "Mr" not in result.text
    assert result.text == f"The name is {result.replacements[0].token}."
    assert (result.replacements[0].start, result.replacements[0].end) == (12, 21)
    assert detect_pii(result.text) == ()


def test_replay_is_byte_identical(settings: RedactionSettings) -> None:
    text = "The name is Mr. Smith. Email a@example.com."
    first = pseudonymise_text(text, call_id="call-9", settings=settings, known_names=["Mr. Smith"])
    second = pseudonymise_text(text, call_id="call-9", settings=settings, known_names=["Mr. Smith"])
    assert first.to_dict() == second.to_dict()


def test_non_pii_text_is_preserved(settings: RedactionSettings) -> None:
    text = "Book a deluxe room for two guests, please."
    result = pseudonymise_text(text, call_id="call-3", settings=settings)
    assert result.text == text
    assert result.findings == ()
    assert result.replacements == ()


def test_email_and_phone_are_masked_not_pseudonymised(settings: RedactionSettings) -> None:
    text = "Mail a@example.com or call +1 555 010 1234."
    result = pseudonymise_text(text, call_id="call-4", settings=settings)
    assert result.text == "Mail [EMAIL_REDACTED] or call [PHONE_REDACTED]."
    assert detect_pii(result.text) == ()
    assert {item.category.value for item in result.findings} == {"email", "phone"}


def test_card_number_is_hard_dropped_with_structured_error(
    settings: RedactionSettings,
) -> None:
    text = "The card is 4111 1111 1111 1111."
    with pytest.raises(CardNumberRejected) as exc:
        pseudonymise_text(text, call_id="call-5", settings=settings)
    assert exc.value.code == "card_number_pii"
    assert "4111" not in json.dumps(exc.value.to_dict())
    assert "PERSON_" not in str(exc.value)


def test_known_name_never_appears_in_output_or_audit(
    settings: RedactionSettings,
) -> None:
    audit = InMemoryAuditSink()
    text = "The name is Mr. Smith."
    result = pseudonymise_text(
        text,
        call_id="call-6",
        settings=settings,
        known_names=["Mr. Smith"],
        audit=audit,
    )
    assert "Smith" not in result.text
    assert "Smith" not in json.dumps(result.to_dict())
    assert "Smith" not in json.dumps(audit.to_dicts())
    assert result.findings[0].start == 12
    assert result.replacements[0].token == make_pseudonym("call-6", "Mr. Smith", settings)


def test_transcript_source_protocol_and_missing_key(settings: RedactionSettings) -> None:
    source = InMemoryTranscriptSource()
    source.add(TranscriptRecord("turn-1", "call-7", "The name is Ada Fixture.", ("Ada Fixture",)))
    result = pseudonymise_transcript(source, "turn-1", settings=settings)
    assert "Ada" not in result.text
    with pytest.raises(StorageError):
        pseudonymise_transcript(source, "turn-missing", settings=settings)


def test_invalid_pseudonym_input_fails_closed(settings: RedactionSettings) -> None:
    with pytest.raises(RedactionError) as exc:
        make_pseudonym("", "Ada Fixture", settings)
    assert exc.value.code == "invalid_pseudonym_input"


def test_all_redacted_fixture_transcripts_are_pii_free(
    settings: RedactionSettings,
) -> None:
    rows = _fixture_rows()
    assert len(rows) == 36
    for row in rows:
        text = str(row["expected_transcript"])
        names = _names(row)
        result = pseudonymise_text(
            text, call_id=str(row["call_id"]), settings=settings, known_names=names
        )
        assert detect_pii(result.text) == ()
        for name in names:
            assert (
                re.search(r"(?<!\w)" + re.escape(name) + r"(?!\w)", result.text, re.IGNORECASE)
                is None
            )
        slots = row.get("expected_slots", {})
        email = slots.get("contact", {}).get("email") if isinstance(slots, dict) else None
        if isinstance(email, str):
            assert email not in result.text


def test_residual_pii_is_detected_before_return(settings: RedactionSettings) -> None:
    """A detector that still finds PII after replacement fails the output."""
    from services.datasets.filters import PiiCategory, PiiFinding
    from services.datasets.redaction.spans import UnverifiedRedactionError

    class LeakyDetector:
        """Report the same synthetic phone finding before and after masking."""

        def detect(self, text: str) -> tuple[PiiFinding, ...]:
            """Always report one phone finding at offset 0."""
            return (PiiFinding(PiiCategory.PHONE, 0, 1),)

    with pytest.raises(UnverifiedRedactionError) as exc:
        pseudonymise_text(
            "x y",
            call_id="call-8",
            settings=settings,
            detector=LeakyDetector(),
        )
    assert exc.value.code == "unverified_text_pii"
