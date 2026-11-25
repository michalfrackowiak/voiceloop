"""P1-09 acceptance: non-consented and card rows are provably excluded."""

from __future__ import annotations

from collections.abc import Callable
from copy import deepcopy
from typing import Any

import pytest

from services.datasets.filters import (
    DropReason,
    PiiCategory,
    filter_eligible_events,
)

CARD_VARIANTS = (
    "My card is 4111 1111 1111 1111, charge the booking.",
    "The number is 5500-0000-0000-0004, please.",
    "Use 378282246310005 for payment.",
)


def test_all_card_variants_are_hard_dropped(
    clean_event: dict[str, Any],
    with_transcript: Callable[[dict[str, Any], str], dict[str, Any]],
) -> None:
    """A card in the transcript always drops the row, never redacts it."""
    for text in CARD_VARIANTS:
        event = with_transcript(clean_event, text)
        kept, dropped = filter_eligible_events([event])
        assert kept == []
        assert len(dropped) == 1
        assert DropReason.CARD_NUMBER_PII in dropped[0].reasons
        assert any(f.category is PiiCategory.CARD_NUMBER for f in dropped[0].pii_findings)
        assert event["stt"]["text"] == text


@pytest.mark.parametrize(
    "text",
    [
        "Charge +1 555 010 0001 4111 1111 1111 1111 now",
        "card4111111111111111",
        "4111111111111111x",
        "Use 4111-11-1111-1111 for payment.",
        "Use 4111 11 1111 1111 for payment.",
        "Charge 4111  1111  1111  1111 now",
        "Charge 4111\t1111\t1111\t1111 now",
        "Charge 4111\n1111\n1111\n1111 now",
        "Charge 4111\r\n1111\r\n1111\r\n1111 now",
        "Charge 4111\u00a01111\u00a01111\u00a01111 now",
        "Charge 4111.1111.1111.1111 now",
        "Charge 4111/1111/1111/1111 now",
        "Charge 4111_1111_1111_1111 now",
        "Charge 4111\u20131111\u20131111\u20131111 now",
        "+15550100001411111111111111",
    ],
)
def test_reviewer_bypass_cases_are_hard_dropped(
    text: str,
    clean_event: dict[str, Any],
    with_transcript: Callable[[dict[str, Any], str], dict[str, Any]],
) -> None:
    """Adjacency, separator and long-run bypasses cannot hide a card."""
    event = with_transcript(clean_event, text)
    kept, dropped = filter_eligible_events([event])

    assert kept == []
    assert len(dropped) == 1
    assert DropReason.CARD_NUMBER_PII in dropped[0].reasons
    assert any(f.category is PiiCategory.CARD_NUMBER for f in dropped[0].pii_findings)


def test_non_consented_fixture_and_card_are_both_excluded(
    fixture_events: list[dict[str, Any]],
    clean_event: dict[str, Any],
    no_training_consent_event: dict[str, Any],
    with_transcript: Callable[[dict[str, Any], str], dict[str, Any]],
) -> None:
    """The two acceptance cases are excluded together with distinct reasons."""
    card_event = with_transcript(clean_event, "Card 4111 1111 1111 1111.")
    combined = with_transcript(no_training_consent_event, "Card 4111 1111 1111 1111.")
    events = [*fixture_events, card_event, combined]
    original = deepcopy(events)

    kept, dropped = filter_eligible_events(events)

    assert events == original
    assert len(events) == 38
    assert len(kept) == 35
    assert len(dropped) == 3
    denied_records = [
        record for record in dropped if DropReason.TRAINING_CONSENT_DENIED in record.reasons
    ]
    assert len(denied_records) == 2
    assert all((record.call_id, record.turn_index) == ("call-0005", 2) for record in denied_records)
    card_records = [record for record in dropped if DropReason.CARD_NUMBER_PII in record.reasons]
    assert len(card_records) == 2
    assert all(record.call_id in {"call-0001", "call-0005"} for record in card_records)
    combined_record = next(
        record for record in card_records if DropReason.TRAINING_CONSENT_DENIED in record.reasons
    )
    assert set(combined_record.reasons) == {
        DropReason.TRAINING_CONSENT_DENIED,
        DropReason.CARD_NUMBER_PII,
    }


def test_drop_records_are_auditable_and_pii_free(
    clean_event: dict[str, Any],
    with_transcript: Callable[[dict[str, Any], str], dict[str, Any]],
) -> None:
    """Dropped rows carry lineage and reason codes, never matched values."""
    event = with_transcript(clean_event, "Card 4111 1111 1111 1111.")
    _, dropped = filter_eligible_events([event])
    record = dropped[0]
    assert record.event_id == clean_event["event_id"]
    assert record.call_id == clean_event["call_id"]
    assert record.turn_index == clean_event["turn_index"]
    assert record.audio_uri == clean_event["audio"]["uri"]
    assert all(reason.value for reason in record.reasons)
    assert "4111" not in repr(record)
    assert "Card" not in repr(record)


def test_clean_synthetic_fixture_passes_eligibility(
    clean_event: dict[str, Any],
) -> None:
    """A consented, PII-free synthetic event remains eligible."""
    kept, dropped = filter_eligible_events([clean_event])
    assert kept == [clean_event]
    assert dropped == []


def test_fixture_set_has_no_card_drops_and_one_consent_drop(
    fixture_events: list[dict[str, Any]],
) -> None:
    """All 36 synthetic fixtures screen clean except the named consent case."""
    kept, dropped = filter_eligible_events(fixture_events)
    assert len(kept) == 35
    assert len(dropped) == 1
    assert not any(DropReason.CARD_NUMBER_PII in record.reasons for record in dropped)
    assert dropped[0].reasons == (DropReason.TRAINING_CONSENT_DENIED,)
