"""Verification that dataset builds call the P1-09 consent/PII filters."""

from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import replace
from typing import Any

from services.datasets import (
    ReviewedItem,
    build_dataset,
    build_dataset_from_fixtures,
    dropped_as_dicts,
)
from services.datasets.filters import DropReason

CARD = "4111 1111 1111 1111"


def _card_item(item: ReviewedItem, text: str) -> ReviewedItem:
    """Return a reviewed item whose source STT hypothesis contains a card."""
    event: dict[str, Any] = deepcopy(dict(item.event))
    event["stt"]["text"] = text
    return replace(item, event=event, corrected_transcript=text)


def _by_source(items: tuple[ReviewedItem, ...], source_id: str) -> ReviewedItem:
    """Find one reviewed item by its lineage key."""
    return next(item for item in items if item.source_id == source_id)


def test_fixture_non_consented_turn_is_dropped_with_reason() -> None:
    """The named fixture with training consent withheld never enters a dataset."""
    build = build_dataset_from_fixtures()
    assert build.filter_counts.consent_dropped == 1
    dropped = build.dropped
    assert len(dropped) == 1
    assert dropped[0].call_id == "call-0005"
    assert dropped[0].turn_index == 2
    assert DropReason.TRAINING_CONSENT_DENIED in dropped[0].reasons
    assert all(
        example.call_id != "call-0005" or example.turn_index != 2 for example in build.examples
    )


def test_card_in_hypothesis_is_a_hard_drop(
    reviewed_items: tuple[ReviewedItem, ...],
) -> None:
    """A card number in the source transcript drops the row with a PII reason."""
    clean = _by_source(reviewed_items, "call-0001:0")
    build = build_dataset([clean, _card_item(clean, CARD)])
    assert build.filter_counts.pii_dropped == 1
    assert build.filter_counts.consent_dropped == 0
    assert DropReason.CARD_NUMBER_PII in build.dropped[0].reasons
    assert "call-0001:0:stt" in {example.example_id for example in build.examples}
    assert not any("4111" in example.source_text for example in build.examples)
    assert "4111" not in json.dumps(dropped_as_dicts(build.dropped))


def test_card_in_corrected_transcript_is_screened_too(
    reviewed_items: tuple[ReviewedItem, ...],
) -> None:
    """The builder screens reviewed transcripts that the event filter cannot see."""
    clean = _by_source(reviewed_items, "call-0001:0")
    card_corrected = replace(clean, corrected_transcript=f"pay with {CARD}")
    build = build_dataset([clean, card_corrected])
    assert build.filter_counts.pii_dropped == 1
    assert DropReason.CARD_NUMBER_PII in build.dropped[0].reasons
    assert len(build.dropped[0].pii_findings) == 1


def test_combined_violation_records_both_reasons(
    reviewed_items: tuple[ReviewedItem, ...],
) -> None:
    """Non-consent and card PII are both audited on the same dropped row."""
    non_consented = _by_source(reviewed_items, "call-0005:2")
    build = build_dataset([_card_item(non_consented, CARD)])
    assert build.filter_counts.consent_dropped == 1
    assert build.filter_counts.pii_dropped == 1
    reasons = set(build.dropped[0].reasons)
    assert DropReason.TRAINING_CONSENT_DENIED in reasons
    assert DropReason.CARD_NUMBER_PII in reasons
