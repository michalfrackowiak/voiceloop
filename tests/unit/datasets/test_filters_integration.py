"""Verification that dataset builds call the P1-09 consent/PII filters."""

from __future__ import annotations

import json
from collections.abc import Sequence
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from services.datasets import (
    DatasetExample,
    DatasetSettings,
    ReviewedItem,
    build_dataset,
    build_dataset_from_fixtures,
    dropped_as_dicts,
    export_all,
)
from services.datasets.filters import DropReason

CARD = "4111 1111 1111 1111"
CARD_DIGITS = "4111111111111111"


def _card_item(item: ReviewedItem, text: str) -> ReviewedItem:
    """Return a reviewed item whose source STT hypothesis contains a card."""
    event: dict[str, Any] = deepcopy(dict(item.event))
    event["stt"]["text"] = text
    return replace(item, event=event, corrected_transcript=text)


def _by_source(items: tuple[ReviewedItem, ...], source_id: str) -> ReviewedItem:
    """Find one reviewed item by its lineage key."""
    return next(item for item in items if item.source_id == source_id)


def _assert_exports_free_of_card(examples: Sequence[DatasetExample], tmp_path: Path) -> None:
    """Assert no card text survives in any of the three plain-text exports."""
    result = export_all(examples, tmp_path)
    assert result.counts["hf"] > 0
    for path in (result.nemo_path, result.hf_path, result.sft_path):
        payload = path.read_text(encoding="utf-8")
        assert CARD not in payload
        assert CARD_DIGITS not in payload


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


@pytest.mark.parametrize("card_slot", [CARD, {"note": CARD}], ids=["flat", "nested"])
def test_card_in_reviewed_slots_is_hard_dropped_and_never_exported(
    reviewed_items: tuple[ReviewedItem, ...],
    tmp_path: Path,
    card_slot: Any,
) -> None:
    """A card anywhere in reviewed slots removes every example from that item."""
    clean = _by_source(reviewed_items, "call-0001:0")
    source = _by_source(reviewed_items, "call-0003:2")
    card_item = replace(source, slots={**source.slots, "contact": card_slot})

    build = build_dataset([clean, card_item])

    assert not any(example.call_id == "call-0003" for example in build.examples)
    assert build.filter_counts.pii_dropped == 1
    assert any(
        drop.call_id == "call-0003" and DropReason.CARD_NUMBER_PII in drop.reasons
        for drop in build.dropped
    )
    _assert_exports_free_of_card(build.examples, tmp_path)


def test_card_in_dialogue_context_is_hard_dropped_and_never_exported(
    reviewed_items: tuple[ReviewedItem, ...],
    tmp_path: Path,
) -> None:
    """A card in a prior-turn context reference hard-drops the reviewed item."""
    clean = _by_source(reviewed_items, "call-0001:0")
    source = _by_source(reviewed_items, "call-0002:2")
    context_path = tmp_path / "prior-turns.jsonl"
    context_path.write_text(
        json.dumps({"turn_index": 0, "hypothesis": f"pay with {CARD}"})
        + "\n"
        + json.dumps({"turn_index": 1, "hypothesis": "A regular turn."})
        + "\n",
        encoding="utf-8",
    )
    event = deepcopy(dict(source.event))
    event["dialogue_context"] = dict(event["dialogue_context"])
    event["dialogue_context"]["prior_turns_ref"] = str(context_path)
    context_item = replace(source, event=event)

    build = build_dataset([clean, context_item])

    assert not any(example.call_id == "call-0002" for example in build.examples)
    assert build.filter_counts.pii_dropped == 1
    assert any(
        drop.call_id == "call-0002" and DropReason.CARD_NUMBER_PII in drop.reasons
        for drop in build.dropped
    )
    _assert_exports_free_of_card(build.examples, tmp_path)


def test_card_in_replay_item_is_hard_dropped(
    reviewed_items: tuple[ReviewedItem, ...],
    tmp_path: Path,
) -> None:
    """Replay candidates receive the same recursive card screening."""
    primary = _by_source(reviewed_items, "call-0002:2")
    replay_source = _by_source(reviewed_items, "call-0001:0")
    replay_card = replace(replay_source, slots={**replay_source.slots, "contact": CARD})
    settings = DatasetSettings(replay_size=1, replay_seed=7)

    build = build_dataset([primary], settings=settings, replay_pool=[replay_card])

    assert build.filter_counts.pii_dropped == 1
    assert build.filter_counts.replay_added == 0
    assert not any(example.replay for example in build.examples)
    assert not any(example.call_id == "call-0001" for example in build.examples)
    _assert_exports_free_of_card(build.examples, tmp_path)
