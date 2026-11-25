"""Tests for reviewed-item expansion into training examples."""

from __future__ import annotations

from typing import Any

import pytest

from services.datasets import (
    DatasetSettings,
    ReviewedItem,
    build_dialogue_context,
    expand_reviewed_item,
)
from voiceloop_common import ValidationError


def _by_source(items: tuple[ReviewedItem, ...], source_id: str) -> ReviewedItem:
    """Find one reviewed item by its lineage key."""
    return next(item for item in items if item.source_id == source_id)


def test_reviewed_item_exposes_lineage_fields(reviewed_items: tuple[ReviewedItem, ...]) -> None:
    """Source call, turn, hypothesis and audio come from the source event."""
    item = _by_source(reviewed_items, "call-0002:2")
    assert item.call_id == "call-0002"
    assert item.turn_index == 2
    assert item.hypothesis == "Mr. Smythe."
    assert item.corrected_transcript == "The name is Mr. Smith."
    assert item.audio_uri.endswith("call-0002/turn-002.wav")
    assert item.duration_ms == 320
    assert item.synthetic is True


def test_failure_item_expands_to_three_contextual_examples(
    reviewed_items: tuple[ReviewedItem, ...],
) -> None:
    """A corrected failure yields STT, clean parser and noisy parser examples."""
    item = _by_source(reviewed_items, "call-0002:2")
    examples = expand_reviewed_item(item)
    assert [example.kind for example in examples] == [
        "stt",
        "parser_clean",
        "parser_noisy",
    ]
    assert [example.example_id for example in examples] == [
        "call-0002:2:stt",
        "call-0002:2:parser_clean",
        "call-0002:2:parser_noisy",
    ]
    assert examples[0].source_text == item.hypothesis
    assert examples[0].target_text == item.corrected_transcript
    assert examples[1].source_text == item.corrected_transcript
    assert examples[1].slots == item.slots
    assert examples[2].source_text == item.hypothesis
    assert examples[2].slots == item.slots
    for example in examples:
        assert example.call_id == "call-0002"
        assert example.turn_index == 2
        assert example.dialogue_context["expected_slot"] == "guest_name"
        prior = example.dialogue_context["prior_turns"]
        assert [turn["turn_index"] for turn in prior] == [0, 1]
    assert examples[0].to_dict()["kind"] == "stt"


def test_clean_turn_yields_two_examples_and_silence_yields_none(
    reviewed_items: tuple[ReviewedItem, ...],
) -> None:
    """Empty targets and absent slots never produce fabricated examples."""
    clean = _by_source(reviewed_items, "call-0001:1")
    assert [example.kind for example in expand_reviewed_item(clean)] == [
        "stt",
        "parser_clean",
    ]
    silence = _by_source(reviewed_items, "call-0003:1")
    assert expand_reviewed_item(silence) == ()


def test_max_examples_and_context_limit(reviewed_items: tuple[ReviewedItem, ...]) -> None:
    """The expansion cap and prior-turn cap are configurable."""
    item = _by_source(reviewed_items, "call-0002:2")
    assert len(expand_reviewed_item(item, max_examples=1)) == 1
    assert len(expand_reviewed_item(item, max_examples=3)) == 3
    limited = expand_reviewed_item(item, prior_turns_limit=1)
    assert len(limited[0].dialogue_context["prior_turns"]) == 1
    with pytest.raises(ValidationError, match="max_examples"):
        expand_reviewed_item(item, max_examples=4)
    with pytest.raises(ValidationError, match="routing"):
        ReviewedItem(
            event=item.event,
            corrected_transcript=item.corrected_transcript,
            slots=item.slots,
            routing="unknown_label",
        )


def test_build_dialogue_context_handles_missing_reference() -> None:
    """A missing context reference yields an empty prior-turn list."""
    event: dict[str, Any] = {
        "dialogue_context": {
            "reference_date": "2026-11-24",
            "prior_turns_ref": "tests/fixtures/contexts/does-not-exist.jsonl",
        }
    }
    context = build_dialogue_context(
        event, 3, prior_turns_limit=DatasetSettings().prior_turns_limit
    )
    assert context["reference_date"] == "2026-11-24"
    assert context["prior_turns"] == []
