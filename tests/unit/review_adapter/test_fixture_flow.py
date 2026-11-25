"""End-to-end fixture rehearsal: two stages, fake store, contract-shaped result."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from services.review_adapter import (
    InMemoryReviewStore,
    InMemoryReviewTool,
    ReviewCoordinator,
    SequenceQueueSource,
    build_review_tool,
)
from tests.unit.review_adapter.conftest import (
    CLOCK,
    FIXTURE_REVIEWER,
    queue_item,
    scripted_review,
)


def test_fixture_item_completes_both_stages_and_lands_in_store() -> None:
    """A P2-02 item from a synthetic turn is reviewed and persisted once."""
    item = queue_item("misheard_surname")
    script = scripted_review("misheard_surname")
    store = InMemoryReviewStore()
    tool = InMemoryReviewTool({item.event_id: script}, clock=CLOCK)
    coordinator = ReviewCoordinator(tool, store)

    item_id = coordinator.push(item)
    transcript = coordinator.stage1(item_id)
    stored = coordinator.stage2(item_id)

    assert transcript == script.corrected_transcript
    assert stored.item_id == item_id
    assert stored.event_id == item.event_id
    assert stored.call_id == item.call_id
    assert stored.turn_index == item.turn_index
    assert stored.audio_uri == item.audio_uri
    assert stored.corrected_transcript == script.corrected_transcript
    assert dict(stored.slots) == script.slots
    assert stored.routing == "stt_wrong"
    assert stored.reviewer_id == FIXTURE_REVIEWER
    assert stored.started_at.isoformat() == "2026-11-25T10:00:00+00:00"
    assert stored.completed_at.isoformat() == "2026-11-25T10:01:12+00:00"
    assert "fake_reviewer_stand_in" in stored.warnings
    assert stored.tool_name == "fake_in_memory"
    assert store.get(item_id) is stored
    assert store.all() == (stored,)
    api = stored.to_api_result()
    assert api["item_id"] == item_id
    assert api["slots"] == script.slots
    assert "warnings" not in api


def test_payload_conforms_to_openapi_result_shape() -> None:
    """The pulled result has exactly the contract's required properties."""
    document: Any = yaml.safe_load(Path("schemas/review_adapter.openapi.yaml").read_text())
    required = set(document["components"]["schemas"]["ReviewResult"]["required"])
    item = queue_item("date_confusion")
    tool = build_review_tool(
        reviews={item.event_id: scripted_review("date_confusion")}, clock=CLOCK
    )
    stored = ReviewCoordinator(tool, InMemoryReviewStore()).run_item(item)
    assert set(stored.to_api_result()) == required
    assert stored.routing == "parser_wrong"


def test_multiple_items_run_through_one_source() -> None:
    """A queue source of P2-02 items runs both stages per item."""
    items = [queue_item("misheard_surname"), queue_item("date_confusion")]
    scripts = {
        item.event_id: scripted_review(case)
        for item, case in zip(items, ["misheard_surname", "date_confusion"], strict=True)
    }
    store = InMemoryReviewStore()
    coordinator = ReviewCoordinator(InMemoryReviewTool(scripts, clock=CLOCK), store)
    results = coordinator.run_all(SequenceQueueSource(items))
    assert len(results) == len(items) == len(store.all()) == 2
    assert {result.routing for result in results} == {"stt_wrong", "parser_wrong"}
