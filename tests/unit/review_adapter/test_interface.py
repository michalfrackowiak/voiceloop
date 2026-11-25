"""Contract-shape checks for tool-neutral review task types."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from services.review_adapter import (
    ROUTING_LABELS,
    SCHEMA_VERSION,
    LowConfidenceSpan,
    ReviewTask,
    SimpleQueueItem,
    build_review_task,
    build_slot_form,
)
from tests.unit.review_adapter.conftest import CONTRACT


def contract() -> dict[str, Any]:
    """Load the read-only OpenAPI contract."""
    document = yaml.safe_load(Path(CONTRACT).read_text(encoding="utf-8"))
    assert isinstance(document, dict)
    return document


def test_routing_labels_match_contract() -> None:
    """The four local labels are exactly the OpenAPI enum."""
    enum = contract()["components"]["schemas"]["RoutingLabel"]["enum"]
    assert tuple(enum) == ROUTING_LABELS
    assert len(ROUTING_LABELS) == 4


def test_push_payload_has_exactly_contract_fields() -> None:
    """The tool-neutral task emits only ``ReviewItemCreate`` fields."""
    item = SimpleQueueItem(
        event_id="event-1",
        call_id="call-1",
        turn_index=2,
        audio_uri="s3://bucket/turn.wav",
        hypothesis="three nights",
        low_confidence_spans=(LowConfidenceSpan(820, 1400, "three nights", 0.42),),
        no_prefill=True,
    )
    task = build_review_task(item, build_slot_form())
    payload = task.to_push_payload()
    required = set(contract()["components"]["schemas"]["ReviewItemCreate"]["required"])
    assert set(payload) == required
    assert payload["schema_version"] == SCHEMA_VERSION
    assert payload["blind"] is True
    assert payload["low_confidence_spans"] == [
        {"start_ms": 820, "end_ms": 1400, "text": "three nights", "confidence": 0.42}
    ]


def test_task_keeps_lineage_outside_contract_payload() -> None:
    """Adapter-local and original IDs stay on the internal task object."""
    item = SimpleQueueItem("event-9", "call-9", 0, "s3://bucket/0.wav", "hello")
    task: ReviewTask = build_review_task(item, build_slot_form())
    assert task.item_id == "item-event-9"
    assert task.event_id == "event-9"
    assert task.call_id == "call-9"
    assert "item_id" not in task.to_push_payload()
    assert "event_id" not in task.to_push_payload()
    assert task.slot_form.fields
