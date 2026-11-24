"""Tests for the stdlib-only example bot and local ingest sink."""

from __future__ import annotations

import json
from typing import Any

from schemas import validate as validate_schema
from sdk.pipecat import (
    CALL_ID_ATTRIBUTE,
    CONVERSATION_ID_ATTRIBUTE,
    TURN_INDEX_ATTRIBUTE,
    DictSpan,
    validate_turn_event,
)
from sdk.pipecat.example import EXAMPLE_CALL_ID, EXAMPLE_CONVERSATION_ID, main, run_example


def test_example_produces_correct_turn_events() -> None:
    """The scripted bot emits three schema-valid events with native rates."""
    span = DictSpan()

    events = run_example(span=span)

    assert len(events) == 3
    for event in events:
        validate_schema(event, "turn_event.schema.json")
        validate_turn_event(event)
    assert [event["turn_index"] for event in events] == [0, 1, 2]
    assert all(event["call_id"] == EXAMPLE_CALL_ID for event in events)
    assert [event["audio"]["sample_rate_hz"] for event in events] == [16000, 8000, 8000]
    assert events[0]["stt"]["text"] == "Book a deluxe room for two nights."
    assert events[0]["stt"]["words"][0]["conf"] == 0.98
    assert events[0]["parser"]["output"]["room_type"] == "deluxe"
    assert events[0]["consent"] == {"recording": True, "training": True}
    assert events[2]["stt"]["text"] == ""
    assert dict(span.attributes) == {
        CALL_ID_ATTRIBUTE: EXAMPLE_CALL_ID,
        CONVERSATION_ID_ATTRIBUTE: EXAMPLE_CONVERSATION_ID,
        TURN_INDEX_ATTRIBUTE: 2,
    }


def test_example_main_prints_json(capsys: Any) -> None:
    """The CLI entry point prints the local ingest payload as JSON."""
    main()
    printed = json.loads(capsys.readouterr().out)
    assert printed["event_count"] == 3
    assert len(printed["events"]) == 3
