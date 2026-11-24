"""Shared helpers for SDK unit tests."""

from collections.abc import Callable
from typing import Any

import pytest

EventFactory = Callable[..., dict[str, Any]]


@pytest.fixture
def make_event() -> EventFactory:
    """Return a factory for deterministic schema-valid turn events."""

    def factory(
        number: int = 1,
        *,
        call_id: str = "call-0001",
        turn_index: int = 0,
    ) -> dict[str, Any]:
        event_id = f"{number:08x}-0000-4000-8000-{number:012x}"
        return {
            "event_id": event_id,
            "call_id": call_id,
            "turn_index": turn_index,
            "audio": {"uri": f"tests/fixtures/audio/{call_id}/turn-{turn_index:03d}.wav"},
            "stt": {"text": "A synthetic utterance."},
        }

    return factory
