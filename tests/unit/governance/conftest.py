"""Shared fixtures for P1-09 governance tests; fixtures are read-only."""

from __future__ import annotations

import json
from collections.abc import Callable
from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
EVENT_DIR = REPO_ROOT / "tests" / "fixtures" / "events"


def _load_event(call_id: str, turn_index: int) -> dict[str, Any]:
    """Load one synthetic fixture event by call and turn index."""
    path = EVENT_DIR / call_id / f"turn-{turn_index:03d}.json"
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


@pytest.fixture
def with_transcript() -> Callable[[dict[str, Any], str], dict[str, Any]]:
    """Return a helper that copies an event with a synthetic STT transcript."""

    def _with_transcript(event: dict[str, Any], text: str) -> dict[str, Any]:
        """Return a deep copy of an event with a replaced transcript."""
        modified = deepcopy(event)
        modified["stt"]["text"] = text
        return modified

    return _with_transcript


@pytest.fixture
def fixture_events() -> list[dict[str, Any]]:
    """Return all 36 synthetic fixture events in deterministic order."""
    return [
        json.loads(path.read_text(encoding="utf-8")) for path in sorted(EVENT_DIR.glob("*/*.json"))
    ]


@pytest.fixture
def clean_event() -> dict[str, Any]:
    """Return a synthetic event with consent and a PII-free transcript."""
    return _load_event("call-0001", 0)


@pytest.fixture
def no_training_consent_event() -> dict[str, Any]:
    """Return the synthetic fixture whose training consent is withheld."""
    return _load_event("call-0005", 2)
