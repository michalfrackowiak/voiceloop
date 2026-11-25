"""Shared fixtures for the sealed golden-set unit tests."""

from __future__ import annotations

from collections.abc import Iterator
from datetime import UTC, datetime

import pytest

from services.datasets.golden import (
    LabelRecord,
    SourceTurn,
    load_source_turns,
    synthetic_labels_from_turns,
)
from voiceloop_common import frozen_clock, use_clock

FROZEN_TIME = datetime(2026, 11, 25, 0, 0, 0, tzinfo=UTC)


@pytest.fixture
def fixture_turns() -> tuple[SourceTurn, ...]:
    """Return the read-only fixture pool as failure-blind source turns."""
    return load_source_turns()


@pytest.fixture
def frozen() -> Iterator[None]:
    """Freeze voiceloop_common.utcnow() for deterministic hashes and seals."""
    with use_clock(frozen_clock(FROZEN_TIME)):
        yield


@pytest.fixture
def small_labels(fixture_turns: tuple[SourceTurn, ...]) -> tuple[LabelRecord, ...]:
    """Return two labelled calls with full second-review coverage."""
    chosen = tuple(turn for turn in fixture_turns if turn.turn_index == 0)[:2]
    return synthetic_labels_from_turns(
        chosen, primary_reviewer_id="reviewer-a", second_reviewer_id="reviewer-b"
    )
