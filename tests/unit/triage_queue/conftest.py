"""Shared read-only fixture loaders and deterministic fakes for P2-02 tests."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from services.triage.queue import (
    FixtureTurnSource,
    InMemoryScoreSource,
    TriageQueueSettings,
    TurnRecord,
)


@pytest.fixture
def repo_root() -> Path:
    """Return the repository root."""
    return Path(__file__).resolve().parents[3]


@pytest.fixture
def turns_path(repo_root: Path) -> Path:
    """Return the read-only fixture labels manifest."""
    return repo_root / "tests" / "fixtures" / "turns.jsonl"


@pytest.fixture
def fixture_records() -> tuple[TurnRecord, ...]:
    """Load all 36 read-only fixture events through the public source."""
    return tuple(FixtureTurnSource().iter_turns())


def _fake_score(event: Mapping[str, Any]) -> float:
    """Derive a P2-01-like score from event metadata only (test fake).

    This deliberately reads no fixture label and does not import the in-flight
    scoring package; it stands in for the injected P2-01 score seam.
    """
    stt = event.get("stt") if isinstance(event.get("stt"), Mapping) else {}
    assert isinstance(stt, Mapping)
    words = stt.get("words", ())
    scores: list[float] = []
    if isinstance(words, (list, tuple)):
        for word in words:
            if isinstance(word, Mapping) and isinstance(word.get("conf"), (int, float)):
                scores.append(float(word["conf"]))
    decoder = stt.get("decoder_stats") if isinstance(stt.get("decoder_stats"), Mapping) else {}
    assert isinstance(decoder, Mapping)
    parser = event.get("parser") if isinstance(event.get("parser"), Mapping) else {}
    assert isinstance(parser, Mapping)

    score = 0.0
    if scores and min(scores) < 0.60:
        score = max(score, 0.8)
    if float(decoder.get("no_speech_prob", 0.0)) > 0.50:
        score = max(score, 0.9)
    if parser.get("schema_valid") is False and parser.get("output"):
        score = max(score, 0.7)
    if not str(stt.get("text", "")).strip():
        score = max(score, 0.6)
    return score


@pytest.fixture
def fixture_scores(fixture_records: tuple[TurnRecord, ...]) -> InMemoryScoreSource:
    """Fake P2-01 scores derived only from fixture event signals."""
    return InMemoryScoreSource(
        {record.event_id: _fake_score(record.event) for record in fixture_records}
    )


@pytest.fixture
def queue_settings() -> TriageQueueSettings:
    """Small, deterministic settings suitable for the 36-turn fixture pool."""
    return TriageQueueSettings(
        random_slice_target=2,
        random_seed=7,
        failure_threshold=0.0,
        cluster_similarity_threshold=0.82,
        daily_cap=8,
        interleave_every=3,
        blind_share=0.5,
    )


@dataclass(frozen=True)
class FakeExplanation:
    """A P2-01 ``ScoreExplanation``-shaped object with no scoring dependency."""

    event_id: str
    score: float
    signal_names: tuple[str, ...] = ()


class FakeExplanationStore:
    """Fake store exposing the ``all()`` seam used by ExplanationScoreSource."""

    def __init__(self, explanations: tuple[FakeExplanation, ...]) -> None:
        self._items = explanations

    def all(self) -> tuple[FakeExplanation, ...]:
        """Return all fake explanations."""
        return self._items


@pytest.fixture
def make_event() -> Callable[..., dict[str, Any]]:
    """Return a factory for small, clearly synthetic turn events."""

    def _make(
        event_id: str,
        hypothesis: str = "book a room for three nights",
        *,
        call_id: str | None = None,
        turn_index: int = 0,
        recording_consent: bool = True,
        training_consent: bool = True,
        conf: float = 0.95,
        input_text: str | None = None,
    ) -> dict[str, Any]:
        event: dict[str, Any] = {
            "event_id": event_id,
            "call_id": call_id or f"call-{event_id}",
            "turn_index": turn_index,
            "audio": {"uri": f"synthetic://audio/{event_id}.wav", "duration_ms": 200},
            "consent": {"recording": recording_consent, "training": training_consent},
            "stt": {
                "text": hypothesis,
                "words": [
                    {"w": "book", "start_ms": 0, "end_ms": 100, "conf": conf},
                    {"w": "room", "start_ms": 100, "end_ms": 200, "conf": conf},
                ],
            },
            "parser": {"input_text": input_text if input_text is not None else hypothesis},
        }
        return event

    return _make


@pytest.fixture
def load_labels(turns_path: Path) -> Callable[[], list[dict[str, Any]]]:
    """Load the read-only labels manifest directly."""

    def _load() -> list[dict[str, Any]]:
        return [
            json.loads(line)
            for line in turns_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

    return _load
