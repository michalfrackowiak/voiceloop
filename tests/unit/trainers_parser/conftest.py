"""Shared fakes for parser-trainer tests (all synthetic, no real guest data)."""

from __future__ import annotations

from typing import Any

import pytest

from services.trainers.core import TrainerSettings
from services.trainers.parser import (
    FakeAdapterStore,
    ParserExample,
    ParserSettings,
)

DATASET_HASH = "d" * 64


def make_slots(*, guests: int = 2, room_type: str = "standard") -> dict[str, Any]:
    """Return a schema-valid synthetic booking target."""
    return {
        "check_in": "2026-12-12",
        "check_out": "2026-12-14",
        "guests": guests,
        "room_type": room_type,
        "guest_name": "Ada Fixture",
        "contact": {"email": "ada.fixture@example.com"},
    }


def make_example(
    example_id: str,
    kind: str,
    source_text: str,
    *,
    call_id: str = "synthetic-call-a",
) -> ParserExample:
    """Build one validated synthetic parser example."""
    return ParserExample.from_mapping(
        {
            "example_id": example_id,
            "kind": kind,
            "source_text": source_text,
            "slots": make_slots(),
            "call_id": call_id,
            "synthetic": True,
        }
    )


@pytest.fixture
def core_settings() -> TrainerSettings:
    """Return deterministic job settings with real-shaped lineage values."""
    return TrainerSettings(
        model="fixture-parser-base",
        seed=7,
        artifact_uri_prefix="fake://parser-adapters",
        image_digest="registry.example/parser@sha256:" + "a" * 64,
        git_sha="b" * 40,
    )


@pytest.fixture
def parser_settings() -> ParserSettings:
    """Return default stand-in parser settings."""
    return ParserSettings()


@pytest.fixture
def manifest() -> dict[str, Any]:
    """Return a minimal contract-valid dataset manifest."""
    return {"dataset_id": "synthetic-parser-fixture", "content_hash": DATASET_HASH}


@pytest.fixture
def examples() -> tuple[ParserExample, ...]:
    """Return two gold and two noisy synthetic examples."""
    return (
        make_example("ex-gold-1", "gold", "Check in on the twelfth for Ada Fixture."),
        make_example("ex-gold-2", "gold", "Make that a deluxe room."),
        make_example("ex-noisy-1", "noisy", "Check in on the twelf for Ada Ficture."),
        make_example("ex-noisy-2", "noisy", "Make that a delux room."),
    )


@pytest.fixture
def store() -> FakeAdapterStore:
    """Return an empty in-memory artifact store."""
    return FakeAdapterStore()
