"""Shared fixtures for P3-10 redaction tests; committed fixtures are read-only."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from services.datasets.redaction.settings import (
    FixedClock,
    InMemoryAuditSink,
    RedactionSettings,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
TURNS_PATH = REPO_ROOT / "tests" / "fixtures" / "turns.jsonl"


@pytest.fixture
def settings() -> RedactionSettings:
    """Return deterministic settings; no environment or real store required."""
    return RedactionSettings(
        padding_ms=20,
        retention_days=7,
        pseudonym_salt="unit-test-salt",
        output_dir="",
    )


@pytest.fixture
def fixed_clock() -> FixedClock:
    """Return a fixed UTC instant for deterministic audit reports."""
    return FixedClock(datetime(2026, 3, 1, 12, 0, tzinfo=UTC))


@pytest.fixture
def audit() -> InMemoryAuditSink:
    """Return a fresh in-memory audit sink."""
    return InMemoryAuditSink()


@pytest.fixture
def fixture_rows() -> list[dict[str, Any]]:
    """Return all 36 read-only synthetic label rows."""
    return [
        json.loads(line)
        for line in TURNS_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


@pytest.fixture
def turns_path() -> Path:
    """Return the committed read-only labels manifest path."""
    return TURNS_PATH
