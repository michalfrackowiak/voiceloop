"""Shared fixtures for dataset builder tests; fixture files are read-only."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from services.datasets import ReviewedItem, load_reviewed_items_from_fixtures

REPO_ROOT = Path(__file__).resolve().parents[3]
TURNS_PATH = REPO_ROOT / "tests" / "fixtures" / "turns.jsonl"


@pytest.fixture
def fixture_rows() -> list[dict[str, Any]]:
    """Return the 36 read-only synthetic label rows."""
    rows = [
        json.loads(line) for line in TURNS_PATH.read_text(encoding="utf-8").splitlines() if line
    ]
    return rows


@pytest.fixture
def fixture_call_ids(fixture_rows: list[dict[str, Any]]) -> tuple[str, ...]:
    """Return the six synthetic call IDs in sorted order."""
    return tuple(sorted({str(row["call_id"]) for row in fixture_rows}))


@pytest.fixture
def reviewed_items() -> tuple[ReviewedItem, ...]:
    """Return reviewed items built from the synthetic label manifest."""
    return load_reviewed_items_from_fixtures(TURNS_PATH)
