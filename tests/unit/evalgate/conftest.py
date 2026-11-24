"""Shared fixtures for P1-07 evaluation harness tests."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest


@pytest.fixture
def repo_root() -> Path:
    """Return the repository root."""
    return Path(__file__).resolve().parents[3]


@pytest.fixture
def turns_path(repo_root: Path) -> Path:
    """Return the read-only synthetic labels manifest path."""
    return repo_root / "tests" / "fixtures" / "turns.jsonl"


@pytest.fixture
def load_rows(turns_path: Path) -> Callable[[], list[dict[str, Any]]]:
    """Load the labels manifest directly, independent of harness code."""

    def _load() -> list[dict[str, Any]]:
        return [
            json.loads(line)
            for line in turns_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

    return _load
