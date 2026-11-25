"""Shared fixtures for P2-01 scoring tests."""

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
    """Return the read-only fixture labels manifest."""
    return repo_root / "tests" / "fixtures" / "turns.jsonl"


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


@pytest.fixture
def load_event(repo_root: Path) -> Callable[[str | Path], dict[str, Any]]:
    """Load a turn event from a repo-relative or absolute path."""

    def _load(path: str | Path) -> dict[str, Any]:
        candidate = Path(path)
        resolved = candidate if candidate.is_absolute() else repo_root / candidate
        return json.loads(resolved.read_text(encoding="utf-8"))

    return _load
