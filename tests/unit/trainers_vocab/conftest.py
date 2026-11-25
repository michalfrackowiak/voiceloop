"""Shared read-only fixture helpers for P2-08 tests."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import pytest

from services.evalgate.harness import load_event, load_fixture_rows, predict
from services.evalgate.metrics import Prediction
from services.trainers.vocab import AliasTable, VocabRuleSet, VocabSettings, derive_alias_table

REPO_ROOT = Path(__file__).resolve().parents[3]


@pytest.fixture()
def settings() -> VocabSettings:
    """Return clean P2-08 settings (no environment leakage between tests)."""
    return VocabSettings()


@pytest.fixture()
def fixture_rows() -> list[dict[str, Any]]:
    """Load the read-only labelled synthetic fixtures via the P1-07 harness."""
    return load_fixture_rows("tests/fixtures/turns.jsonl")


@pytest.fixture()
def alias_table() -> AliasTable:
    """Return aliases derived only from read-only fixtures."""
    return derive_alias_table("tests/fixtures/turns.jsonl")


@pytest.fixture()
def ruleset(settings: VocabSettings) -> VocabRuleSet:
    """Return the full offline rule pipeline."""
    return VocabRuleSet(settings)


@pytest.fixture()
def row_by_case(fixture_rows: list[dict[str, Any]]) -> Callable[[str], dict[str, Any]]:
    """Look up a labelled fixture by its named case ID."""

    def lookup(case_id: str) -> dict[str, Any]:
        for row in fixture_rows:
            if row.get("case_id") == case_id:
                return row
        raise KeyError(case_id)

    return lookup


@pytest.fixture()
def identity_prediction() -> Callable[[Mapping[str, Any]], tuple[Prediction, Mapping[str, Any]]]:
    """Return a helper producing the P1-07 identity-fake prediction for a row."""

    def build(row: Mapping[str, Any]) -> tuple[Prediction, Mapping[str, Any]]:
        event = load_event(str(row["event_path"]))
        return predict("identity", row, event), event

    return build


@pytest.fixture()
def event_json() -> Callable[[str], dict[str, Any]]:
    """Read one fixture event JSON file directly for metadata assertions."""

    def read(relative_path: str) -> dict[str, Any]:
        decoded = json.loads((REPO_ROOT / relative_path).read_text(encoding="utf-8"))
        assert isinstance(decoded, dict)
        return decoded

    return read
