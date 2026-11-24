"""Fixtures for validating the P0-04 contract artifacts."""

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import jsonschema
import pytest
import yaml


@pytest.fixture
def repo_root() -> Path:
    """Return the repository root."""
    return Path(__file__).resolve().parents[3]


@pytest.fixture
def load_json(repo_root: Path) -> Callable[[str], Any]:
    """Return a loader for example payloads under schemas/examples."""

    def _load(name: str) -> Any:
        with (repo_root / "schemas" / "examples" / name).open(encoding="utf-8") as stream:
            return json.load(stream)

    return _load


@pytest.fixture
def load_schema(repo_root: Path) -> Callable[[str], dict[str, Any]]:
    """Return a loader for JSON Schemas under schemas."""

    def _load(name: str) -> dict[str, Any]:
        with (repo_root / "schemas" / name).open(encoding="utf-8") as stream:
            return json.load(stream)

    return _load


@pytest.fixture
def load_yaml(repo_root: Path) -> Callable[[str], dict[str, Any]]:
    """Return a loader for OpenAPI YAML under schemas."""

    def _load(name: str) -> dict[str, Any]:
        with (repo_root / "schemas" / name).open(encoding="utf-8") as stream:
            return yaml.safe_load(stream)

    return _load


@pytest.fixture
def validate() -> Callable[[Any, dict[str, Any]], None]:
    """Return a Draft 2020-12 validator that also checks date-time formats."""

    def _validate(instance: Any, schema: dict[str, Any]) -> None:
        validator = jsonschema.Draft202012Validator(
            schema, format_checker=jsonschema.FormatChecker()
        )
        validator.validate(instance)

    return _validate
