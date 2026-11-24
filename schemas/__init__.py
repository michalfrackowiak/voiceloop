"""Self-contained helpers for loading and validating VoiceLoop JSON Schemas."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator, FormatChecker
from jsonschema.exceptions import ValidationError

REPO_ROOT: Path = Path(__file__).resolve().parent.parent
SCHEMA_DIR: Path = REPO_ROOT / "schemas"

__all__ = ["REPO_ROOT", "SCHEMA_DIR", "load_schema", "schema_path", "validate"]


def schema_path(name: str) -> Path:
    """Resolve a schema file name relative to the repository's schemas directory.

    Args:
        name: Relative path such as ``turn_event.schema.json`` or
            ``examples/turn_event.valid.minimal.json``.

    Returns:
        Absolute path to the schema file.

    Raises:
        ValueError: If the name is empty, absolute, or escapes the schemas directory.

    """
    if not isinstance(name, str) or not name.strip():
        raise ValueError(f"Schema name must be a non-empty relative path: {name!r}")
    candidate = Path(name)
    if candidate.is_absolute() or ".." in candidate.parts:
        raise ValueError(f"Schema name must stay inside schemas/: {name!r}")
    resolved = (SCHEMA_DIR / candidate).resolve()
    if not resolved.is_relative_to(SCHEMA_DIR.resolve()):
        raise ValueError(f"Schema name must stay inside schemas/: {name!r}")
    return resolved


def load_schema(name: str) -> dict[str, Any]:
    """Load a JSON Schema from the schemas directory.

    Args:
        name: Relative path resolved by :func:`schema_path`.

    Returns:
        The parsed schema as a dictionary.

    Raises:
        ValueError: If the name is unsafe or the file does not contain a JSON object.
        FileNotFoundError: If the file is missing or unreadable.

    """
    path = schema_path(name)
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise FileNotFoundError(f"Schema file not found or unreadable: {path}") from exc
    try:
        schema = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in schema {path}: {exc}") from exc
    if not isinstance(schema, dict):
        raise ValueError(f"Schema {path} must contain a JSON object")
    return schema


def validate(instance: Any, schema_name: str) -> None:
    """Validate an instance against a named schema, including format checks.

    Args:
        instance: Decoded JSON value to validate.
        schema_name: Relative schema path resolved by :func:`schema_path`.

    Raises:
        jsonschema.ValidationError: If the instance does not conform; the message
            includes the schema name and the JSON path of the first error.

    """
    validator = Draft202012Validator(load_schema(schema_name), format_checker=FormatChecker())
    errors = sorted(
        validator.iter_errors(instance),
        key=lambda error: [str(part) for part in error.path],
    )
    if not errors:
        return
    error = errors[0]
    raise ValidationError(
        f"{schema_name}: {error.message} at {error.json_path}",
        validator=error.validator,
        validator_value=error.validator_value,
        instance=error.instance,
        schema=error.schema,
        schema_path=error.schema_path,
        path=error.path,
        cause=error,
        context=error.context,
    )
