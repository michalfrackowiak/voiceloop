"""Load and validate JSON Schemas from the repository-root ``schemas/`` directory."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator, FormatChecker
from jsonschema.exceptions import SchemaError

from .errors import ValidationError

_SCHEMA_DIR = Path(__file__).resolve().parent.parent / "schemas"
_NAME_RE = re.compile(r"[A-Za-z0-9_][A-Za-z0-9_.-]*\Z")


def _schema_file(name: str) -> Path:
    """Resolve a schema name to a file path, rejecting traversal and odd names."""
    if not isinstance(name, str) or not _NAME_RE.fullmatch(name):
        raise ValidationError(f"Invalid schema name: {name!r}")
    filename = name if name.endswith(".json") else f"{name}.schema.json"
    return _SCHEMA_DIR / filename


def schema_path(name: str) -> Path:
    """Return the on-disk path for a named schema; the file need not exist.

    Args:
        name: Schema name with or without the ``.schema.json`` suffix.

    Returns:
        A path under the repository's ``schemas/`` directory.

    Raises:
        ValidationError: If the name is not a simple schema filename.

    """
    return _schema_file(name)


def load_schema(name: str) -> dict[str, Any]:
    """Load a named JSON Schema document from disk.

    Raises:
        ValidationError: If the name is invalid, the JSON is malformed, or the
            document is not a JSON object.
        FileNotFoundError: If no schema file exists for the name.

    """
    path = _schema_file(name)
    try:
        raw = path.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Schema not found: {name} ({path})") from exc
    try:
        document = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValidationError(f"Schema {name} is not valid JSON: {exc}") from exc
    if not isinstance(document, dict):
        raise ValidationError(f"Schema {name} must be a JSON object")
    return document


def validate(instance: Any, schema_name: str) -> None:
    """Validate an instance against a named schema.

    Args:
        instance: Decoded JSON value to check.
        schema_name: Name of the schema in the repository's ``schemas/`` directory.

    Raises:
        ValidationError: If the schema is unusable or the instance is invalid.
        FileNotFoundError: If the named schema does not exist.

    """
    document = load_schema(schema_name)
    try:
        Draft202012Validator.check_schema(document)
    except SchemaError as exc:
        raise ValidationError(f"Schema {schema_name} is invalid: {exc.message}") from exc
    validator = Draft202012Validator(document, format_checker=FormatChecker())
    error = next(iter(validator.iter_errors(instance)), None)
    if error is not None:
        location = ".".join(str(part) for part in error.absolute_path) or "<root>"
        raise ValidationError(
            f"Validation failed against {schema_name} at {location}: {error.message}"
        )
