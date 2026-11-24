"""Tests for the repository-root JSON Schema loader and validator."""

import json
from pathlib import Path

import pytest

from voiceloop_common import ValidationError
from voiceloop_common import schemas as schema_helpers

SIMPLE_SCHEMA = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "type": "object",
    "required": ["event_id"],
    "properties": {"event_id": {"type": "string"}},
}


@pytest.fixture
def schema_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point the loader at an isolated temporary schemas directory."""
    monkeypatch.setattr(schema_helpers, "_SCHEMA_DIR", tmp_path)
    return tmp_path


def write_schema(directory: Path, name: str, document: object) -> Path:
    """Write a JSON document into the temporary schemas directory."""
    path = directory / f"{name}.schema.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    return path


def test_schema_path_defaults_to_repo_root_schemas() -> None:
    path = schema_helpers.schema_path("turn_event")

    assert path.name == "turn_event.schema.json"
    assert path.parent.name == "schemas"
    assert path.parent.parent == Path(__file__).resolve().parents[3]


def test_schema_path_accepts_filename_with_suffix() -> None:
    expected = Path(__file__).resolve().parents[3] / "schemas" / "turn_event.schema.json"

    assert schema_helpers.schema_path("turn_event.schema.json") == expected


@pytest.mark.parametrize("name", ["../secret", "/absolute", "sub/name", "", "."])
def test_schema_path_rejects_unsafe_names(name: str) -> None:
    with pytest.raises(ValidationError, match="Invalid schema name"):
        schema_helpers.schema_path(name)


def test_load_schema_reads_named_document(schema_dir: Path) -> None:
    write_schema(schema_dir, "demo", SIMPLE_SCHEMA)

    assert schema_helpers.load_schema("demo") == SIMPLE_SCHEMA
    assert schema_helpers.load_schema("demo.schema.json") == SIMPLE_SCHEMA


def test_load_schema_missing_file(schema_dir: Path) -> None:
    with pytest.raises(FileNotFoundError, match="Schema not found"):
        schema_helpers.load_schema("missing")


def test_load_schema_invalid_json(schema_dir: Path) -> None:
    (schema_dir / "broken.schema.json").write_text("not json", encoding="utf-8")

    with pytest.raises(ValidationError, match="not valid JSON"):
        schema_helpers.load_schema("broken")


def test_load_schema_requires_json_object(schema_dir: Path) -> None:
    (schema_dir / "array.schema.json").write_text("[]", encoding="utf-8")

    with pytest.raises(ValidationError, match="must be a JSON object"):
        schema_helpers.load_schema("array")


def test_validate_accepts_valid_instance(schema_dir: Path) -> None:
    write_schema(schema_dir, "demo", SIMPLE_SCHEMA)

    schema_helpers.validate({"event_id": "turn-1"}, "demo")


def test_validate_reports_invalid_instance(schema_dir: Path) -> None:
    write_schema(schema_dir, "demo", SIMPLE_SCHEMA)

    with pytest.raises(ValidationError, match="event_id"):
        schema_helpers.validate({"turn_index": 0}, "demo")


def test_validate_reports_invalid_schema(schema_dir: Path) -> None:
    write_schema(schema_dir, "demo", {"type": "not-a-json-schema-type"})

    with pytest.raises(ValidationError, match="Schema demo is invalid"):
        schema_helpers.validate({}, "demo")


def test_validate_propagates_missing_schema(schema_dir: Path) -> None:
    with pytest.raises(FileNotFoundError, match="Schema not found"):
        schema_helpers.validate({}, "absent")
