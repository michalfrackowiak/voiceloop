import json
from pathlib import Path
from typing import Any

import pytest
from jsonschema.exceptions import ValidationError

from schemas import REPO_ROOT, load_schema, schema_path, validate

EXAMPLES = REPO_ROOT / "schemas" / "examples"


def _example(name: str) -> dict[str, Any]:
    return json.loads((EXAMPLES / name).read_text(encoding="utf-8"))


@pytest.mark.parametrize("name", ["turn_event.schema.json", "booking.schema.json"])
def test_load_schema_metadata(name: str) -> None:
    schema = load_schema(name)
    assert schema["$schema"] == "https://json-schema.org/draft/2020-12/schema"
    assert schema["schema_version"] == "1.0.0"
    assert schema["$id"].startswith("https://voiceloop.example/schemas/")


def test_schema_path_is_repo_relative() -> None:
    assert schema_path("booking.schema.json") == REPO_ROOT / "schemas" / "booking.schema.json"


@pytest.mark.parametrize("name", ["", "../pyproject.toml", "a/../../b.json", "/etc/passwd"])
def test_schema_path_rejects_unsafe_names(name: str) -> None:
    with pytest.raises(ValueError, match="Schema name"):
        schema_path(name)


def test_validate_enforces_uuid_format() -> None:
    event = _example("turn_event.valid.minimal.json")
    event["event_id"] = "not-a-uuid"
    with pytest.raises(ValidationError, match="uuid"):
        validate(event, "turn_event.schema.json")


def test_validate_error_reports_json_path() -> None:
    event = _example("turn_event.valid.minimal.json")
    event["audio"]["uri"] = ""
    with pytest.raises(ValidationError, match=r"\$\.audio\.uri"):
        validate(event, "turn_event.schema.json")


def test_load_schema_missing_file() -> None:
    with pytest.raises(FileNotFoundError, match="Schema file not found"):
        load_schema("does_not_exist.schema.json")


def test_examples_directory_is_repo_relative() -> None:
    assert isinstance(EXAMPLES, Path)
    assert EXAMPLES.name == "examples"
