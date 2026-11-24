"""Tests for the trainer interface contract schema and examples."""

from typing import Any

import jsonschema
import pytest

SCHEMA_NAME = "trainer_contract.schema.json"


def test_schema_declares_versioned_input_and_output(load_schema: Any) -> None:
    """The schema pins a version and discriminates request from response."""
    schema = load_schema(SCHEMA_NAME)
    assert schema["$schema"] == "https://json-schema.org/draft/2020-12/schema"
    assert schema["$id"].endswith(SCHEMA_NAME)
    assert schema["title"]
    assert schema["description"]

    request = schema["$defs"]["TrainRequest"]
    response = schema["$defs"]["TrainResponse"]
    assert {"dataset_manifest", "config"} <= set(request["required"])
    assert {"artifact_uri", "metrics", "run_id"} <= set(response["required"])
    assert (
        request["properties"]["dataset_manifest"]["properties"]["content_hash"]["type"] == "string"
    )
    assert request["properties"]["config"]["type"] == "object"
    assert [branch["$ref"] for branch in schema["oneOf"]] == [
        "#/$defs/TrainRequest",
        "#/$defs/TrainResponse",
    ]

    lineage = response["properties"]["lineage"]
    assert {"dataset_hash", "git_sha", "image_digest"} <= set(lineage["required"])
    assert all(prop["type"] == "string" for prop in lineage["properties"].values())


def test_examples_validate(load_json: Any, load_schema: Any, validate: Any) -> None:
    """Both the input and output examples satisfy the contract."""
    schema = load_schema(SCHEMA_NAME)
    for name in ("trainer_contract.input.json", "trainer_contract.output.json"):
        validate(load_json(name), schema)


def test_invalid_example_is_rejected(load_json: Any, load_schema: Any, validate: Any) -> None:
    """An output without run_id must not validate as either branch."""
    schema = load_schema(SCHEMA_NAME)
    with pytest.raises(jsonschema.ValidationError) as exc_info:
        validate(load_json("trainer_contract.invalid.json"), schema)
    messages = [error.message for error in exc_info.value.context]
    assert any("run_id" in message for message in messages)
