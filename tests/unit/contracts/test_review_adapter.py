"""Tests for the review adapter OpenAPI contract and example payloads."""

from typing import Any

import jsonschema
import pytest

SPEC_NAME = "review_adapter.openapi.yaml"
EXPECTED_PATHS = {
    "/v1/items": {"post"},
    "/v1/items/{item_id}": {"get"},
    "/v1/items/{item_id}/result": {"get", "post"},
}
EXPECTED_SCHEMAS = {
    "ReviewItem",
    "ReviewResult",
    "RoutingLabel",
    "ReviewItemCreate",
    "ReviewResultCreate",
    "LowConfidenceSpan",
    "Error",
}


def _openapi_schemas(spec: dict[str, Any]) -> dict[str, Any]:
    """Rewrite OpenAPI component refs so jsonschema can resolve them locally."""

    def convert(value: Any) -> Any:
        if isinstance(value, dict):
            converted: dict[str, Any] = {}
            for key, item in value.items():
                prefix = "#/components/schemas/"
                if key == "$ref" and isinstance(item, str) and item.startswith(prefix):
                    converted[key] = "#/$defs/" + item.removeprefix(prefix)
                else:
                    converted[key] = convert(item)
            return converted
        if isinstance(value, list):
            return [convert(item) for item in value]
        return value

    return convert(spec["components"]["schemas"])


def _component_schema(spec: dict[str, Any], name: str) -> dict[str, Any]:
    """Wrap a component schema with a local ref for standalone validation."""
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$defs": _openapi_schemas(spec),
        "$ref": f"#/$defs/{name}",
    }


def test_spec_is_openapi_31_with_required_paths(load_yaml: Any) -> None:
    """The spec parses with PyYAML and exposes push, status and pull endpoints."""
    spec = load_yaml(SPEC_NAME)
    assert spec["openapi"] == "3.1.0"
    assert spec["info"]["version"] == "1.0.0"
    assert "adapter hides" in spec["info"]["description"]
    assert set(EXPECTED_PATHS) <= set(spec["paths"])
    for path, methods in EXPECTED_PATHS.items():
        assert methods <= set(spec["paths"][path])
    assert EXPECTED_SCHEMAS <= set(spec["components"]["schemas"])


def test_push_and_result_fields_and_routing_labels(load_yaml: Any) -> None:
    """Push/result payloads and the routing enum match the task contract."""
    spec = load_yaml(SPEC_NAME)
    schemas = spec["components"]["schemas"]
    assert set(schemas["ReviewItemCreate"]["required"]) == {
        "schema_version",
        "call_id",
        "turn_index",
        "audio_uri",
        "hypothesis",
        "low_confidence_spans",
        "blind",
    }
    assert set(schemas["ReviewResultCreate"]["required"]) == {
        "schema_version",
        "reviewer_id",
        "corrected_transcript",
        "slots",
        "routing",
        "started_at",
        "completed_at",
    }
    assert schemas["RoutingLabel"]["enum"] == [
        "stt_wrong",
        "parser_wrong",
        "both_fine",
        "unusable_audio",
    ]
    assert schemas["ReviewResult"]["properties"]["routing"]["$ref"].endswith("RoutingLabel")
    assert schemas["Error"]["required"] == ["code", "message", "details"]


@pytest.mark.parametrize(
    ("example_file", "component"),
    [
        ("review_adapter.push_item.json", "ReviewItemCreate"),
        ("review_adapter.item.json", "ReviewItem"),
        ("review_adapter.result.json", "ReviewResultCreate"),
        ("review_adapter.error.json", "Error"),
    ],
)
def test_example_files_validate_against_components(
    example_file: str, component: str, load_json: Any, load_yaml: Any, validate: Any
) -> None:
    """Each standalone example validates against its OpenAPI component schema."""
    spec = load_yaml(SPEC_NAME)
    validate(load_json(example_file), _component_schema(spec, component))


def test_inline_examples_match_example_files(load_json: Any, load_yaml: Any, validate: Any) -> None:
    """Embedded OpenAPI examples stay in sync with the standalone files."""
    spec = load_yaml(SPEC_NAME)
    paths = spec["paths"]
    push = paths["/v1/items"]["post"]["requestBody"]["content"]["application/json"]
    push_example = push["examples"]["push_item"]["value"]
    item_example = paths["/v1/items"]["post"]["responses"]["201"]["content"]["application/json"][
        "examples"
    ]["item"]["value"]
    result_post = paths["/v1/items/{item_id}/result"]["post"]["requestBody"]["content"][
        "application/json"
    ]
    result_example = result_post["examples"]["result"]["value"]
    error_example = spec["components"]["responses"]["ValidationError"]["content"][
        "application/json"
    ]["examples"]["error"]["value"]

    assert push_example == load_json("review_adapter.push_item.json")
    assert item_example == load_json("review_adapter.item.json")
    assert result_example == load_json("review_adapter.result.json")
    assert error_example == load_json("review_adapter.error.json")

    validate(push_example, _component_schema(spec, "ReviewItemCreate"))
    validate(item_example, _component_schema(spec, "ReviewItem"))
    validate(result_example, _component_schema(spec, "ReviewResultCreate"))
    with pytest.raises(jsonschema.ValidationError):
        validate({"code": "not_found", "message": "missing"}, _component_schema(spec, "Error"))
