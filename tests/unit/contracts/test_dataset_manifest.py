"""Tests for the dataset manifest schema and examples."""

from copy import deepcopy
from typing import Any

import jsonschema
import pytest

SCHEMA_NAME = "dataset_manifest.schema.json"


def test_schema_declares_hash_splits_and_filter_counts(load_schema: Any) -> None:
    """The manifest pins lineage, call-level splits and audited filter counts."""
    schema = load_schema(SCHEMA_NAME)
    assert schema["$schema"] == "https://json-schema.org/draft/2020-12/schema"
    assert schema["$id"].endswith(SCHEMA_NAME)
    assert schema["title"]
    assert schema["description"]
    assert {
        "schema_version",
        "dataset_id",
        "content_hash",
        "source_call_ids",
        "splits",
        "filter_counts",
        "created_at",
        "examples",
    } <= set(schema["required"])
    assert schema["properties"]["content_hash"]["type"] == "string"
    assert schema["properties"]["created_at"]["format"] == "date-time"
    assert schema["properties"]["examples"]["minimum"] == 0
    assert "synthetic" in schema["properties"]

    splits = schema["properties"]["splits"]
    assert set(splits["required"]) == {"train", "val", "test"}
    assert all(
        splits["properties"][name]["$ref"] == "#/$defs/CallIdList"
        for name in ("train", "val", "test")
    )
    assert schema["$defs"]["CallIdList"]["type"] == "array"

    counts = schema["properties"]["filter_counts"]
    assert set(counts["required"]) == {
        "consent_dropped",
        "pii_dropped",
        "replay_added",
        "golden_excluded",
        "regression_excluded",
        "synthetic_tagged",
    }
    assert all(counts["properties"][name]["$ref"] == "#/$defs/Count" for name in counts["required"])


def test_examples_validate_and_invalid_example_is_rejected(
    load_json: Any, load_schema: Any, validate: Any
) -> None:
    """The complete manifest validates; a missing PII count does not."""
    schema = load_schema(SCHEMA_NAME)
    validate(load_json("dataset_manifest.valid.json"), schema)
    with pytest.raises(jsonschema.ValidationError, match="pii_dropped"):
        validate(load_json("dataset_manifest.invalid.json"), schema)


def test_splits_are_call_id_arrays_and_disjoint(
    load_json: Any, load_schema: Any, validate: Any
) -> None:
    """Split values are call ID arrays; no call appears in two splits."""
    schema = load_schema(SCHEMA_NAME)
    manifest = load_json("dataset_manifest.valid.json")
    assigned = manifest["splits"]["train"] + manifest["splits"]["val"] + manifest["splits"]["test"]
    assert len(assigned) == len(set(assigned))
    assert set(assigned) == set(manifest["source_call_ids"])

    count_style = deepcopy(manifest)
    count_style["splits"]["train"] = 4
    with pytest.raises(jsonschema.ValidationError):
        validate(count_style, schema)
