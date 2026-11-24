"""Tests for the evaluation report schema and examples."""

from copy import deepcopy
from typing import Any

import jsonschema
import pytest

SCHEMA_NAME = "eval_report.schema.json"
CORE_METRICS = (
    "wer",
    "entity_error",
    "slot_f1",
    "schema_valid_rate",
    "e2e_booking_accuracy",
)


def test_schema_declares_version_metrics_and_intervals(load_schema: Any) -> None:
    """The schema pins all core metrics, their ranges and paired-bootstrap CIs."""
    schema = load_schema(SCHEMA_NAME)
    assert schema["$schema"] == "https://json-schema.org/draft/2020-12/schema"
    assert schema["$id"].endswith(SCHEMA_NAME)
    assert schema["title"]
    assert schema["description"]
    assert {
        "schema_version",
        "artifact_uri",
        "generated_at",
        "eval_sets",
        "metrics",
        "confidence_intervals",
        "slice_metrics",
    } <= set(schema["required"])

    assert schema["properties"]["metrics"]["$ref"] == "#/$defs/Metrics"
    metrics = schema["$defs"]["Metrics"]
    assert set(CORE_METRICS) <= set(metrics["required"])
    assert metrics["properties"]["wer"]["minimum"] == 0
    assert "maximum" not in metrics["properties"]["wer"]
    for name in CORE_METRICS[1:]:
        assert metrics["properties"][name]["maximum"] == 1
    assert "primary" in metrics["properties"]["e2e_booking_accuracy"]["description"].lower()

    assert schema["$defs"]["EvalSet"]["enum"] == [
        "golden",
        "regression",
        "general_capability",
    ]
    assert set(CORE_METRICS) <= set(schema["$defs"]["ConfidenceIntervals"]["required"])
    interval = schema["$defs"]["ConfidenceInterval"]
    assert {"lower", "upper", "level"} <= set(interval["required"])
    for name in ("decoding_settings", "shadow_evidence", "champion_report_uri", "gate_decision"):
        assert name in schema["properties"]
    assert "gate_decision" not in schema["required"]


def test_valid_example(load_json: Any, load_schema: Any, validate: Any) -> None:
    """The full report example validates, including date-time formats."""
    validate(load_json("eval_report.valid.json"), load_schema(SCHEMA_NAME))


def test_invalid_example_missing_ci_bound(load_json: Any, load_schema: Any, validate: Any) -> None:
    """A confidence interval missing its upper bound is rejected."""
    with pytest.raises(jsonschema.ValidationError, match="upper"):
        validate(load_json("eval_report.invalid.json"), load_schema(SCHEMA_NAME))


def test_wer_may_exceed_one_but_bounded_metrics_cannot(
    load_json: Any, load_schema: Any, validate: Any
) -> None:
    """WER allows insertions; the other core metrics stay in 0..1."""
    schema = load_schema(SCHEMA_NAME)
    report = deepcopy(load_json("eval_report.valid.json"))
    report["metrics"]["wer"] = 1.25
    validate(report, schema)

    report["metrics"]["schema_valid_rate"] = 1.5
    with pytest.raises(jsonschema.ValidationError):
        validate(report, schema)
