"""Internal comparison artifact and embedded report contract tests."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from services.evalgate.benchmarks.compare import compare_candidates
from services.evalgate.benchmarks.report import (
    COMPARISON_ARTIFACT_TYPE,
    COMPARISON_SCHEMA_VERSION,
    build_comparison_artifact,
    validate_comparison_artifact,
    write_comparison_artifact,
)
from services.evalgate.benchmarks.settings import BenchmarkSettings
from services.evalgate.metrics import CORE_METRIC_NAMES
from voiceloop_common import ValidationError, validate


def test_artifact_has_valid_embedded_reports(
    benchmark_settings: BenchmarkSettings, tmp_path: Path
) -> None:
    """The outer shape is internal; each nested report is eval_report 1.0.0."""
    data = compare_candidates(benchmark_settings)
    artifact = build_comparison_artifact(data)
    assert artifact["artifact_type"] == COMPARISON_ARTIFACT_TYPE
    assert artifact["comparison_schema_version"] == COMPARISON_SCHEMA_VERSION
    assert artifact["status"].startswith("SYNTHETIC/FIXTURE STAND-IN")
    assert artifact["generated_at"].endswith("Z")
    assert set(artifact["sets"]) == {"golden_standin", "general_capability"}

    for set_payload in artifact["sets"].values():
        assert set_payload["eval_set_label"] == "general_capability"
        assert len(set_payload["candidate_reports"]) == 2
        assert set(set_payload["deltas"]) == set(CORE_METRIC_NAMES)
        for report in set_payload["candidate_reports"]:
            validate(report, "eval_report")
            assert report["eval_sets"] == ["general_capability"]
            assert report["gate_decision"]["outcome"] == "not_evaluated"
            assert report["schema_version"] == "1.0.0"
            assert set(CORE_METRIC_NAMES) <= set(report["confidence_intervals"])

    destination = tmp_path / "nested" / "comparison.json"
    written = write_comparison_artifact(artifact, destination)
    assert written == destination.resolve()
    assert json.loads(destination.read_text(encoding="utf-8")) == artifact


def test_internal_shape_validation_rejects_bad_artifact(
    benchmark_settings: BenchmarkSettings,
) -> None:
    """The internal artifact cannot masquerade as the eval_report contract."""
    artifact = build_comparison_artifact(compare_candidates(benchmark_settings))
    bad_type = dict(artifact)
    bad_type["artifact_type"] = "eval_report"
    with pytest.raises(ValidationError, match="artifact_type"):
        validate_comparison_artifact(bad_type)

    bad_sets: dict[str, Any] = json.loads(json.dumps(artifact))
    for set_payload in bad_sets["sets"].values():
        del set_payload["deltas"]["slot_f1"]
    with pytest.raises(ValidationError, match="slot_f1"):
        validate_comparison_artifact(bad_sets)
