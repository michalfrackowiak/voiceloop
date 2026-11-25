"""Internal comparison artifact builder for the P3-05 benchmark harness.

Each embedded per-candidate report is a full ``eval_report`` 1.0.0 payload,
built and validated by the committed P1-07 report writer. The outer comparison
artifact is deliberately a **different, internal shape**: it is documented here
and in the README, and it never pretends to be ``eval_report`` 1.0.0.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from services.evalgate.harness import build_report_timestamp
from services.evalgate.metrics import CORE_METRIC_NAMES
from voiceloop_common import ValidationError, validate

from .settings import BenchmarkSettings

COMPARISON_ARTIFACT_TYPE = "voiceloop.benchmark.comparison"
COMPARISON_SCHEMA_VERSION = "internal-1.0.0"
SCHEMA_REPORT_LABEL = "general_capability"
PRIMARY_METRIC = "e2e_booking_accuracy"

_REQUIRED_COMPARISON_KEYS = (
    "artifact_type",
    "comparison_schema_version",
    "generated_at",
    "status",
    "candidates",
    "provenance",
    "settings",
    "metric_semantics",
    "sets",
)


@dataclass(frozen=True)
class SetEvaluation:
    """Scored results for one read-only benchmark input set."""

    name: str
    label: str
    turns: int
    calls: int
    candidate_reports: tuple[dict[str, Any], ...]
    metrics_by_candidate: Mapping[str, Mapping[str, float]]
    slice_by_candidate: Mapping[str, Mapping[str, Mapping[str, float]]]
    deltas: Mapping[str, Mapping[str, float]]
    candidate_a: str
    candidate_b: str


@dataclass(frozen=True)
class ComparisonData:
    """Raw, in-memory comparison result before artifact serialisation."""

    candidates: tuple[str, str]
    provenance: Mapping[str, Any]
    settings: BenchmarkSettings
    sets: tuple[SetEvaluation, ...]


def build_comparison_artifact(data: ComparisonData) -> dict[str, Any]:
    """Assemble and validate the internal benchmark comparison artifact.

    Args:
        data: Candidate names, provenance, settings and evaluated sets.

    Returns:
        A JSON-serialisable artifact with embedded schema-valid per-candidate
        reports and paired-bootstrap ``A - B`` deltas for all five metrics.

    """
    artifact: dict[str, Any] = {
        "artifact_type": COMPARISON_ARTIFACT_TYPE,
        "comparison_schema_version": COMPARISON_SCHEMA_VERSION,
        "schema_contract": "internal; not schemas/eval_report.schema.json",
        "generated_at": build_report_timestamp(),
        "status": "SYNTHETIC/FIXTURE STAND-IN - NOT A PRODUCTION BENCHMARK",
        "candidates": {
            "a": data.candidates[0],
            "b": data.candidates[1],
            "delta_definition": "candidate A - candidate B",
        },
        "provenance": dict(data.provenance),
        "settings": {
            "bootstrap_level": data.settings.bootstrap_level,
            "bootstrap_iterations": data.settings.bootstrap_iterations,
            "bootstrap_seed": data.settings.bootstrap_seed,
            "decoding_settings": data.settings.decoding_settings(),
        },
        "metric_semantics": {
            "primary_metric": PRIMARY_METRIC,
            "primary_definition": "end-to-end booking-field accuracy from audio",
            "ci_definition": (
                "paired bootstrap over shared fixture turn indices; bounds are "
                "nearest-rank percentiles of candidate A - candidate B"
            ),
            "delta_direction": {
                "wer": "negative favors A",
                "entity_error": "negative favors A",
                "slot_f1": "positive favors A",
                "schema_valid_rate": "positive favors A",
                "e2e_booking_accuracy": "positive favors A",
            },
            "promotion_decision": "not_evaluated",
            "promotion_rule": (
                "E2E booking accuracy from audio is the only promotion metric; "
                "WER alone never promotes (context.md section 2 invariant 4)."
            ),
        },
        "sets": {
            evaluation.name: {
                "set_name": evaluation.name,
                "eval_set_label": evaluation.label,
                "turns": evaluation.turns,
                "calls": evaluation.calls,
                "synthetic": True,
                "candidate_reports": [dict(report) for report in evaluation.candidate_reports],
                "metrics": {
                    name: dict(values) for name, values in evaluation.metrics_by_candidate.items()
                },
                "slice_metrics": {
                    name: {slice_name: dict(values) for slice_name, values in slices.items()}
                    for name, slices in evaluation.slice_by_candidate.items()
                },
                "deltas": {name: dict(values) for name, values in evaluation.deltas.items()},
                "candidate_a": evaluation.candidate_a,
                "candidate_b": evaluation.candidate_b,
            }
            for evaluation in data.sets
        },
    }
    validate_comparison_artifact(artifact)
    return artifact


def validate_comparison_artifact(artifact: Mapping[str, Any]) -> None:
    """Check the internal artifact shape and every embedded contract report.

    Raises:
        ValidationError: If a required field is missing or an embedded report
            fails ``eval_report`` 1.0.0 validation.

    """
    missing = [key for key in _REQUIRED_COMPARISON_KEYS if key not in artifact]
    if missing:
        raise ValidationError(f"comparison artifact missing: {', '.join(missing)}")
    if artifact["artifact_type"] != COMPARISON_ARTIFACT_TYPE:
        raise ValidationError("comparison artifact has an unexpected artifact_type")
    if artifact["comparison_schema_version"] != COMPARISON_SCHEMA_VERSION:
        raise ValidationError("unsupported comparison_schema_version")
    candidates = artifact["candidates"]
    if not isinstance(candidates, dict) or candidates.get("a") == candidates.get("b"):
        raise ValidationError("comparison requires two distinct candidates")
    sets = artifact["sets"]
    if not isinstance(sets, dict) or not sets:
        raise ValidationError("comparison artifact has no evaluated sets")
    for set_payload in sets.values():
        if not isinstance(set_payload, dict):
            raise ValidationError("each comparison set must be an object")
        deltas = set_payload.get("deltas")
        if not isinstance(deltas, dict):
            raise ValidationError("each comparison set needs delta intervals")
        for metric in CORE_METRIC_NAMES:
            interval = deltas.get(metric)
            if not isinstance(interval, dict):
                raise ValidationError(f"missing delta interval for {metric}")
            if not interval["lower"] <= interval["upper"]:
                raise ValidationError(f"invalid delta bounds for {metric}")
        reports = set_payload.get("candidate_reports")
        if not isinstance(reports, Sequence) or isinstance(reports, (str, bytes)):
            raise ValidationError("candidate_reports must be a list")
        for report in reports:
            if not isinstance(report, Mapping):
                raise ValidationError("each candidate report must be an object")
            validate(dict(report), "eval_report")


def write_comparison_artifact(artifact: Mapping[str, Any], path: str | Path) -> Path:
    """Validate and write the internal comparison artifact as sorted UTF-8 JSON.

    Args:
        artifact: Payload from :func:`build_comparison_artifact`.
        path: Destination file; parent directories are created if needed.

    Returns:
        The resolved destination path.

    """
    validate_comparison_artifact(artifact)
    destination = Path(path).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(artifact, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return destination
