"""Build, validate and write ``report.json`` for the evaluation contract."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from voiceloop_common import utcnow, validate

from .bootstrap import ConfidenceInterval

REPORT_SCHEMA = "eval_report"
REPORT_SCHEMA_VERSION = "1.0.0"


def format_timestamp(value: datetime) -> str:
    """Return an RFC 3339 UTC timestamp with a ``Z`` suffix."""
    return value.astimezone(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


def build_report(
    *,
    artifact_uri: str,
    eval_sets: Sequence[str],
    metrics: Mapping[str, float],
    confidence_intervals: Mapping[str, ConfidenceInterval],
    slice_metrics: Mapping[str, Mapping[str, float]],
    decoding_settings: Mapping[str, Any],
    generated_at: datetime | None = None,
    gate_decision: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Assemble a report and validate it against ``eval_report`` 1.0.0.

    ``confidence_intervals`` bounds describe paired ``challenger - champion``
    differences; see :mod:`services.evalgate.bootstrap`. The optional
    ``gate_decision`` is not computed here: P1-07 never promotes a model.
    """
    report: dict[str, Any] = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "artifact_uri": artifact_uri,
        "generated_at": format_timestamp(generated_at or utcnow()),
        "eval_sets": list(eval_sets),
        "metrics": dict(metrics),
        "confidence_intervals": {
            name: interval.to_dict() for name, interval in confidence_intervals.items()
        },
        "slice_metrics": {name: dict(values) for name, values in slice_metrics.items()},
        "decoding_settings": dict(decoding_settings),
    }
    if gate_decision is not None:
        report["gate_decision"] = dict(gate_decision)
    validate(report, REPORT_SCHEMA)
    return report


def write_report(report: Mapping[str, Any], path: str | Path) -> Path:
    """Validate and write a report as deterministic, sorted UTF-8 JSON.

    Args:
        report: Report payload produced by :func:`build_report`.
        path: Destination file; parent directories are created if needed.

    Returns:
        The resolved path written to.

    Raises:
        voiceloop_common.errors.ValidationError: If the report violates the
            evaluation report contract.

    """
    validate(report, REPORT_SCHEMA)
    destination = Path(path).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return destination
