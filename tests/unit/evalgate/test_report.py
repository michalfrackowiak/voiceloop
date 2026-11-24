"""Report builder/writer conformance tests."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from services.evalgate.bootstrap import ConfidenceInterval
from services.evalgate.metrics import CORE_METRIC_NAMES
from services.evalgate.report import build_report, format_timestamp, write_report
from voiceloop_common import ValidationError, validate


def make_report(**overrides: Any) -> dict[str, Any]:
    """Build a minimal contract-valid report for tests."""
    arguments: dict[str, Any] = {
        "artifact_uri": "fake://evalgate/identity",
        "eval_sets": ["general_capability"],
        "metrics": {name: 0.5 for name in CORE_METRIC_NAMES},
        "confidence_intervals": {
            name: ConfidenceInterval(lower=0.25, upper=0.75, level=0.95)
            for name in CORE_METRIC_NAMES
        },
        "slice_metrics": {"names": {"wer": 0.1}, "other_languages": {"turns": 0.0}},
        "decoding_settings": {
            "hotwords": [],
            "initial_prompt": "",
            "condition_on_previous_text": False,
            "language": "en",
        },
        "generated_at": datetime(2026, 1, 2, 3, 4, 5, tzinfo=UTC),
    }
    arguments.update(overrides)
    return build_report(**arguments)


def test_build_report_uses_utc_and_validates() -> None:
    """Timestamps are RFC 3339 UTC and the full report passes the contract."""
    report = make_report()
    assert report["schema_version"] == "1.0.0"
    assert report["generated_at"] == "2026-01-02T03:04:05Z"
    assert report["confidence_intervals"]["wer"] == {
        "lower": 0.25,
        "upper": 0.75,
        "level": 0.95,
    }
    assert "gate_decision" not in report
    validate(report, "eval_report")
    assert format_timestamp(datetime(2026, 1, 2, 3, 4, 5, tzinfo=UTC)) == "2026-01-02T03:04:05Z"


def test_write_report_creates_parents_and_round_trips(tmp_path: Path) -> None:
    """Writing validates, creates directories and produces readable JSON."""
    destination = tmp_path / "nested" / "report.json"
    report = make_report(gate_decision={"outcome": "not_evaluated"})
    written = write_report(report, destination)
    assert written == destination.resolve()
    assert written.exists()
    decoded = written.read_text(encoding="utf-8")
    assert decoded.endswith("\n")
    validate(json.loads(decoded), "eval_report")


def test_writer_rejects_invalid_reports(tmp_path: Path) -> None:
    """Missing CI bounds are rejected before any file is written."""
    report = make_report()
    report["confidence_intervals"]["wer"].pop("upper")
    with pytest.raises(ValidationError):
        write_report(report, tmp_path / "invalid.json")
    assert not (tmp_path / "invalid.json").exists()
