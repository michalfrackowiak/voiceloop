"""P3-07: one-command fake-only CLI and structured errors."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from services.release.shadow.cli import main


def test_cli_writes_outcome_comparison_from_synthetic_fixtures(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    destination = tmp_path / "nested" / "comparison.json"

    code = main(
        [
            "--fixtures",
            "tests/fixtures/turns.jsonl",
            "--output",
            str(destination),
            "--max-pending",
            "64",
            "--budget-ms",
            "2.0",
        ]
    )

    assert code == 0
    text = destination.read_text(encoding="utf-8")
    report = json.loads(text)
    assert report["sample_counts"]["turns_observed"] == 36
    assert report["sample_counts"]["turns_mirrored"] == 36
    assert report["sample_counts"]["turns_dropped"] == 0
    assert report["sample_counts"]["calls_scored"] == 6
    assert report["metrics"]["candidate"]["outcome_call_accuracy"] == 1.0
    assert report["metrics"]["champion"]["outcome_call_accuracy"] == 0.5
    assert report["deltas"]["outcome_call_accuracy"] == 0.5
    assert report["promotion_decision"] == "not_evaluated"
    for secret in ("Ada Fixture", "ada.fixture@example.com", "+1 555 010 0001"):
        assert secret not in text
    stdout = capsys.readouterr().out
    assert "SYNTHETIC/FIXTURE STAND-IN" in stdout
    assert "Calls scored: 6" in stdout
    assert "promotion_decision: not_evaluated" in stdout


def test_cli_missing_fixtures_returns_structured_error(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code = main(
        ["--fixtures", str(tmp_path / "missing.jsonl"), "--output", str(tmp_path / "x.json")]
    )

    assert code == 2
    error = json.loads(capsys.readouterr().err)
    assert error["code"] == "io_error"
    assert "missing.jsonl" in error["message"]


def test_cli_rejects_non_positive_budget(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code = main(
        [
            "--fixtures",
            "tests/fixtures/turns.jsonl",
            "--output",
            str(tmp_path / "x.json"),
            "--budget-ms",
            "0",
        ]
    )

    assert code == 2
    error = json.loads(capsys.readouterr().err)
    assert error["code"] == "config_error"
