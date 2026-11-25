"""CLI tests for the fixture-only review metrics report."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from services.review_adapter.metrics.cli import main
from services.review_adapter.metrics.events import sha256_directory, sha256_file
from services.review_adapter.metrics.report import STATUS


def test_main_writes_synthetic_report(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], repo_root: Path
) -> None:
    """The CLI writes one fixture-only report and states its synthetic status."""
    output = tmp_path / "nested" / "report.json"
    fixture_before = sha256_file(repo_root / "tests/fixtures/turns.jsonl")
    golden_before = sha256_directory(repo_root / "services/datasets/golden/artifacts")

    exit_code = main(["--output", str(output)])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert output.is_file()
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["status"] == STATUS
    assert report["provenance"]["fixture_hash_unchanged"] is True
    assert report["provenance"]["golden_files_hash_unchanged"] is True
    assert "SYNTHETIC/FIXTURE STAND-IN" in captured.out
    assert "no custom-UI decision made" in captured.out
    assert captured.err == ""
    assert sha256_file(repo_root / "tests/fixtures/turns.jsonl") == fixture_before
    assert sha256_directory(repo_root / "services/datasets/golden/artifacts") == golden_before


def test_main_emits_structured_error_for_missing_golden(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A bad read-only input produces a structured error and no report."""
    output = tmp_path / "report.json"

    exit_code = main(
        [
            "--output",
            str(output),
            "--golden-version",
            "missing-version",
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 2
    assert not output.exists()
    error = json.loads(captured.err)
    assert error["code"] == "validation_error"
    assert error["message"]
