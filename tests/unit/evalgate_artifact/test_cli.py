"""CLI round-trip and fail-closed behaviour for P3-04."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from services.evalgate.artifact.cli import main
from services.evalgate.artifact.report import EXTENSION_KEY, validate_stt_report
from services.evalgate.artifact.settings import ArtifactSettings


def _argv(settings: ArtifactSettings, output: Path, behaviour: str) -> list[str]:
    """Build deterministic CLI arguments that write only into ``tmp_path``."""
    return [
        "--output",
        str(output),
        "--artifact-store",
        str(settings.artifact_store_dir),
        "--behaviour",
        behaviour,
        "--iterations",
        str(settings.bootstrap_iterations),
        "--seed",
        str(settings.bootstrap_seed),
    ]


def test_cli_healthy_writes_valid_report(
    settings: ArtifactSettings, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    """A healthy fake produces a validated report and a concise stdout summary."""
    output = tmp_path / "healthy.json"
    assert main(_argv(settings, output, "healthy")) == 0

    captured = capsys.readouterr()
    assert "Wrote STT artifact report" in captured.out
    assert "Primary metric: e2e_booking_accuracy" in captured.out
    assert "SYNTHETIC/FIXTURE STAND-IN" in captured.out
    assert captured.err == ""

    report = json.loads(output.read_text(encoding="utf-8"))
    validate_stt_report(report)
    assert report["decoding_settings"][EXTENSION_KEY]["probes"]["status"] == "pass"


def test_cli_empty_output_fails_closed(
    settings: ArtifactSettings, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    """An injected empty-output artifact still writes evidence but exits 2."""
    output = tmp_path / "empty.json"
    assert main(_argv(settings, output, "empty")) == 2

    captured = capsys.readouterr()
    error = json.loads(captured.err)
    assert error["code"] == "artifact_probe_failed"
    assert error["errors_by_code"]["empty_output"] > 0
    assert error["artifact_sha256"]
    assert "Wrote STT artifact report" in captured.out

    report = json.loads(output.read_text(encoding="utf-8"))
    validate_stt_report(report)
    assert report["gate_decision"]["outcome"] == "fail"
