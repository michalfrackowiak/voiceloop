"""CLI round-trip and structured-error tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from services.evalgate.benchmarks.cli import main
from voiceloop_common import validate


def test_one_command_writes_comparison_artifact(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The documented command writes an artifact containing schema-valid reports."""
    output = tmp_path / "comparison.json"
    assert (
        main(
            [
                "--candidates",
                "identity,degraded",
                "--output",
                str(output),
                "--iterations",
                "20",
                "--seed",
                "3",
            ]
        )
        == 0
    )
    artifact = json.loads(output.read_text(encoding="utf-8"))
    assert artifact["candidates"]["a"] == "identity"
    assert artifact["candidates"]["b"] == "degraded"
    assert artifact["candidates"]["delta_definition"] == "candidate A - candidate B"
    for set_payload in artifact["sets"].values():
        for report in set_payload["candidate_reports"]:
            validate(report, "eval_report")
    stdout = capsys.readouterr().out
    assert "Wrote comparison artifact:" in stdout
    assert "WER alone never justifies promotion" in stdout


def test_cli_reports_unknown_candidate(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """An unregistered candidate yields a structured config error, exit 2."""
    code = main(
        [
            "--candidates",
            "identity,whisper-large-v3",
            "--output",
            str(tmp_path / "unused.json"),
            "--iterations",
            "10",
        ]
    )
    assert code == 2
    error = json.loads(capsys.readouterr().err)
    assert error["code"] == "config_error"
    assert "unknown candidate runner" in error["message"]


def test_cli_rejects_single_candidate(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """A comparison needs two candidates, not one."""
    code = main(
        [
            "--candidates",
            "identity",
            "--output",
            str(tmp_path / "unused.json"),
            "--iterations",
            "10",
        ]
    )
    assert code == 2
    error = json.loads(capsys.readouterr().err)
    assert error["code"] == "config_error"
    assert "exactly two" in error["message"]
