"""P3-09 CLI: one-command rehearsal, verification and descriptive drift."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from services.datasets.golden.maintenance import (
    ARTIFACT_LABEL,
    HUMAN_REVIEW_GATE,
    SEALED_CONTENT_HASH,
    GoldenMaintenanceSettings,
    cli,
)


def test_cli_propose_verify_and_drift(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """The three commands emit canonical JSON and never change sealed provenance."""
    output_dir = tmp_path / "cli-artifacts"
    assert (
        cli.main(
            [
                "propose",
                "--output-dir",
                str(output_dir),
                "--actor",
                "data-steward",
                "--purpose",
                "P3-09 synthetic rehearsal",
            ]
        )
        == 0
    )
    proposed = json.loads(capsys.readouterr().out)
    assert proposed["ok"] is True
    assert proposed["label"] == ARTIFACT_LABEL
    assert proposed["version"] == "candidate.v1"
    assert proposed["sealed_content_hash"] == SEALED_CONTENT_HASH

    assert cli.main(["verify", "--output-dir", str(output_dir), "--version", "candidate.v1"]) == 0
    verified = json.loads(capsys.readouterr().out)
    assert verified["ok"] is True
    assert verified["errors"] == []

    report_path = tmp_path / "drift.json"
    assert cli.main(["drift", "--output", str(report_path)]) == 0
    drift = json.loads(capsys.readouterr().out)
    assert drift["sample_counts"]["sealed_golden_turns"] == 36
    assert drift["interpretation"]["mode"] == "descriptive_only"
    assert drift["human_review_gate"] == HUMAN_REVIEW_GATE
    assert report_path.exists()


def test_cli_refuses_sealed_output_and_pii_bearing_purpose(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Unsafe destinations and free text fail closed with structured errors."""
    code = cli.main(
        [
            "propose",
            "--output-dir",
            "services/datasets/golden/artifacts",
            "--actor",
            "data-steward",
            "--purpose",
            "P3-09 synthetic rehearsal",
        ]
    )
    assert code == 2
    error = json.loads(capsys.readouterr().err)
    assert error["code"] == "validation_error"

    code = cli.main(
        [
            "propose",
            "--actor",
            "data-steward",
            "--purpose",
            "contact alice@example.com",
        ]
    )
    assert code == 2
    error = json.loads(capsys.readouterr().err)
    assert error["code"] == "validation_error"
    assert "alice@example.com" not in json.dumps(error)


def test_cli_rejects_unlisted_and_reviewer_actors(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """CLI proposal actors must be allowlisted and separate from reviewers."""
    settings = GoldenMaintenanceSettings(output_dir=str(tmp_path / "blocked"))
    for actor_id, expected in (
        ("unknown-actor-999", "allowlist"),
        (settings.first_reviewer_id, "reviewer"),
    ):
        code = cli.main(
            [
                "propose",
                "--output-dir",
                settings.output_dir,
                "--actor",
                actor_id,
                "--purpose",
                "P3-09 synthetic rehearsal",
            ]
        )
        assert code == 2
        error = json.loads(capsys.readouterr().err)
        assert error["code"] == "validation_error"
        assert expected in error["message"]
    assert not Path(settings.output_dir).exists()
