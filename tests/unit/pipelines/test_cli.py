"""Tests for the P2-10 fixture CLI."""

from __future__ import annotations

import io
import json
from pathlib import Path

import pytest

from infra.pipelines.chain import PipelineSteps
from infra.pipelines.cli import main
from infra.pipelines.settings import PipelineSettings

from .conftest import make_recording_steps


def _run(
    *argv: str,
    settings: PipelineSettings | None = None,
    steps: PipelineSteps | None = None,
) -> tuple[int, str]:
    """Invoke main with a string buffer and return the exit code and output."""
    buffer = io.StringIO()
    code = main(list(argv), settings=settings, steps=steps, stdout=buffer)
    return code, buffer.getvalue()


def test_dry_run_manual_executes_full_chain(pipeline_settings: PipelineSettings) -> None:
    code, output = _run("--dry-run", "--manual", settings=pipeline_settings)
    assert code == 0
    assert '"reason": "manual_override"' in output
    assert "STEP dataset:" in output
    assert "STEP train:" in output
    assert "STEP evaluate:" in output
    assert '"status": "success"' in output
    assert '"eval_report_path": "fake://evalgate/report.json"' in output


def test_dry_run_never_records_an_implicit_manual_override(
    pipeline_settings: PipelineSettings,
) -> None:
    high_threshold = pipeline_settings.model_copy(update={"min_new_verified": 200})
    code, skipped_output = _run("--dry-run", settings=high_threshold)
    assert code == 0
    assert '"reason": null' in skipped_output
    assert "SKIPPED: no data-volume" in skipped_output
    assert "manual_override" not in skipped_output

    code, manual_output = _run("--dry-run", "--manual", settings=high_threshold)
    assert code == 0
    assert '"reason": "manual_override"' in manual_output


def test_check_without_trigger_never_runs_chain(pipeline_settings: PipelineSettings) -> None:
    high_threshold = pipeline_settings.model_copy(update={"min_new_verified": 200})
    code, output = _run("--dry-run", "--check", settings=high_threshold)
    assert code == 0
    assert '"triggered": false' in output
    assert "STEP " not in output
    assert "RUN RECORD:" not in output


def test_check_with_drift_reports_trigger_but_no_chain(pipeline_settings: PipelineSettings) -> None:
    code, output = _run("--dry-run", "--check", "--drift-alarm", settings=pipeline_settings)
    assert code == 0
    assert '"reason": "drift_alarm"' in output
    assert "STEP " not in output


def test_dry_run_drift_executes_chain_without_manual_reason(
    pipeline_settings: PipelineSettings,
) -> None:
    high_threshold = pipeline_settings.model_copy(update={"min_new_verified": 200})
    code, output = _run("--dry-run", "--drift-alarm", settings=high_threshold)
    assert code == 0
    assert '"reason": "drift_alarm"' in output
    assert '"manual": false' in output
    assert "manual_override" not in output
    assert "STEP dataset:" in output
    assert '"status": "success"' in output


def test_state_file_counts_new_verified_data_since_last_run(
    tmp_path: Path,
    pipeline_settings: PipelineSettings,
) -> None:
    state_path = tmp_path / "state" / "last-run.json"
    small_threshold = pipeline_settings.model_copy(update={"min_new_verified": 1})

    code, first_output = _run(
        "--dry-run", "--manual", "--state", str(state_path), settings=small_threshold
    )
    assert code == 0
    assert '"status": "success"' in first_output
    assert state_path.exists()

    code, second_output = _run("--dry-run", "--state", str(state_path), settings=small_threshold)
    assert code == 0
    assert '"new_verified_count": 0' in second_output
    assert "SKIPPED: no data-volume" in second_output
    assert "STEP " not in second_output

    state = json.loads(state_path.read_text(encoding="utf-8"))
    state["verified_ids"] = state["verified_ids"][:-1]
    state_path.write_text(json.dumps(state), encoding="utf-8")

    code, third_output = _run("--dry-run", "--state", str(state_path), settings=small_threshold)
    assert code == 0
    assert '"reason": "new_verified_data"' in third_output
    assert '"status": "success"' in third_output


def test_injected_step_failure_exits_one_with_partial_record(
    pipeline_settings: PipelineSettings,
) -> None:
    steps = make_recording_steps(fail_at="dataset").steps
    code, output = _run("--manual", settings=pipeline_settings, steps=steps)
    assert code == 1
    assert "ERROR: dataset step failed: RuntimeError: dataset boom" in output
    assert '"status": "failed"' in output
    assert "STEP " not in output


def test_missing_fixture_file_exits_two(pipeline_settings: PipelineSettings) -> None:
    code, output = _run(
        "--dry-run",
        "--manual",
        "--fixtures",
        "tests/fixtures/does-not-exist.jsonl",
        settings=pipeline_settings,
    )
    assert code == 2
    assert "ERROR: ConfigError" in output


def test_non_dry_run_refuses_unwired_production_steps(
    pipeline_settings: PipelineSettings,
) -> None:
    code, output = _run("--manual", settings=pipeline_settings)
    assert code == 2
    assert "no production steps are configured yet" in output


def test_min_new_verified_override_is_recorded(
    pipeline_settings: PipelineSettings,
) -> None:
    code, output = _run(
        "--dry-run", "--check", "--min-new-verified", "100", settings=pipeline_settings
    )
    assert code == 0
    assert '"threshold": 100' in output
    assert '"triggered": false' in output


@pytest.mark.parametrize("invalid_threshold", ["0", "-5"])
def test_invalid_min_new_verified_override_cannot_false_trigger(
    tmp_path: Path,
    pipeline_settings: PipelineSettings,
    invalid_threshold: str,
) -> None:
    """CLI overrides must pass PipelineSettings validation even with zero new data."""
    state_path = tmp_path / "state" / "last-run.json"
    code, setup_output = _run(
        "--dry-run", "--manual", "--state", str(state_path), settings=pipeline_settings
    )
    assert code == 0
    assert '"status": "success"' in setup_output
    state_before = state_path.read_text(encoding="utf-8")

    code, output = _run(
        "--dry-run",
        "--state",
        str(state_path),
        "--min-new-verified",
        invalid_threshold,
        settings=pipeline_settings,
    )
    assert code == 2
    assert "ERROR:" in output
    assert "min_new_verified" in output
    assert "TRIGGER:" not in output
    assert "STEP " not in output
    assert state_path.read_text(encoding="utf-8") == state_before
