"""Fixture harness, slice and CLI tests."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from services.evalgate.harness import (
    ARTIFACT_KINDS,
    CRITICAL_SLICES,
    build_slice_metrics,
    evaluate,
    evaluate_rows,
    load_event,
    load_fixture_rows,
    main,
    predict,
    slice_names_for_row,
)
from services.evalgate.metrics import CORE_METRIC_NAMES, EntityPatterns, EvaluationError
from services.evalgate.settings import EvalgateSettings
from voiceloop_common import validate


def test_load_fixture_rows_reads_labels(turns_path: Path) -> None:
    """The read-only manifest loads in file order with the expected 36 turns."""
    rows = load_fixture_rows(turns_path)
    assert len(rows) == 36
    assert rows[0]["call_id"] == "call-0001"
    assert rows[-1]["turn_index"] == 5


def test_fake_artifacts_are_distinct_and_deterministic(
    load_rows: Any,
) -> None:
    """Identity replays recorded errors; oracle uses labels; no_slots emits nothing."""
    rows = load_rows()
    row = next(item for item in rows if item["case_id"] == "misheard_surname")
    event = load_event(str(row["event_path"]))
    identity = predict("identity", row, event)
    assert identity.transcript == "Mr. Smythe."
    assert identity.slots["guest_name"] == "Mr. Smythe"

    oracle = predict("oracle", row, {})
    assert oracle.transcript == "The name is Mr. Smith."
    assert oracle.slots["guest_name"] == "Mr. Smith"

    empty = predict("no_slots", row, {})
    assert empty.transcript == ""
    assert empty.slots == {}
    with pytest.raises(EvaluationError, match="unknown fake artifact"):
        predict("bogus", row, {})  # type: ignore[arg-type]


def test_slice_rules(load_rows: Any) -> None:
    """Critical slices follow reference entities, audio kind and language regex."""
    rows = load_rows()
    silence = next(row for row in rows if row["audio_kind"] == "silence")
    assert "silence" in slice_names_for_row(silence)

    named = next(row for row in rows if row["case_id"] == "misheard_surname")
    assert "names" in slice_names_for_row(named)
    assert "other_languages" not in slice_names_for_row(named)

    dated = next(
        row
        for row in rows
        if "December" in row["expected_transcript"] and "two" in row["expected_transcript"]
    )
    assert "dates" in slice_names_for_row(dated)
    assert "numbers" in slice_names_for_row(dated)

    custom = slice_names_for_row(
        {"expected_transcript": "Bonjour", "audio_kind": "tone"},
        EntityPatterns(),
        r"Bonjour",
    )
    assert "other_languages" in custom


def test_fixture_run_produces_valid_report(tmp_path: Path, turns_path: Path) -> None:
    """The harness runs on fixtures and emits a contract-valid report."""
    settings = EvalgateSettings(
        fixtures_path=str(turns_path),
        output_path=str(tmp_path / "report.json"),
        bootstrap_iterations=40,
        bootstrap_seed=11,
    )
    report = evaluate("fake://evalgate/identity", ["general_capability"], settings)
    validate(report, "eval_report")

    assert report["schema_version"] == "1.0.0"
    assert report["artifact_uri"] == "fake://evalgate/identity"
    assert report["eval_sets"] == ["general_capability"]
    assert report["generated_at"].endswith("Z")
    assert set(CORE_METRIC_NAMES) <= set(report["metrics"])
    assert set(CORE_METRIC_NAMES) <= set(report["confidence_intervals"])
    for interval in report["confidence_intervals"].values():
        assert interval["lower"] <= interval["upper"]
        assert interval["level"] == 0.95
    assert set(CRITICAL_SLICES) <= set(report["slice_metrics"])
    assert report["slice_metrics"]["other_languages"]["turns"] == 0.0
    assert report["decoding_settings"]["condition_on_previous_text"] is False
    assert report["gate_decision"]["outcome"] == "not_evaluated"
    assert 0.0 <= report["metrics"]["e2e_booking_accuracy"] <= 1.0

    again = evaluate("fake://evalgate/identity", ["general_capability"], settings)
    assert again["confidence_intervals"] == report["confidence_intervals"]


def test_oracle_challenger_wins_on_primary_metric(turns_path: Path) -> None:
    """A settings-selected better artifact yields a positive E2E delta CI."""
    settings = EvalgateSettings(
        fixtures_path=str(turns_path),
        champion_artifact="identity",
        challenger_artifact="oracle",
        bootstrap_iterations=100,
        bootstrap_seed=7,
    )
    report = evaluate("fake://evalgate/oracle", ["general_capability"], settings)
    assert report["metrics"]["e2e_booking_accuracy"] == 1.0
    primary_ci = report["confidence_intervals"]["e2e_booking_accuracy"]
    assert primary_ci["lower"] > 0.0
    assert report["confidence_intervals"]["wer"]["upper"] < 0.0


def test_slice_metrics_align_with_turns(load_rows: Any) -> None:
    """Slice aggregation uses the same ordered turns and reports counts."""
    rows = load_rows()
    champion, challenger = evaluate_rows(rows, "identity", "oracle", EntityPatterns())
    assert len(champion) == len(challenger) == len(rows)
    slices = build_slice_metrics(rows, challenger, EntityPatterns())
    assert set(slices) == set(CRITICAL_SLICES)
    assert slices["names"]["turns"] > 0
    assert slices["names"]["e2e_booking_accuracy"] == 1.0
    assert all(values["turns"] >= 0 for values in slices.values())
    assert ARTIFACT_KINDS == ("identity", "oracle", "no_slots")


def test_cli_writes_valid_report(tmp_path: Path) -> None:
    """``python -m services.evalgate.harness`` equivalent writes a valid file."""
    output = tmp_path / "cli-report.json"
    exit_code = main(
        ["--output", str(output), "--iterations", "20", "--seed", "3", "--eval-set", "regression"]
    )
    assert exit_code == 0
    report = json.loads(output.read_text(encoding="utf-8"))
    validate(report, "eval_report")
    assert report["eval_sets"] == ["regression"]
