"""Rule pipeline and read-only golden regression tests."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import pytest

from services.evalgate.metrics import Prediction, evaluate_turn
from services.trainers.vocab import (
    RuleOutcome,
    VocabRuleSet,
    VocabSettings,
    check_golden_regression,
)
from services.trainers.vocab.rules import main

_E2E_CASES = {
    "misheard_surname",
    "date_confusion",
    "schema_invalid_parse",
    "misheard_guests",
    "nights_misparsed",
    "schema_invalid_missing_field",
}


def _e2e_correct(
    row: Mapping[str, Any],
    outcome: RuleOutcome,
) -> bool:
    """Score one transformed identity prediction with P1-07 turn metrics."""
    return bool(
        evaluate_turn(
            str(row["expected_transcript"]),
            dict(row["expected_slots"]),
            Prediction(transcript=outcome.transcript, slots=outcome.slots),
        ).e2e_correct
    )


@pytest.fixture(scope="module")
def regression_report() -> dict[str, Any]:
    """Run the read-only harness comparison once for the module."""
    return check_golden_regression()


def test_alias_rule_fixes_misheard_surname(
    ruleset: VocabRuleSet,
    row_by_case: Callable[[str], dict[str, Any]],
    identity_prediction: Callable[[Mapping[str, Any]], tuple[Prediction, Mapping[str, Any]]],
) -> None:
    row = row_by_case("misheard_surname")
    prediction, _ = identity_prediction(row)
    outcome = ruleset.apply(prediction.transcript, prediction.slots, use_boosting=False)
    assert "alias_name" in outcome.rules
    assert "alias_guest_name" in outcome.rules
    assert outcome.slots["guest_name"] == "Mr. Smith"
    assert _e2e_correct(row, outcome)


def test_number_rules_fix_date_confusion_and_nights_misparsed(
    ruleset: VocabRuleSet,
    row_by_case: Callable[[str], dict[str, Any]],
    identity_prediction: Callable[[Mapping[str, Any]], tuple[Prediction, Mapping[str, Any]]],
) -> None:
    for case_id in ("date_confusion", "nights_misparsed"):
        row = row_by_case(case_id)
        prediction, _ = identity_prediction(row)
        outcome = ruleset.apply(prediction.transcript, prediction.slots, use_boosting=False)
        assert "nights_to_checkout" in outcome.rules
        assert _e2e_correct(row, outcome)


def test_date_rule_fixes_schema_invalid_parse(
    ruleset: VocabRuleSet,
    row_by_case: Callable[[str], dict[str, Any]],
    identity_prediction: Callable[[Mapping[str, Any]], tuple[Prediction, Mapping[str, Any]]],
) -> None:
    row = row_by_case("schema_invalid_parse")
    prediction, _ = identity_prediction(row)
    outcome = ruleset.apply(prediction.transcript, prediction.slots, use_boosting=False)
    assert "arrival_departure_order" in outcome.rules
    assert _e2e_correct(row, outcome)


def test_synthetic_boosting_seam_fixes_misheard_guests(
    settings: VocabSettings,
    row_by_case: Callable[[str], dict[str, Any]],
    identity_prediction: Callable[[Mapping[str, Any]], tuple[Prediction, Mapping[str, Any]]],
) -> None:
    row = row_by_case("misheard_guests")
    prediction, event = identity_prediction(row)
    stt = event.get("stt") or {}
    words = stt.get("words", []) if isinstance(stt, Mapping) else []
    without = VocabRuleSet(settings.model_copy(update={"enable_boosting": False}))
    assert not _e2e_correct(
        row, without.apply(prediction.transcript, prediction.slots, words=words)
    )

    with_boosting = VocabRuleSet(settings)
    outcome = with_boosting.apply(prediction.transcript, prediction.slots, words=words)
    assert "phrase_boost_synthetic" in outcome.rules
    assert outcome.transcript == "Make it four guests."
    assert _e2e_correct(row, outcome)


def test_clean_turns_are_not_modified(
    ruleset: VocabRuleSet,
    fixture_rows: list[dict[str, Any]],
    identity_prediction: Callable[[Mapping[str, Any]], tuple[Prediction, Mapping[str, Any]]],
) -> None:
    changed_clean = []
    for row in fixture_rows:
        if row["failure_type"] != "clean":
            continue
        prediction, _ = identity_prediction(row)
        outcome = ruleset.apply(prediction.transcript, prediction.slots)
        if outcome.changed:
            changed_clean.append(row["case_id"])
    assert changed_clean == []


def test_golden_regression_report_is_clean_and_fixes_named_failures(
    regression_report: dict[str, Any],
) -> None:
    assert regression_report["ok"] is True
    assert regression_report["regressions"] == []
    assert regression_report["seal_ok_after"] is True
    assert regression_report["artifact_hashes_unchanged"] is True
    assert (
        regression_report["golden_content_hash"]
        == "662cdaf43ee2d5fde03faf5e69fde196114d4f312406d94c00bf53714d03da8e"
    )
    before = regression_report["metrics_before"]
    after = regression_report["metrics_after"]
    assert before["e2e_booking_accuracy"] == pytest.approx(30 / 36)
    assert after["e2e_booking_accuracy"] == 1.0
    assert after["wer"] <= before["wer"]
    assert after["entity_error"] <= before["entity_error"]
    for metric in ("slot_f1", "schema_valid_rate"):
        assert after[metric] >= before[metric]
    fixed = {item["case_id"] for item in regression_report["fixture_e2e_fixed"]}
    assert _E2E_CASES <= fixed
    assert "SYNTHETIC/FIXTURE STAND-IN" in regression_report["synthetic_warning"]


def test_cli_writes_report_and_boosting_config(tmp_path: Path) -> None:
    report_path = tmp_path / "report.json"
    config_path = tmp_path / "boost.json"
    assert main(["--report", str(report_path), "--config-output", str(config_path)]) == 0
    report = json.loads(report_path.read_text(encoding="utf-8"))
    config = json.loads(config_path.read_text(encoding="utf-8"))
    assert report["ok"] is True
    assert config["synthetic"] is True
    assert config["schema_version"] == "voiceloop.vocab-boost.v1"
