"""Fail-closed P2-04 metric ingestion and measured go/no-go seam tests."""

from __future__ import annotations

import copy
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest

from ui import cli
from ui.constants import CLICK_ACTIONS, SYNTHETIC_LABEL, USABILITY_ACCEPTANCE
from ui.go_no_go import (
    MALFORMED_PER_ITEM_SECTIONS,
    PER_ACTION_CLICK_REGRESSION,
    decide_from_files,
    decide_go_no_go,
    read_json,
    write_json_report,
)


def _measurement(
    *,
    time_seconds: float,
    clicks: float,
    edit_cost: float,
    extra_seek_clicks: float = 0.0,
    omit_action: str | None = None,
) -> dict[str, Any]:
    """Build one deterministic, non-subjective tool measurement for tests."""
    counts: dict[str, float] = {action: 1.0 for action in CLICK_ACTIONS}
    seconds: dict[str, float] = {action: 0.5 for action in CLICK_ACTIONS}
    if omit_action is not None:
        counts.pop(omit_action)
        seconds.pop(omit_action)
    if extra_seek_clicks:
        counts["seek"] = counts.get("seek", 0.0) + extra_seek_clicks
        seconds["seek"] = seconds.get("seek", 0.0) + extra_seek_clicks
    return {
        "time_per_item_seconds": time_seconds,
        "clicks_per_item": clicks,
        "edit_cost_per_item": edit_cost,
        "click_counts": counts,
        "click_seconds_by_action": seconds,
    }


def _measured(
    adapter: dict[str, Any],
    custom: dict[str, Any],
    source: str = "injected_test_double",
) -> dict[str, Any]:
    """Build an injected measurement envelope; it is a seam, not real evidence."""
    return {
        "real_reviewers_used": True,
        "source": source,
        "items": 12,
        "tools": {"adapter_baseline": adapter, "custom_ui": custom},
    }


def _real_provenance_report(report: Mapping[str, Any]) -> dict[str, Any]:
    """Clone the P2-04 shape with injected real-reviewer provenance for seam tests."""
    result: dict[str, Any] = copy.deepcopy(dict(report))
    result["status"] = "INJECTED_TEST_DOUBLE_REVIEWED"
    provenance: dict[str, Any] = dict(result["provenance"])
    provenance["real_reviewers_used"] = True
    result["provenance"] = provenance
    result["limitations"] = ["injected_test_double"]
    return result


def test_committed_synthetic_metrics_fail_closed(p2_04_report: dict[str, Any]) -> None:
    """The committed P2-04 report cannot justify a custom UI build decision."""
    report = decide_go_no_go(p2_04_report)

    payload = report.to_dict()
    assert payload["decision"] == "no_go_fail_closed"
    assert set(payload["reasons"]) == {
        "no_clickstream_or_ui_timing_measured",
        "no_real_reviewed_metrics",
        "p2_04_synthetic_fixture_stand_in",
    }
    assert payload["measured_comparison"] == {}
    assert payload["clicks_that_hurt"] == []
    assert payload["label"] == SYNTHETIC_LABEL
    assert payload["usability_status"] == "NOT DONE"
    assert payload["usability_acceptance"] == USABILITY_ACCEPTANCE
    assert payload["promotion"] == "not_evaluated"
    assert payload["alias_threshold_change"] == "none"


def test_subjective_input_is_rejected(p2_04_report: dict[str, Any]) -> None:
    """A preference field cannot enter the go/no-go decision."""
    report = decide_go_no_go({"preference": "custom", **p2_04_report})

    assert "subjective_input_rejected" in report.to_dict()["reasons"]


def test_sensitive_input_is_rejected(p2_04_report: dict[str, Any]) -> None:
    """A PII-looking field in injected measurements fails closed."""
    measured = _measured(
        _measurement(time_seconds=50, clicks=10, edit_cost=2),
        _measurement(time_seconds=45, clicks=9, edit_cost=1),
    )
    measured["reviewer_email"] = "ada.fixture@example.com"

    report = decide_go_no_go(_real_provenance_report(p2_04_report), measured)

    assert report.to_dict()["decision"] == "no_go_fail_closed"
    assert "sensitive_input_rejected" in report.to_dict()["reasons"]


def test_real_provenance_still_requires_measurements(p2_04_report: dict[str, Any]) -> None:
    """Real-reviewer provenance alone is not a click/time/edit comparison."""
    report = decide_go_no_go(_real_provenance_report(p2_04_report))

    assert set(report.to_dict()["reasons"]) == {
        "no_measured_click_time_edit_comparison",
        "no_real_reviewed_metrics",
    }


def test_synthetic_measurements_are_rejected(p2_04_report: dict[str, Any]) -> None:
    """Synthetic injected measurements cannot be promoted to real evidence."""
    measured = _measured(
        _measurement(time_seconds=50, clicks=10, edit_cost=2),
        _measurement(time_seconds=45, clicks=9, edit_cost=1),
        source="SYNTHETIC_TEST_DOUBLE",
    )

    report = decide_go_no_go(_real_provenance_report(p2_04_report), measured)

    assert "measured_comparison_synthetic" in report.to_dict()["reasons"]


def test_measured_no_worse_is_go_but_usability_stays_open(
    p2_04_report: dict[str, Any],
) -> None:
    """An injected, real-provenance comparison can pass the build-seam check only."""
    measured = _measured(
        _measurement(time_seconds=50, clicks=10, edit_cost=2),
        _measurement(time_seconds=45, clicks=9, edit_cost=1),
    )

    payload = decide_go_no_go(_real_provenance_report(p2_04_report), measured).to_dict()

    assert payload["decision"] == "go"
    assert payload["reasons"] == ["measured_no_worse_than_adapter_baseline"]
    assert payload["clicks_that_hurt"] == []
    assert payload["measured_comparison"]["deltas_custom_minus_adapter"] == {
        "time_per_item_seconds": -5.0,
        "clicks_per_item": -1.0,
        "edit_cost_per_item": -1.0,
    }
    assert payload["usability_status"] == "NOT DONE"
    assert "RELEASE BLOCKER" in payload["usability_acceptance"]
    assert payload["promotion"] == "not_evaluated"
    assert payload["alias_threshold_change"] == "none"


def test_measured_regression_returns_no_go_and_lists_hurting_clicks(
    p2_04_report: dict[str, Any],
) -> None:
    """A measured regression names the click action that got worse."""
    measured = _measured(
        _measurement(time_seconds=50, clicks=10, edit_cost=2),
        _measurement(
            time_seconds=55,
            clicks=12,
            edit_cost=1,
            extra_seek_clicks=2,
        ),
    )

    payload = decide_go_no_go(_real_provenance_report(p2_04_report), measured).to_dict()

    assert payload["decision"] == "no_go_measured_regression"
    assert "time_per_item_seconds_worse" in payload["reasons"]
    assert "clicks_per_item_worse" in payload["reasons"]
    assert payload["clicks_that_hurt"] == [
        {"action": "seek", "extra_clicks": 2.0, "extra_seconds": 2.0}
    ]


def test_neutral_aggregates_with_one_worse_click_action_are_no_go(
    p2_04_report: dict[str, Any],
) -> None:
    """A per-action click regression gates no-go even when totals look neutral."""
    adapter = _measurement(time_seconds=50, clicks=10, edit_cost=2)
    custom = _measurement(time_seconds=50, clicks=10, edit_cost=2)
    custom["click_counts"]["seek"] += 1.0
    measured = _measured(adapter, custom)

    payload = decide_go_no_go(_real_provenance_report(p2_04_report), measured).to_dict()

    assert payload["measured_comparison"]["deltas_custom_minus_adapter"] == {
        "time_per_item_seconds": 0.0,
        "clicks_per_item": 0.0,
        "edit_cost_per_item": 0.0,
    }
    assert payload["decision"] == "no_go_measured_regression"
    assert payload["reasons"] == [PER_ACTION_CLICK_REGRESSION]
    assert payload["clicks_that_hurt"] == [
        {"action": "seek", "extra_clicks": 1.0, "extra_seconds": 0.0}
    ]
    assert payload["usability_status"] == "NOT DONE"
    assert payload["promotion"] == "not_evaluated"


def test_neutral_aggregates_with_one_worse_action_time_are_no_go(
    p2_04_report: dict[str, Any],
) -> None:
    """A per-action time regression gates no-go even when totals look neutral."""
    adapter = _measurement(time_seconds=50, clicks=10, edit_cost=2)
    custom = _measurement(time_seconds=50, clicks=10, edit_cost=2)
    custom["click_seconds_by_action"]["diff_toggle"] += 0.25
    measured = _measured(adapter, custom)

    payload = decide_go_no_go(_real_provenance_report(p2_04_report), measured).to_dict()

    assert payload["decision"] == "no_go_measured_regression"
    assert payload["reasons"] == [PER_ACTION_CLICK_REGRESSION]
    assert payload["clicks_that_hurt"] == [
        {"action": "diff_toggle", "extra_clicks": 0.0, "extra_seconds": 0.25}
    ]


def test_malformed_per_item_section_returns_structured_fail_closed(
    p2_04_report: dict[str, Any],
) -> None:
    """A malformed summary section never raises a bare ``dict`` error."""
    report = _real_provenance_report(p2_04_report)
    report["sections"]["per_item"]["time_summary"] = "not-a-mapping"

    payload = decide_go_no_go(report).to_dict()

    assert payload["decision"] == "no_go_fail_closed"
    assert MALFORMED_PER_ITEM_SECTIONS in payload["reasons"]
    assert payload["p2_04_summary"] == {}


def test_cli_malformed_per_item_section_uses_structured_failure_path(
    p2_04_report: dict[str, Any], tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Malformed metrics reach the CLI's normal exit-2 structured error path."""
    report = _real_provenance_report(p2_04_report)
    report["sections"]["per_item"]["edit_distance_summary"] = "not-a-mapping"
    metrics_path = tmp_path / "malformed_metrics.json"
    output_path = tmp_path / "malformed_decision.json"
    write_json_report(report, metrics_path)

    code = cli.main(["--go-no-go", "--metrics", str(metrics_path), "--output", str(output_path)])
    captured = capsys.readouterr()

    assert code == 2
    assert json.loads(captured.err) == {
        "code": "validation_error",
        "message": MALFORMED_PER_ITEM_SECTIONS,
    }
    payload = read_json(output_path)
    assert payload["decision"] == "no_go_fail_closed"
    assert MALFORMED_PER_ITEM_SECTIONS in payload["reasons"]


def test_incomplete_click_measurement_fails_closed(p2_04_report: dict[str, Any]) -> None:
    """Every deterministic click action must be measured for both tools."""
    measured = _measured(
        _measurement(time_seconds=50, clicks=10, edit_cost=2),
        _measurement(time_seconds=45, clicks=9, edit_cost=1, omit_action="seek"),
    )

    payload = decide_go_no_go(_real_provenance_report(p2_04_report), measured).to_dict()

    assert payload["decision"] == "no_go_fail_closed"
    assert "missing_click_measurement_seek" in payload["reasons"]


def test_file_round_trip_is_deterministic(p2_04_report: dict[str, Any], tmp_path: Path) -> None:
    """Writing and re-reading a report produces the same decision payload."""
    metrics_path = tmp_path / "p2_04_metrics.json"
    output_path = tmp_path / "go_no_go_report.json"
    write_json_report(dict(p2_04_report), metrics_path)
    expected = decide_go_no_go(p2_04_report).to_dict()

    actual = decide_from_files(metrics_path, None, output_path).to_dict()

    assert actual == expected
    assert output_path.read_text(encoding="utf-8") == (
        json.dumps(expected, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    )


def test_read_json_rejects_non_object(tmp_path: Path) -> None:
    """A non-object metrics file fails closed before any decision is made."""
    path = tmp_path / "metrics.json"
    path.write_text("[]", encoding="utf-8")

    with pytest.raises(ValueError):
        read_json(path)
