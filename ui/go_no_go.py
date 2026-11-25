"""P4-03.1 go/no-go decision from P2-04 review metrics.

The only committed P2-04 evidence is a synthetic fixture replay, so the default
decision fails closed. A ``go`` requires independently injected measurements of
time, clicks and edit cost for both tools with real-reviewer provenance; no
subjective preference field is accepted and no promotion threshold is changed.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .constants import (
    CLICK_ACTIONS,
    EXPECTED_FIXTURE_SHA256,
    EXPECTED_GOLDEN_CONTENT_HASH,
    SUBJECTIVE_KEYS,
    SYNTHETIC_LABEL,
    USABILITY_ACCEPTANCE,
)
from .errors import UiValidationError
from .privacy import assert_no_sensitive_text

MEASUREMENT_KEYS = frozenset(
    {
        "time_per_item_seconds",
        "clicks_per_item",
        "edit_cost_per_item",
        "click_counts",
        "click_seconds_by_action",
    }
)


@dataclass(frozen=True)
class ToolMeasurement:
    """Measured, non-subjective numbers for one review tool."""

    time_per_item_seconds: float
    clicks_per_item: float
    edit_cost_per_item: float
    click_counts: dict[str, float]
    click_seconds_by_action: dict[str, float]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable measurement snapshot."""
        return {
            "time_per_item_seconds": self.time_per_item_seconds,
            "clicks_per_item": self.clicks_per_item,
            "edit_cost_per_item": self.edit_cost_per_item,
            "click_counts": dict(sorted(self.click_counts.items())),
            "click_seconds_by_action": dict(sorted(self.click_seconds_by_action.items())),
        }


@dataclass(frozen=True)
class GoNoGoReport:
    """Deterministic go/no-go report with an always-open usability criterion."""

    decision: str
    decision_scope: str
    reasons: tuple[str, ...]
    clicks_that_hurt: tuple[dict[str, Any], ...]
    measured_comparison: dict[str, Any]
    p2_04_summary: dict[str, Any]
    label: str
    usability_status: str
    usability_acceptance: str
    promotion: str
    alias_threshold_change: str

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable report, screened for PII-like strings."""
        result: dict[str, Any] = {
            "decision": self.decision,
            "decision_scope": self.decision_scope,
            "reasons": list(self.reasons),
            "clicks_that_hurt": [dict(row) for row in self.clicks_that_hurt],
            "measured_comparison": self.measured_comparison,
            "p2_04_summary": self.p2_04_summary,
            "label": self.label,
            "usability_status": self.usability_status,
            "usability_acceptance": self.usability_acceptance,
            "promotion": self.promotion,
            "alias_threshold_change": self.alias_threshold_change,
        }
        assert_no_sensitive_text(result)
        return result


def _walk_mapping_keys(node: Any) -> set[str]:
    """Collect all mapping keys in a JSON-like payload."""
    keys: set[str] = set()
    if isinstance(node, Mapping):
        for key, value in node.items():
            if isinstance(key, str):
                keys.add(key)
            keys.update(_walk_mapping_keys(value))
    elif isinstance(node, (list, tuple)):
        for value in node:
            keys.update(_walk_mapping_keys(value))
    return keys


def _number(value: Any) -> float | None:
    """Return a finite non-negative float, or ``None`` for invalid input."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    if not math.isfinite(number) or number < 0:
        return None
    return number


def _measurement(raw: Any) -> ToolMeasurement:
    """Parse one tool measurement, failing closed on missing or extra fields."""
    if not isinstance(raw, Mapping) or set(raw) != MEASUREMENT_KEYS:
        raise UiValidationError("invalid_measurement_fields")
    scalar_names = ("time_per_item_seconds", "clicks_per_item", "edit_cost_per_item")
    scalars: dict[str, float] = {}
    for name in scalar_names:
        value = _number(raw[name])
        if value is None:
            raise UiValidationError("invalid_measurement_scalar")
        scalars[name] = value
    counts: dict[str, float] = {}
    seconds: dict[str, float] = {}
    for field_name, target in (("click_counts", counts), ("click_seconds_by_action", seconds)):
        incoming = raw[field_name]
        if not isinstance(incoming, Mapping):
            raise UiValidationError("invalid_click_measurement")
        for action, value in incoming.items():
            if action not in CLICK_ACTIONS:
                raise UiValidationError("unknown_click_action")
            number = _number(value)
            if number is None:
                raise UiValidationError("invalid_click_measurement")
            target[str(action)] = number
    return ToolMeasurement(
        time_per_item_seconds=scalars["time_per_item_seconds"],
        clicks_per_item=scalars["clicks_per_item"],
        edit_cost_per_item=scalars["edit_cost_per_item"],
        click_counts=counts,
        click_seconds_by_action=seconds,
    )


def _fail(
    reasons: list[str],
    summary: dict[str, Any],
    decision: str = "no_go_fail_closed",
) -> GoNoGoReport:
    """Build a fail-closed report with no measured comparison claim."""
    return GoNoGoReport(
        decision=decision,
        decision_scope="ui_build_evaluation_only_not_a_release_or_promotion_decision",
        reasons=tuple(sorted(set(reasons))),
        clicks_that_hurt=(),
        measured_comparison={},
        p2_04_summary=summary,
        label=SYNTHETIC_LABEL,
        usability_status="NOT DONE",
        usability_acceptance=USABILITY_ACCEPTANCE,
        promotion="not_evaluated",
        alias_threshold_change="none",
    )


def decide_go_no_go(
    report: Mapping[str, Any], measured: Mapping[str, Any] | None = None
) -> GoNoGoReport:
    """Decide whether a custom UI is worth evaluating, using measurements only.

    Args:
        report: The read-only P2-04 internal metrics report object.
        measured: Optional independently supplied comparison with real-reviewer
            time, click and edit-cost measurements for both tools.

    Returns:
        A fail-closed ``no_go`` when evidence is synthetic, absent or subjective;
        otherwise ``go`` only when custom time, clicks and edit cost are all no
        worse than the adapter baseline, and ``no_go_measured_regression`` if
        any is worse. The usability acceptance criterion stays OPEN regardless.

    """
    summary: dict[str, Any] = {}
    reasons: list[str] = []
    try:
        assert_no_sensitive_text(report)
        if measured is not None:
            assert_no_sensitive_text(measured)
    except UiValidationError:
        return _fail(["sensitive_input_rejected"], summary)
    if SUBJECTIVE_KEYS & _walk_mapping_keys(report) or (
        measured is not None and SUBJECTIVE_KEYS & _walk_mapping_keys(measured)
    ):
        reasons.append("subjective_input_rejected")
    provenance = report.get("provenance")
    scope = report.get("scope")
    if not isinstance(provenance, Mapping) or not isinstance(scope, Mapping):
        reasons.append("p2_04_report_shape_invalid")
        provenance = {}
        scope = {}
    if report.get("artifact_type") != "voiceloop.review.metrics":
        reasons.append("p2_04_report_shape_invalid")
    if report.get("report_schema_version") != "internal-1.0.0":
        reasons.append("p2_04_report_shape_invalid")
    if provenance.get("real_reviewers_used") is not True:
        reasons.append("p2_04_synthetic_fixture_stand_in")
        reasons.append("no_real_reviewed_metrics")
    if "SYNTHETIC" in str(report.get("status", "")):
        reasons.append("p2_04_synthetic_fixture_stand_in")
    limitations = report.get("limitations")
    if (
        isinstance(limitations, (list, tuple))
        and "no_clickstream_or_ui_timing_measured" in limitations
    ):
        reasons.append("no_clickstream_or_ui_timing_measured")
    if provenance.get("fixture_sha256") != EXPECTED_FIXTURE_SHA256:
        reasons.append("p2_04_fixture_hash_mismatch")
    if provenance.get("golden_content_hash") != EXPECTED_GOLDEN_CONTENT_HASH:
        reasons.append("p2_04_golden_hash_mismatch")
    sections = report.get("sections")
    per_item = sections.get("per_item") if isinstance(sections, Mapping) else None
    if isinstance(per_item, Mapping):
        summary = {
            "time_summary": dict(per_item.get("time_summary", {})),
            "edit_distance_summary": dict(per_item.get("edit_distance_summary", {})),
            "routing_label_distribution": dict(per_item.get("routing_label_distribution", {})),
        }
    else:
        reasons.append("p2_04_report_shape_invalid")
    if reasons:
        return _fail(reasons, summary)
    if measured is None:
        return _fail(
            ["no_measured_click_time_edit_comparison", "no_real_reviewed_metrics"],
            summary,
        )
    if SUBJECTIVE_KEYS & _walk_mapping_keys(measured):
        return _fail(["subjective_input_rejected"], summary)
    if measured.get("real_reviewers_used") is not True:
        return _fail(["measured_comparison_not_real_reviewed"], summary)
    if "SYNTHETIC" in str(measured.get("source", "")):
        return _fail(["measured_comparison_synthetic"], summary)
    items = measured.get("items")
    if isinstance(items, bool) or not isinstance(items, int) or items <= 0:
        return _fail(["insufficient_measured_items"], summary)
    tools = measured.get("tools")
    if not isinstance(tools, Mapping) or set(tools) != {"adapter_baseline", "custom_ui"}:
        return _fail(["measured_tools_incomplete"], summary)
    try:
        adapter = _measurement(tools["adapter_baseline"])
        custom = _measurement(tools["custom_ui"])
    except UiValidationError as exc:
        return _fail([str(exc)], summary)
    deltas = {
        "time_per_item_seconds": custom.time_per_item_seconds - adapter.time_per_item_seconds,
        "clicks_per_item": custom.clicks_per_item - adapter.clicks_per_item,
        "edit_cost_per_item": custom.edit_cost_per_item - adapter.edit_cost_per_item,
    }
    hurts: list[dict[str, Any]] = []
    for action in sorted(CLICK_ACTIONS):
        if action not in custom.click_counts or action not in adapter.click_counts:
            return _fail([f"missing_click_measurement_{action}"], summary)
        if (
            action not in custom.click_seconds_by_action
            or action not in adapter.click_seconds_by_action
        ):
            return _fail([f"missing_click_seconds_{action}"], summary)
        extra_clicks = custom.click_counts[action] - adapter.click_counts[action]
        extra_seconds = (
            custom.click_seconds_by_action[action] - adapter.click_seconds_by_action[action]
        )
        if extra_clicks > 0 or extra_seconds > 0:
            hurts.append(
                {
                    "action": action,
                    "extra_clicks": round(extra_clicks, 6),
                    "extra_seconds": round(extra_seconds, 6),
                }
            )
    hurts.sort(key=lambda row: (-row["extra_seconds"], row["action"]))
    reasons.extend(
        name + "_worse"
        for name, delta in deltas.items()
        if delta > 0 and name + "_worse" not in reasons
    )
    comparison = {
        "items": items,
        "source": str(measured.get("source")),
        "real_reviewers_used": True,
        "adapter_baseline": adapter.to_dict(),
        "custom_ui": custom.to_dict(),
        "deltas_custom_minus_adapter": {key: round(value, 6) for key, value in deltas.items()},
    }
    decision = "no_go_measured_regression" if reasons else "go"
    if not reasons:
        reasons.append("measured_no_worse_than_adapter_baseline")
    return GoNoGoReport(
        decision=decision,
        decision_scope="ui_build_evaluation_only_not_a_release_or_promotion_decision",
        reasons=tuple(sorted(set(reasons))),
        clicks_that_hurt=tuple(hurts),
        measured_comparison=comparison,
        p2_04_summary=summary,
        label=SYNTHETIC_LABEL,
        usability_status="NOT DONE",
        usability_acceptance=USABILITY_ACCEPTANCE,
        promotion="not_evaluated",
        alias_threshold_change="none",
    )


def read_json(path: Path) -> dict[str, Any]:
    """Read a JSON object from a read-only input file."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise UiValidationError(f"expected a JSON object in {path}")
    return payload


def write_json_report(report: Mapping[str, Any], path: Path) -> str:
    """Write a deterministic, sorted JSON report to ``path``.

    Args:
        report: Report object from :func:`decide_go_no_go`.
        path: Destination inside the UI lane (or a test temporary directory).

    Returns:
        The destination path as a string.

    """
    assert_no_sensitive_text(report)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return str(path)


def decide_from_files(
    metrics_path: Path, measured_path: Path | None = None, output_path: Path | None = None
) -> GoNoGoReport:
    """Read P2-04 metrics (and optional measurements) and persist a decision.

    Args:
        metrics_path: Read-only P2-04 report path.
        measured_path: Optional real-evidence comparison path.
        output_path: Optional destination; no file is written when omitted.

    Returns:
        The deterministic go/no-go report.

    """
    report = decide_go_no_go(
        read_json(metrics_path),
        read_json(measured_path) if measured_path is not None else None,
    )
    if output_path is not None:
        write_json_report(report.to_dict(), output_path)
    return report
